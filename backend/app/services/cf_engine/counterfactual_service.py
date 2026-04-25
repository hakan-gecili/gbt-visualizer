from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

import pandas as pd

from .example_loader import BoosterModelAdapter, LoadedExample, load_example
from .lgbm_counterfactual import LightGBMCounterfactualExplainer
from .moves_lookup import build_moves_lookup
from .schema_adapter import (
    booster_category_maps,
    build_projection_from_schema,
    category_decoders as build_category_decoders,
    encode_dataset_for_model,
)
from .schema_adapter import decode_change


def _prediction_reaches_target(probability: float, target_class: int, decision_threshold: float) -> bool:
    if int(target_class) == 1:
        return float(probability) >= float(decision_threshold)
    return float(probability) < float(decision_threshold)


def _onehot_group_for_feature(projection_layer: Any, feature: str) -> Optional[str]:
    col_to_group = getattr(projection_layer, "onehot_col_to_group", None)
    if not isinstance(col_to_group, dict):
        return None
    group = col_to_group.get(feature)
    return str(group) if group is not None else None


def _atomic_change_units(changes: List[Dict[str, Any]], projection_layer: Any = None) -> List[Dict[str, Any]]:
    onehot_groups: Dict[str, Dict[str, Any]] = {}
    units: List[Dict[str, Any]] = []

    for change in changes:
        feature = str(change["feature"])
        group = _onehot_group_for_feature(projection_layer, feature)
        if group is None:
            units.append({"key": ("feature", feature), "changes": [change]})
            continue

        unit = onehot_groups.get(group)
        if unit is None:
            unit = {"key": ("onehot_group", group), "changes": []}
            onehot_groups[group] = unit
            units.append(unit)
        unit["changes"].append(change)

    return units


def _apply_change_units(original_row: pd.DataFrame, units: List[Dict[str, Any]]) -> pd.DataFrame:
    candidate = original_row.copy(deep=True)
    row_index = candidate.index[0]
    for unit in units:
        for change in unit["changes"]:
            candidate.loc[row_index, change["feature"]] = change["new_value"]
    return candidate


def prune_counterfactual_changes(
    original_row: pd.DataFrame,
    changes: List[Dict[str, Any]],
    target_class: int,
    decision_threshold: float,
    predict_fn: Callable[[pd.DataFrame], float],
    projection_layer: Any = None,
    max_passes: int = 5,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Remove redundant counterfactual edits using full-model predictions."""
    if not changes:
        return original_row.copy(deep=True), [], [], True

    active_units = _atomic_change_units(changes, projection_layer)
    candidate = _apply_change_units(original_row, active_units)
    probability = float(predict_fn(candidate))
    if not _prediction_reaches_target(probability, int(target_class), float(decision_threshold)):
        return candidate, list(changes), [], False

    removed_changes: List[Dict[str, Any]] = []
    for _ in range(max(1, int(max_passes))):
        removed_this_pass = False
        index = 0
        while index < len(active_units):
            trial_units = active_units[:index] + active_units[index + 1 :]
            trial = _apply_change_units(original_row, trial_units)
            trial_probability = float(predict_fn(trial))
            if _prediction_reaches_target(trial_probability, int(target_class), float(decision_threshold)):
                removed_changes.extend(active_units[index]["changes"])
                active_units = trial_units
                removed_this_pass = True
                continue
            index += 1

        if not removed_this_pass:
            break

    pruned = _apply_change_units(original_row, active_units)
    pruned_probability = float(predict_fn(pruned))
    is_minimal = _prediction_reaches_target(pruned_probability, int(target_class), float(decision_threshold))
    if is_minimal:
        for index in range(len(active_units)):
            trial_units = active_units[:index] + active_units[index + 1 :]
            trial_probability = float(predict_fn(_apply_change_units(original_row, trial_units)))
            if _prediction_reaches_target(trial_probability, int(target_class), float(decision_threshold)):
                is_minimal = False
                break

    pruned_changes = [change for unit in active_units for change in unit["changes"]]
    return pruned, pruned_changes, removed_changes, is_minimal


@dataclass
class CounterfactualService:
    model: Any
    dataset: pd.DataFrame
    schema: Dict[str, Any]
    lookup: Dict[Any, Any]
    projection: Any
    feature_names: List[str]
    category_decoders: Dict[str, Dict[float, Any]]

    def __post_init__(self) -> None:
        self.explainer = LightGBMCounterfactualExplainer(
            self.model,
            projection=self.projection,
            moves_lookup=self.lookup,
        )

    @classmethod
    def from_loaded_example(cls, example: LoadedExample) -> "CounterfactualService":
        return cls(
            model=example.model,
            dataset=example.model_dataset,
            schema=example.schema,
            lookup=example.lookup,
            projection=example.projection,
            feature_names=example.feature_names,
            category_decoders=example.category_decoders,
        )

    def generate_counterfactual_for_row(
        self,
        row_index: int,
        threshold: float,
        target_class: Optional[int] = None,
        max_steps: int = 3,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        if not 0 <= int(row_index) < len(self.dataset):
            raise IndexError(f"row_index out of range; dataset has {len(self.dataset)} rows")
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("threshold must be between 0 and 1")

        x_row = self.dataset.iloc[[int(row_index)]].copy()
        p0 = float(self.model.predict_proba(x_row)[0, 1])
        current_prediction = int(p0 >= float(threshold))
        target = int(1 - current_prediction if target_class is None else target_class)

        x_cf, steps = self.explainer.greedy_counterfactual(
            x_row,
            thr_proba=float(threshold),
            target_label=target,
            max_steps=int(max_steps),
            exact_eval_top_n=50,
        )

        counterfactuals: List[Dict[str, Any]] = []
        if not steps.empty:
            original = x_row.iloc[0]
            final = x_cf.iloc[0]
            raw_changes = []
            for feature in self.feature_names:
                old_value = original[feature]
                new_value = final[feature]
                if old_value != new_value:
                    raw_changes.append({"feature": feature, "old_value": old_value, "new_value": new_value})

            x_pruned, pruned_raw_changes, removed_raw_changes, is_minimal = prune_counterfactual_changes(
                original_row=x_row,
                changes=raw_changes,
                target_class=target,
                decision_threshold=float(threshold),
                predict_fn=lambda row: float(self.model.predict_proba(row)[0, 1]),
                projection_layer=self.projection,
            )
            p1 = float(self.model.predict_proba(x_pruned)[0, 1])
            pred1 = int(p1 >= float(threshold))
            if pred1 != target:
                return {
                    "current_probability": p0,
                    "current_prediction": current_prediction,
                    "threshold": float(threshold),
                    "counterfactuals": counterfactuals,
                    "runtime_ms": 1000.0 * (time.perf_counter() - started),
                }

            changes = [
                decode_change(
                    self.schema,
                    self.category_decoders,
                    str(change["feature"]),
                    change["old_value"],
                    change["new_value"],
                )
                for change in pruned_raw_changes
            ]
            removed_changes = [
                decode_change(
                    self.schema,
                    self.category_decoders,
                    str(change["feature"]),
                    change["old_value"],
                    change["new_value"],
                )
                for change in removed_raw_changes
            ]

            counterfactuals.append(
                {
                    "new_probability": p1,
                    "new_prediction": pred1,
                    "cost": float(steps["cost"].sum()),
                    "changes": changes,
                    "steps": steps.to_dict(orient="records"),
                    "original_num_changes": len(raw_changes),
                    "pruned_num_changes": len(pruned_raw_changes),
                    "removed_changes": removed_changes,
                    "is_minimal_after_pruning": bool(is_minimal),
                }
            )

        return {
            "current_probability": p0,
            "current_prediction": current_prediction,
            "threshold": float(threshold),
            "counterfactuals": counterfactuals,
            "runtime_ms": 1000.0 * (time.perf_counter() - started),
        }


def build_counterfactual_engine(
    model: Any,
    dataset: pd.DataFrame,
    schema: Dict[str, Any],
    lookup: Optional[Dict[Any, Any]] = None,
    projection: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    category_decoders: Optional[Dict[str, Dict[float, Any]]] = None,
) -> CounterfactualService:
    booster = model.booster_ if hasattr(model, "booster_") else model
    wrapped_model = model if hasattr(model, "predict_proba") and hasattr(model, "booster_") else BoosterModelAdapter(booster)

    pandas_categorical = getattr(booster, "pandas_categorical", None)
    if hasattr(booster, "pandas_categorical"):
        booster.pandas_categorical = None

    names = feature_names or list(booster.feature_name())
    cat_maps = booster_category_maps(schema, names, pandas_categorical)
    model_dataset = encode_dataset_for_model(dataset, schema, names, cat_maps)
    resolved_lookup = lookup if lookup is not None else build_moves_lookup(booster)
    resolved_projection = projection or build_projection_from_schema(
        schema,
        model_dataset,
        names,
        {name: sorted(set(mapping.values())) for name, mapping in cat_maps.items()},
    )

    return CounterfactualService(
        model=wrapped_model,
        dataset=model_dataset,
        schema=schema,
        lookup=resolved_lookup,
        projection=resolved_projection,
        feature_names=names,
        category_decoders=category_decoders or build_category_decoders(cat_maps),
    )


def generate_counterfactual_for_row(
    example_path: Path | str,
    row_index: int,
    threshold: float,
    target_class: Optional[int] = None,
    max_steps: int = 3,
) -> Dict[str, Any]:
    example = load_example(example_path)
    service = CounterfactualService.from_loaded_example(example)
    return service.generate_counterfactual_for_row(
        row_index=row_index,
        threshold=threshold,
        target_class=target_class,
        max_steps=max_steps,
    )
