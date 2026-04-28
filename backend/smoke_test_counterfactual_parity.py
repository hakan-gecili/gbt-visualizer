from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.session_store import session_store
from app.domain.model_types import EnsembleModel
from app.domain.session_types import SessionState
from app.services.cf_engine.unified_counterfactual_engine import UnifiedPrediction, generate_unified_counterfactual
from app.services.counterfactual_service import generate_counterfactual_for_session
from app.services.dataset_service import apply_dataset_ranges, extract_feature_vector_from_row, load_dataset_from_path, summarize_dataset
from app.services.feature_schema_service import FeatureValue, build_feature_metadata, parse_feature_schema_json
from app.services.model_loader import load_ensemble_model_from_path
from app.services.prediction_service import predict_model


EXACT_EVAL_TOP_N = 64
MAX_STEPS = 3

DATASET_ROWS = {
    "breast_cancer": [0, 1],
    "titanic": [0, 2],
    "adult_income": [1, 8],
    "bank_marketing": [2, 29],
}


@dataclass
class ReplaySummary:
    probability: float | None
    margin: float | None
    label: int | None


def _normalize_feature_key(feature_name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(feature_name)).strip("_").lower()


def _align_schema_and_dataframe(
    schema_overrides: list[dict[str, Any]] | None,
    dataframe: pd.DataFrame,
    feature_names: list[str],
) -> tuple[list[dict[str, Any]] | None, pd.DataFrame]:
    normalized_names = {_normalize_feature_key(feature_name): feature_name for feature_name in feature_names}
    rename_columns: dict[str, str] = {}
    aligned_schema: list[dict[str, Any]] = []

    for feature in schema_overrides or []:
        next_feature = dict(feature)
        feature_name = str(next_feature.get("name", ""))
        model_feature_name = normalized_names.get(_normalize_feature_key(feature_name), feature_name)
        if model_feature_name != feature_name:
            rename_columns[feature_name] = model_feature_name
            next_feature["name"] = model_feature_name
        aligned_schema.append(next_feature)

    for column in dataframe.columns:
        model_feature_name = normalized_names.get(_normalize_feature_key(str(column)))
        if model_feature_name is not None and str(column) != model_feature_name:
            rename_columns[str(column)] = model_feature_name

    if rename_columns:
        dataframe = dataframe.rename(columns=rename_columns)
    return (aligned_schema if schema_overrides is not None else None), dataframe


def _load_lightgbm_example_session(example_name: str) -> SessionState:
    example_dir = REPO_ROOT / "examples" / example_name / "lightgbm"
    model_path = example_dir / "model.txt"
    dataset_path = example_dir / "dataset.csv"
    schema_path = example_dir / "feature_schema.json"

    model, predictor = load_ensemble_model_from_path(model_path)
    dataframe = load_dataset_from_path(str(dataset_path))
    schema_overrides = parse_feature_schema_json(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else None
    schema_overrides, dataframe = _align_schema_and_dataframe(schema_overrides, dataframe, model.feature_names)
    feature_metadata = apply_dataset_ranges(build_feature_metadata(model.feature_names, schema_overrides), dataframe)
    session = SessionState(
        session_id=f"parity-lightgbm-{example_name}",
        model=model,
        predictor=predictor,
        feature_metadata=feature_metadata,
        dataset_frame=dataframe,
        dataset_summary=summarize_dataset(dataframe, model.feature_names),
    )
    session_store.save(session)
    return session


def _generate_shared_counterfactual(
    *,
    model: EnsembleModel,
    predictor: object | None,
    feature_metadata: list[dict[str, Any]],
    original_vector: dict[str, FeatureValue],
    threshold: float,
    target_class: int,
) -> dict[str, Any]:
    def _predict(vector: dict[str, FeatureValue]) -> UnifiedPrediction:
        _, prediction = predict_model(model, predictor, feature_metadata, vector)
        probability = float(prediction.probability)
        return UnifiedPrediction(
            probability=probability,
            margin=float(prediction.margin),
            label=int(probability >= threshold),
        )

    return generate_unified_counterfactual(
        model=model,
        feature_metadata=feature_metadata,
        original_vector=original_vector,
        prediction_evaluator=_predict,
        threshold=threshold,
        target_class=target_class,
        max_steps=MAX_STEPS,
        exact_eval_top_n=EXACT_EVAL_TOP_N,
        debug_label="lightgbm-parity",
    )


def _assert_result_replays(
    *,
    label: str,
    model: EnsembleModel,
    predictor: object | None,
    feature_metadata: list[dict[str, Any]],
    original_vector: dict[str, FeatureValue],
    result: dict[str, Any],
    target_class: int,
    threshold: float,
) -> ReplaySummary:
    counterfactuals = result.get("counterfactuals", [])
    if not counterfactuals:
        return ReplaySummary(probability=None, margin=None, label=None)

    counterfactual = counterfactuals[0]
    replay_vector = dict(original_vector)
    for change in counterfactual.get("changes", []):
        _assert_feature_value_is_valid(label, feature_metadata, change)
        replay_vector[str(change["feature"])] = change["new_value"]

    _, replay_prediction = predict_model(model, predictor, feature_metadata, replay_vector)
    probability = float(replay_prediction.probability)
    margin = float(replay_prediction.margin)
    replay_label = int(probability >= threshold)
    if replay_label != target_class:
        raise AssertionError(f"{label} returned non-flipping counterfactual: got {replay_label}, expected {target_class}.")
    if replay_label != int(counterfactual["new_prediction"]):
        raise AssertionError(f"{label} replay label does not match returned label.")
    if "new_probability" in counterfactual and not math.isclose(
        probability,
        float(counterfactual["new_probability"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise AssertionError(f"{label} replay probability does not match returned probability.")
    if "new_margin" in counterfactual and not math.isclose(
        margin,
        float(counterfactual["new_margin"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise AssertionError(f"{label} replay margin does not match returned margin.")
    return ReplaySummary(probability=probability, margin=margin, label=replay_label)


def _assert_feature_value_is_valid(
    label: str,
    feature_metadata: list[dict[str, Any]],
    change: dict[str, Any],
) -> None:
    feature_by_name = {str(feature["name"]): feature for feature in feature_metadata}
    feature_name = str(change["feature"])
    feature = feature_by_name.get(feature_name)
    if feature is None:
        raise AssertionError(f"{label} returned unknown feature {feature_name!r}.")

    if str(feature.get("type")) in {"categorical", "binary"}:
        allowed_values = {str(option["value"]) for option in feature.get("options", [])}
        if str(change["new_value"]) not in allowed_values:
            raise AssertionError(
                f"{label} returned invalid categorical value {change['new_value']!r} "
                f"for {feature_name!r}; expected one of {sorted(allowed_values)}."
            )
    if _is_integer_like_feature(feature) and isinstance(change["new_value"], (int, float)):
        if not float(change["new_value"]).is_integer():
            raise AssertionError(f"{label} returned fractional integer-like value {change!r}.")
    if _is_missing_value(change.get("old_value")) and not _is_missing_value(change.get("new_value")):
        raise AssertionError(f"{label} changed missing value for {feature_name!r}; missing values should be preserved.")


def _changed_features(result: dict[str, Any]) -> list[str]:
    counterfactuals = result.get("counterfactuals", [])
    if not counterfactuals:
        return []
    return [str(change["feature"]) for change in counterfactuals[0].get("changes", [])]


def _run_dataset_row(example_name: str, row_index: int) -> bool:
    session = _load_lightgbm_example_session(example_name)
    assert session.dataset_frame is not None
    if row_index >= len(session.dataset_frame):
        return False

    threshold = 0.5
    raw_vector = extract_feature_vector_from_row(session.dataset_frame, row_index, session.feature_metadata)
    original_vector, original_prediction = predict_model(
        session.model,
        session.predictor,
        session.feature_metadata,
        raw_vector,
    )
    target_class = 1 - int(float(original_prediction.probability) >= threshold)

    legacy_started = time.perf_counter()
    legacy_result = generate_counterfactual_for_session(
        session.session_id,
        row_index=row_index,
        threshold=threshold,
        target_class=target_class,
        max_steps=MAX_STEPS,
    )
    legacy_runtime_ms = 1000.0 * (time.perf_counter() - legacy_started)
    shared_result = _generate_shared_counterfactual(
        model=session.model,
        predictor=session.predictor,
        feature_metadata=session.feature_metadata,
        original_vector=original_vector,
        threshold=threshold,
        target_class=target_class,
    )

    legacy_replay = _assert_result_replays(
        label=f"{example_name}[{row_index}] legacy",
        model=session.model,
        predictor=session.predictor,
        feature_metadata=session.feature_metadata,
        original_vector=original_vector,
        result=legacy_result,
        target_class=target_class,
        threshold=threshold,
    )
    shared_replay = _assert_result_replays(
        label=f"{example_name}[{row_index}] shared",
        model=session.model,
        predictor=session.predictor,
        feature_metadata=session.feature_metadata,
        original_vector=original_vector,
        result=shared_result,
        target_class=target_class,
        threshold=threshold,
    )

    legacy_found = bool(legacy_result.get("counterfactuals"))
    shared_found = bool(shared_result.get("counterfactuals"))
    legacy_features = _changed_features(legacy_result)
    shared_features = _changed_features(shared_result)
    diagnostics = shared_result.get("diagnostics", {})
    top_ranked_features = diagnostics.get("top_ranked_features", [])
    ranking_overlap = sorted(set(legacy_features) & set(top_ranked_features))

    print(f"dataset={example_name} row={row_index} target={target_class}")
    print(f"  legacy_found={legacy_found} shared_found={shared_found}")
    print(f"  legacy_changed_features={legacy_features}")
    print(f"  shared_changed_features={shared_features}")
    print(f"  shared_top_ranked_features={top_ranked_features}")
    print(f"  candidate_ranking_overlap={ranking_overlap}")
    print(f"  legacy_replay_probability={legacy_replay.probability} legacy_replay_label={legacy_replay.label}")
    print(f"  shared_replay_probability={shared_replay.probability} shared_replay_label={shared_replay.label}")
    print(f"  legacy_runtime_ms={legacy_runtime_ms:.3f} shared_runtime_ms={shared_result['runtime_ms']:.3f}")
    print(
        f"  shared_candidate_count={diagnostics.get('candidate_count')} "
        f"shared_replay_count={diagnostics.get('replay_count')} "
        f"shared_fallback_used={diagnostics.get('fallback_used')}"
    )

    if legacy_found and not shared_found:
        print(f"  parity_gap=shared_missing_counterfactual")
    if shared_found and not legacy_found:
        print(f"  parity_gap=legacy_missing_counterfactual")
    return legacy_found != shared_found


def _is_integer_like_feature(feature: dict[str, Any]) -> bool:
    name = str(feature.get("name", "")).lower()
    return name.endswith("-num") or name.endswith("_num") or name.endswith(" count") or name.endswith("_count")


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    parity_gaps = 0
    rows_checked = 0
    for example_name, row_indices in DATASET_ROWS.items():
        for row_index in row_indices:
            if _run_dataset_row(example_name, row_index):
                parity_gaps += 1
            rows_checked += 1

    print("parity_rows_checked", rows_checked)
    print("parity_gaps_reported", parity_gaps)
    print("counterfactual_parity_smoke_ok", True)


if __name__ == "__main__":
    main()
