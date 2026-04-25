from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .moves_lookup import build_moves_lookup
from .schema_adapter import (
    booster_category_maps,
    build_projection_from_schema,
    category_decoders,
    encode_dataset_for_model,
    load_schema,
    target_column,
)


DEFAULT_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class BoosterModelAdapter:
    """Adapter exposing the LGBMClassifier-like methods used by the explainer."""

    def __init__(self, booster: lgb.Booster):
        self.booster_ = booster

    def predict(self, X: pd.DataFrame, raw_score: bool = False, pred_contrib: bool = False, pred_leaf: bool = False):
        return self.booster_.predict(X, raw_score=raw_score, pred_contrib=pred_contrib, pred_leaf=pred_leaf)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.asarray(self.booster_.predict(X), dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class ExampleConfig:
    example_id: str
    path: Path
    model_path: Path
    dataset_path: Path
    schema_path: Optional[Path]


@dataclass
class LoadedExample:
    example_id: str
    path: Path
    model: BoosterModelAdapter
    dataset: pd.DataFrame
    model_dataset: pd.DataFrame
    schema: Dict[str, Any]
    lookup: Dict[Any, Any]
    projection: Any
    target_column: Optional[str]
    feature_names: List[str]
    category_decoders: Dict[str, Dict[float, Any]]
    loaded_at: float


_EXAMPLE_CACHE: Dict[str, LoadedExample] = {}


def _find_one(folder: Path, suffix: str) -> Optional[Path]:
    matches = sorted(folder.glob(f"*{suffix}"))
    return matches[0] if matches else None


def scan_examples(examples_dir: Path | str = DEFAULT_EXAMPLES_DIR) -> List[Dict[str, Any]]:
    root = Path(examples_dir)
    configs: List[Dict[str, Any]] = []
    if not root.exists():
        return configs

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        model_path = _find_one(folder, ".txt")
        dataset_path = _find_one(folder, ".csv")
        if model_path is None or dataset_path is None:
            continue
        schema_path = _find_one(folder, ".json")
        configs.append(
            {
                "example_id": folder.name,
                "name": folder.name.replace("_", " ").title(),
                "path": str(folder),
                "model_path": str(model_path),
                "dataset_path": str(dataset_path),
                "schema_path": str(schema_path) if schema_path else None,
                "has_schema": schema_path is not None,
            }
        )
    return configs


def _resolve_example_path(example_path: Path | str) -> ExampleConfig:
    path = Path(example_path)
    if not path.exists():
        matches = [cfg for cfg in scan_examples() if cfg["example_id"] == str(example_path)]
        if not matches:
            matches = [cfg for cfg in scan_examples() if cfg["example_id"].startswith(str(example_path))]
        if not matches:
            raise FileNotFoundError(f"Example path or id not found: {example_path}")
        if len(matches) > 1:
            ids = [m["example_id"] for m in matches]
            raise ValueError(f"Ambiguous example id {example_path!r}; matches={ids}")
        path = Path(matches[0]["path"])

    if not path.is_dir():
        raise ValueError(f"Example path must be a directory: {path}")

    model_path = _find_one(path, ".txt")
    dataset_path = _find_one(path, ".csv")
    if model_path is None or dataset_path is None:
        raise FileNotFoundError(f"Example {path} must contain a .txt model and .csv dataset")

    return ExampleConfig(
        example_id=path.name,
        path=path,
        model_path=model_path,
        dataset_path=dataset_path,
        schema_path=_find_one(path, ".json"),
    )


def load_example(example_path: Path | str, force_reload: bool = False) -> LoadedExample:
    cfg = _resolve_example_path(example_path)
    cache_key = str(cfg.path.resolve())
    if not force_reload and cache_key in _EXAMPLE_CACHE:
        return _EXAMPLE_CACHE[cache_key]

    booster = lgb.Booster(model_file=str(cfg.model_path))
    pandas_categorical = getattr(booster, "pandas_categorical", None)
    booster.pandas_categorical = None
    model = BoosterModelAdapter(booster)
    feature_names = list(booster.feature_name())

    dataset = pd.read_csv(cfg.dataset_path)
    schema = load_schema(cfg.schema_path, dataset, feature_names)
    cat_maps = booster_category_maps(schema, feature_names, pandas_categorical)
    model_dataset = encode_dataset_for_model(dataset, schema, feature_names, cat_maps)
    lookup = build_moves_lookup(booster)
    projection = build_projection_from_schema(
        schema,
        model_dataset,
        feature_names,
        {name: sorted(set(mapping.values())) for name, mapping in cat_maps.items()},
    )

    loaded = LoadedExample(
        example_id=cfg.example_id,
        path=cfg.path,
        model=model,
        dataset=dataset,
        model_dataset=model_dataset,
        schema=schema,
        lookup=lookup,
        projection=projection,
        target_column=target_column(dataset, schema),
        feature_names=feature_names,
        category_decoders=category_decoders(cat_maps),
        loaded_at=time.time(),
    )
    _EXAMPLE_CACHE[cache_key] = loaded
    return loaded


def clear_example_cache() -> None:
    _EXAMPLE_CACHE.clear()
