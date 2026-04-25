from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re

import numpy as np
import pandas as pd

from .projection import CategoricalSpec, IntSpec, ProjectionEngine


TARGET_CANDIDATES = ("target", "label", "class", "survived", "y")


def normalize_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()


def source_column(df: pd.DataFrame, feature_name: str) -> Optional[str]:
    if feature_name in df.columns:
        return feature_name
    normalized_feature = normalize_name(feature_name)
    for col in df.columns:
        if normalize_name(col) == normalized_feature:
            return str(col)
    return None


def target_column(df: pd.DataFrame, schema: Dict[str, Any]) -> Optional[str]:
    target = schema.get("target_column")
    if target in df.columns:
        return str(target)
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    return None


def feature_by_name(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(f["name"]): dict(f) for f in schema.get("features", []) if "name" in f}


def infer_schema(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    features = []
    for name in feature_names:
        src = source_column(df, name)
        if src is None:
            continue
        values = pd.to_numeric(df[src], errors="coerce")
        features.append(
            {
                "name": name,
                "type": "continuous",
                "missing_allowed": bool(values.isna().any()),
                "min_value": None if values.dropna().empty else float(values.min()),
                "max_value": None if values.dropna().empty else float(values.max()),
                "default_value": None if values.dropna().empty else float(values.median()),
                "options": [],
            }
        )
    return {"features": features}


def load_schema(schema_path: Optional[Path], df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    if schema_path is None:
        return infer_schema(df, feature_names)
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    schema.setdefault("features", [])
    return schema


def option_maps(feature: Dict[str, Any]) -> Dict[Any, float]:
    mapping: Dict[Any, float] = {}
    for opt in feature.get("options", []) or []:
        if "encoded_value" not in opt:
            continue
        encoded = float(opt["encoded_value"])
        mapping[opt.get("value")] = encoded
        mapping[str(opt.get("value"))] = encoded
        mapping[opt.get("label")] = encoded
        mapping[str(opt.get("label"))] = encoded
        mapping[encoded] = encoded
    return mapping


def booster_category_maps(
    schema: Dict[str, Any],
    feature_names: List[str],
    pandas_categorical: Optional[List[List[Any]]],
) -> Dict[str, Dict[Any, float]]:
    if not pandas_categorical:
        return {}

    features = feature_by_name(schema)
    maps: Dict[str, Dict[Any, float]] = {}
    cat_idx = 0
    for name in feature_names:
        feature = features.get(name, {})
        if str(feature.get("type", "")).lower() != "categorical":
            continue
        if cat_idx >= len(pandas_categorical):
            break
        categories = list(pandas_categorical[cat_idx])
        maps[name] = {value: float(i) for i, value in enumerate(categories)}
        maps[name].update({str(value): float(i) for i, value in enumerate(categories)})
        cat_idx += 1
    return maps


def encode_feature_series(
    raw: pd.Series,
    feature: Optional[Dict[str, Any]],
    booster_category_map: Optional[Dict[Any, float]] = None,
) -> pd.Series:
    if feature is None:
        return pd.to_numeric(raw, errors="coerce")

    ftype = str(feature.get("type", "continuous")).lower()
    if ftype in {"categorical", "binary"}:
        mapping = booster_category_map or option_maps(feature)
        encoded = raw.map(mapping)
        numeric_fallback = pd.to_numeric(raw, errors="coerce")
        encoded = encoded.where(encoded.notna(), numeric_fallback)
        default = feature.get("default_value")
        if encoded.isna().any() and default is not None:
            default_encoded = mapping.get(default, mapping.get(str(default)))
            if default_encoded is not None:
                encoded = encoded.fillna(float(default_encoded))
        return encoded.astype(float)

    numeric = pd.to_numeric(raw, errors="coerce")
    default = feature.get("default_value")
    if numeric.isna().any() and default is not None:
        try:
            numeric = numeric.fillna(float(default))
        except (TypeError, ValueError):
            pass
    return numeric


def encode_dataset_for_model(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    feature_names: List[str],
    category_maps: Optional[Dict[str, Dict[Any, float]]] = None,
) -> pd.DataFrame:
    features = feature_by_name(schema)
    category_maps = category_maps or {}
    encoded = pd.DataFrame(index=df.index)
    for name in feature_names:
        src = source_column(df, name)
        if src is None:
            raise ValueError(f"Dataset is missing model feature {name!r}")
        encoded[name] = encode_feature_series(df[src], features.get(name), category_maps.get(name))
    return encoded[feature_names].astype(float)


def build_projection_from_schema(
    schema: Dict[str, Any],
    model_df: pd.DataFrame,
    feature_names: List[str],
    categorical_encoded_values: Optional[Dict[str, List[float]]] = None,
) -> ProjectionEngine:
    features = feature_by_name(schema)
    int_specs: Dict[str, IntSpec] = {}
    categorical_specs: Dict[str, CategoricalSpec] = {}
    feature_scales: Dict[str, float] = {}
    categorical_encoded_values = categorical_encoded_values or {}

    for name in feature_names:
        series = pd.to_numeric(model_df[name], errors="coerce")
        values = series.dropna().to_numpy(dtype=float)
        std = float(np.std(values)) if values.size else 0.0
        feature_scales[name] = std if std > 1e-12 else 1.0

        feature = features.get(name, {})
        ftype = str(feature.get("type", "continuous")).lower()
        min_value = feature.get("min_value")
        max_value = feature.get("max_value")

        if ftype == "integer":
            int_specs[name] = IntSpec(
                min_val=None if min_value is None else int(np.floor(float(min_value))),
                max_val=None if max_value is None else int(np.ceil(float(max_value))),
                step=int(feature.get("step", 1) or 1),
                weight=float(feature.get("weight", 1.0) or 1.0),
            )
        elif ftype in {"categorical", "binary"}:
            values = categorical_encoded_values.get(name) or [
                float(opt["encoded_value"]) for opt in feature.get("options", []) or [] if "encoded_value" in opt
            ]
            if not values:
                values = sorted(float(v) for v in series.dropna().unique())
            categorical_specs[name] = CategoricalSpec(
                encoded_values=values,
                switch_cost=float(feature.get("switch_cost", 1.0) or 1.0),
                immutable=bool(feature.get("immutable", False)),
            )

    return ProjectionEngine(
        feature_scales=feature_scales,
        int_specs=int_specs,
        categorical_specs=categorical_specs,
        immutable_features=[name for name, f in features.items() if bool(f.get("immutable", False)) and name in feature_names],
    )


def category_decoders(category_maps: Dict[str, Dict[Any, float]]) -> Dict[str, Dict[float, Any]]:
    return {
        name: {float(code): value for value, code in mapping.items() if not isinstance(value, float)}
        for name, mapping in category_maps.items()
    }


def decode_change(
    schema: Dict[str, Any],
    decoders: Dict[str, Dict[float, Any]],
    feature: str,
    old_value: Any,
    new_value: Any,
) -> Dict[str, Any]:
    feature_meta = feature_by_name(schema).get(feature, {})
    result = {"feature": feature, "old_value": old_value, "new_value": new_value}
    if str(feature_meta.get("type", "")).lower() not in {"categorical", "binary"}:
        return result

    reverse = decoders.get(feature) or {
        float(opt["encoded_value"]): opt.get("value", opt.get("label"))
        for opt in feature_meta.get("options", []) or []
        if "encoded_value" in opt
    }
    try:
        result["old_value"] = reverse.get(float(old_value), old_value)
    except (TypeError, ValueError):
        pass
    try:
        result["new_value"] = reverse.get(float(new_value), new_value)
    except (TypeError, ValueError):
        pass
    return result
