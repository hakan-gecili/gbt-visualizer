from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

FEATURE_TYPE_NUMERIC = "numeric"
FEATURE_TYPE_BINARY = "binary"
FEATURE_TYPE_CATEGORICAL = "categorical"
SUPPORTED_FEATURE_TYPES = {
    FEATURE_TYPE_NUMERIC,
    FEATURE_TYPE_BINARY,
    FEATURE_TYPE_CATEGORICAL,
}


class SchemaGenerationError(ValueError):
    pass


def load_overrides(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    override_path = Path(path)
    try:
        with override_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise SchemaGenerationError(f"Overrides file was not found: {override_path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaGenerationError(f"Overrides file is not valid JSON: {override_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise SchemaGenerationError("Overrides file must contain a top-level JSON object.")

    features = payload.get("features")
    columns = payload.get("columns")
    if features is not None and columns is not None:
        raise SchemaGenerationError("Overrides file may define either 'features' or 'columns', but not both.")
    resolved_features = features if features is not None else columns
    if resolved_features is None:
        payload["features"] = {}
    elif not isinstance(resolved_features, dict):
        raise SchemaGenerationError("Overrides field 'features' or 'columns' must be an object keyed by column name.")
    else:
        payload["features"] = resolved_features

    exclude_columns = payload.get("exclude_columns", [])
    if exclude_columns is None:
        payload["exclude_columns"] = []
    elif not isinstance(exclude_columns, list):
        raise SchemaGenerationError("Overrides field 'exclude_columns' must be a list.")

    return payload


def resolve_target_column(cli_target: str | None, overrides: dict[str, Any]) -> str | None:
    target_column = cli_target or overrides.get("target_column")
    if target_column is None:
        return None
    return str(target_column)


def infer_feature_type(series: pd.Series, feature_override: dict[str, Any]) -> str:
    forced_type = feature_override.get("type")
    if forced_type is not None:
        feature_type = str(forced_type).strip().lower()
        if feature_type not in SUPPORTED_FEATURE_TYPES:
            raise SchemaGenerationError(
                f"Unsupported override type '{feature_type}' for column '{series.name}'."
            )
        return feature_type

    non_null = series.dropna()
    unique_values = unique_non_null_values(non_null)
    if len(unique_values) == 2:
        return FEATURE_TYPE_BINARY
    if is_bool_dtype(series) or is_numeric_dtype(series):
        return FEATURE_TYPE_NUMERIC
    return FEATURE_TYPE_CATEGORICAL


def infer_feature_schema(series: pd.Series, feature_override: dict[str, Any]) -> dict[str, Any]:
    feature_name = str(series.name)
    feature_type = infer_feature_type(series, feature_override)
    missing_allowed = bool(series.isna().any())
    if "missing_allowed" in feature_override:
        missing_allowed = bool(feature_override["missing_allowed"])

    base_feature = {
        "name": str(feature_override.get("name", feature_name)),
        "short_name": str(feature_override.get("short_name", feature_name[:10])),
        "type": feature_type,
        "missing_allowed": missing_allowed,
    }

    if feature_type == FEATURE_TYPE_NUMERIC:
        base_feature.update(build_numeric_feature(series, feature_override))
        return base_feature

    base_feature.update(build_option_feature(series, feature_type, feature_override))
    return base_feature


def build_numeric_feature(series: pd.Series, feature_override: dict[str, Any]) -> dict[str, Any]:
    feature_name = str(series.name)
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        raise SchemaGenerationError(f"Column '{feature_name}' cannot be inferred as numeric because it has no values.")

    min_value = float(non_null.min())
    max_value = float(non_null.max())
    default_value = float(non_null.median())

    if "min_value" in feature_override:
        min_value = float(feature_override["min_value"])
    if "max_value" in feature_override:
        max_value = float(feature_override["max_value"])
    if "default_value" in feature_override:
        default_value = float(feature_override["default_value"])

    return {
        "min_value": min_value,
        "max_value": max_value,
        "default_value": default_value,
        "options": [],
    }


def build_option_feature(series: pd.Series, feature_type: str, feature_override: dict[str, Any]) -> dict[str, Any]:
    if "options" in feature_override:
        options = normalize_override_options(feature_override["options"], str(series.name))
    else:
        options = build_options_from_series(series, labels_map=feature_override.get("labels", {}))

    if feature_type == FEATURE_TYPE_BINARY and len(options) != 2:
        raise SchemaGenerationError(
            f"Column '{series.name}' was declared binary but resolved to {len(options)} options instead of 2."
        )
    if len(options) < 2:
        raise SchemaGenerationError(
            f"Column '{series.name}' needs at least two distinct non-null values to build {feature_type} options."
        )

    default_value = infer_default_option(series, options)
    if "default_value" in feature_override:
        default_value = str(feature_override["default_value"])
        valid_values = {item["value"] for item in options}
        if default_value not in valid_values:
            raise SchemaGenerationError(
                f"Override default_value '{default_value}' for column '{series.name}' is not one of {sorted(valid_values)}."
            )

    return {
        "min_value": None,
        "max_value": None,
        "default_value": default_value,
        "options": options,
    }


def build_options_from_series(series: pd.Series, labels_map: Any) -> list[dict[str, Any]]:
    if labels_map is None:
        labels_map = {}
    if not isinstance(labels_map, dict):
        raise SchemaGenerationError(f"Override labels for column '{series.name}' must be an object.")

    values = unique_non_null_values(series.dropna())
    options: list[dict[str, Any]] = []
    for index, raw_value in enumerate(values):
        raw_key = stringify_value(raw_value)
        label = str(labels_map.get(raw_key, raw_key))
        options.append(
            {
                "value": raw_key,
                "label": label,
                "encoded_value": float(index),
            }
        )
    return options


def normalize_override_options(raw_options: Any, feature_name: str) -> list[dict[str, Any]]:
    if not isinstance(raw_options, list):
        raise SchemaGenerationError(f"Override options for column '{feature_name}' must be a list.")

    normalized: list[dict[str, Any]] = []
    seen_values: set[str] = set()
    for index, item in enumerate(raw_options):
        if not isinstance(item, dict):
            raise SchemaGenerationError(
                f"Override option #{index} for column '{feature_name}' must be an object."
            )
        value = str(item.get("value", "")).strip()
        if not value:
            raise SchemaGenerationError(
                f"Override option #{index} for column '{feature_name}' must include a non-empty value."
            )
        if value in seen_values:
            raise SchemaGenerationError(
                f"Override options for column '{feature_name}' contain duplicate value '{value}'."
            )
        label = str(item.get("label", value))
        encoded_value = float(item.get("encoded_value", index))
        normalized.append(
            {
                "value": value,
                "label": label,
                "encoded_value": encoded_value,
            }
        )
        seen_values.add(value)
    return normalized


def infer_default_option(series: pd.Series, options: list[dict[str, Any]]) -> str:
    counts = series.dropna().map(stringify_value).value_counts()
    if counts.empty:
        return options[0]["value"]

    highest_count = int(counts.max())
    top_values = {str(index) for index, count in counts.items() if int(count) == highest_count}
    ordered_option_values = [item["value"] for item in options]
    for option_value in ordered_option_values:
        if option_value in top_values:
            return option_value
    return ordered_option_values[0]


def unique_non_null_values(series: pd.Series) -> list[Any]:
    values = []
    seen: set[tuple[str, str]] = set()
    for value in series.tolist():
        key = (type(value).__name__, stringify_value(value))
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    values.sort(key=sort_key)
    return values


def stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value)


def sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (1, float(value))
    return (2, stringify_value(value).lower())


def should_exclude_column(column_name: str, target_column: str | None, overrides: dict[str, Any]) -> bool:
    if target_column is not None and column_name == target_column:
        return True

    excluded = {str(item) for item in overrides.get("exclude_columns", [])}
    if column_name in excluded:
        return True

    feature_override = overrides.get("features", {}).get(column_name, {})
    return bool(feature_override.get("exclude", False))


def iter_feature_definitions(
    dataframe: pd.DataFrame,
    target_column: str | None,
    overrides: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    feature_overrides = overrides.get("features", {})
    features: list[tuple[str, dict[str, Any]]] = []

    for column_name in sorted(map(str, dataframe.columns)):
        if should_exclude_column(column_name, target_column, overrides):
            continue

        feature_override = feature_overrides.get(column_name, {})
        if feature_override is None:
            feature_override = {}
        if not isinstance(feature_override, dict):
            raise SchemaGenerationError(f"Override for column '{column_name}' must be an object.")

        feature = infer_feature_schema(dataframe[column_name], feature_override)
        features.append((column_name, feature))

    return features


def build_schema(dataframe: pd.DataFrame, target_column: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    return {"features": [feature for _, feature in iter_feature_definitions(dataframe, target_column, overrides)]}
