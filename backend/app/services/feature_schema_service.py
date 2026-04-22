from __future__ import annotations

import json
import math
from typing import Any


FeatureValue = float | str | None

FEATURE_TYPE_NUMERIC = "numeric"
FEATURE_TYPE_BINARY = "binary"
FEATURE_TYPE_CATEGORICAL = "categorical"
SUPPORTED_FEATURE_TYPES = {
    FEATURE_TYPE_NUMERIC,
    FEATURE_TYPE_BINARY,
    FEATURE_TYPE_CATEGORICAL,
}


class FeatureSchemaError(ValueError):
    pass


def parse_feature_schema_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        features = payload.get("features")
    else:
        features = payload
    if not isinstance(features, list):
        raise FeatureSchemaError("Feature schema payload must contain a features list.")
    return features


def parse_feature_schema_json(raw_text: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FeatureSchemaError(f"Feature schema file is not valid JSON: {exc}") from exc
    return parse_feature_schema_payload(payload)


def _default_feature_item(feature_name: str) -> dict[str, Any]:
    return {
        "name": feature_name,
        "short_name": feature_name[:10],
        "type": FEATURE_TYPE_NUMERIC,
        "missing_allowed": True,
        "min_value": None,
        "max_value": None,
        "default_value": 0.0,
        "options": [],
    }


def build_feature_metadata(
    feature_names: list[str],
    schema_overrides: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    metadata_by_name = {feature_name: _default_feature_item(feature_name) for feature_name in feature_names}

    for override in schema_overrides or []:
        feature_name = str(override.get("name", "")).strip()
        if not feature_name:
            raise FeatureSchemaError("Feature schema entries must include a non-empty name.")
        if feature_name not in metadata_by_name:
            raise FeatureSchemaError(f"Feature schema references unknown model feature '{feature_name}'.")

        base_item = metadata_by_name[feature_name]
        next_item = dict(base_item)
        if "short_name" in override:
            next_item["short_name"] = str(override["short_name"]).strip() or base_item["short_name"]
        if "missing_allowed" in override:
            next_item["missing_allowed"] = bool(override["missing_allowed"])

        feature_type = str(override.get("type", base_item["type"])).strip().lower() or FEATURE_TYPE_NUMERIC
        if feature_type not in SUPPORTED_FEATURE_TYPES:
            raise FeatureSchemaError(
                f"Feature '{feature_name}' has unsupported type '{feature_type}'. "
                f"Expected one of {sorted(SUPPORTED_FEATURE_TYPES)}."
            )
        next_item["type"] = feature_type

        if feature_type == FEATURE_TYPE_NUMERIC:
            next_item["min_value"] = _optional_float(override.get("min_value", base_item["min_value"]), "min_value")
            next_item["max_value"] = _optional_float(override.get("max_value", base_item["max_value"]), "max_value")
            default_value = override.get("default_value", base_item["default_value"])
            next_item["default_value"] = _normalize_numeric_value(
                feature_name,
                next_item,
                default_value,
                source="schema default",
            )
            next_item["options"] = []
        else:
            options = _normalize_options(feature_name, override.get("options"))
            if len(options) < 2:
                raise FeatureSchemaError(
                    f"Feature '{feature_name}' must define at least two options for type '{feature_type}'."
                )
            next_item["options"] = options
            next_item["min_value"] = None
            next_item["max_value"] = None
            default_value = override.get("default_value", options[0]["value"])
            next_item["default_value"] = _normalize_option_value(
                feature_name,
                next_item,
                default_value,
                source="schema default",
            )

        metadata_by_name[feature_name] = next_item

    return [metadata_by_name[feature_name] for feature_name in feature_names]


def feature_metadata_by_name(feature_metadata: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in feature_metadata}


def prepare_feature_vector(
    feature_metadata: list[dict[str, Any]],
    raw_feature_vector: dict[str, Any],
) -> dict[str, FeatureValue]:
    prepared: dict[str, FeatureValue] = {}
    for feature in feature_metadata:
        feature_name = str(feature["name"])
        source_value = raw_feature_vector.get(feature_name, feature.get("default_value"))
        prepared[feature_name] = normalize_feature_value(
            feature,
            source_value,
            source="request payload",
        )
    return prepared


def encode_feature_vector(
    feature_metadata: list[dict[str, Any]],
    prepared_feature_vector: dict[str, FeatureValue],
) -> dict[str, float]:
    encoded: dict[str, float] = {}
    for feature in feature_metadata:
        feature_name = str(feature["name"])
        value = prepared_feature_vector.get(feature_name)
        if value is None:
            encoded[feature_name] = math.nan
            continue
        if feature["type"] == FEATURE_TYPE_NUMERIC:
            encoded[feature_name] = float(value)
            continue

        option = _match_option(feature, value)
        if option is None:
            raise FeatureSchemaError(
                f"Feature '{feature_name}' received unsupported value '{value}'. "
                f"Expected one of {[item['value'] for item in feature.get('options', [])]}."
            )
        encoded[feature_name] = float(option["encoded_value"])
    return encoded


def normalize_feature_value(feature: dict[str, Any], value: Any, source: str) -> FeatureValue:
    feature_type = str(feature.get("type", FEATURE_TYPE_NUMERIC))
    feature_name = str(feature["name"])
    if feature_type == FEATURE_TYPE_NUMERIC:
        return _normalize_numeric_value(feature_name, feature, value, source=source)
    if feature_type in {FEATURE_TYPE_BINARY, FEATURE_TYPE_CATEGORICAL}:
        return _normalize_option_value(feature_name, feature, value, source=source)
    raise FeatureSchemaError(f"Feature '{feature_name}' has unsupported type '{feature_type}'.")


def _normalize_numeric_value(feature_name: str, feature: dict[str, Any], value: Any, source: str) -> float | None:
    if _is_missing_value(value):
        if feature.get("missing_allowed", True):
            return None
        raise FeatureSchemaError(f"Feature '{feature_name}' does not allow missing values in {source}.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureSchemaError(
            f"Feature '{feature_name}' expected a numeric value in {source}, received '{value}'."
        ) from exc

    if math.isnan(numeric_value):
        if feature.get("missing_allowed", True):
            return None
        raise FeatureSchemaError(f"Feature '{feature_name}' does not allow missing values in {source}.")

    return numeric_value


def _normalize_option_value(feature_name: str, feature: dict[str, Any], value: Any, source: str) -> str | None:
    if _is_missing_value(value):
        if feature.get("missing_allowed", True):
            return None
        raise FeatureSchemaError(f"Feature '{feature_name}' does not allow missing values in {source}.")

    option = _match_option(feature, value)
    if option is None:
        raise FeatureSchemaError(
            f"Feature '{feature_name}' received unsupported value '{value}' in {source}. "
            f"Expected one of {[item['value'] for item in feature.get('options', [])]}."
        )
    return str(option["value"])


def _normalize_options(feature_name: str, raw_options: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_options, list):
        raise FeatureSchemaError(f"Feature '{feature_name}' must define an options list.")

    normalized: list[dict[str, Any]] = []
    seen_values: set[str] = set()
    for index, item in enumerate(raw_options):
        if not isinstance(item, dict):
            raise FeatureSchemaError(f"Feature '{feature_name}' option #{index} must be an object.")

        value = str(item.get("value", "")).strip()
        label = str(item.get("label", value)).strip() or value
        if not value:
            raise FeatureSchemaError(f"Feature '{feature_name}' option #{index} must include a non-empty value.")
        if value in seen_values:
            raise FeatureSchemaError(f"Feature '{feature_name}' defines duplicate option value '{value}'.")
        encoded_value = _optional_float(item.get("encoded_value"), "encoded_value")
        if encoded_value is None:
            raise FeatureSchemaError(
                f"Feature '{feature_name}' option '{value}' must include an encoded_value."
            )
        normalized.append(
            {
                "value": value,
                "label": label,
                "encoded_value": encoded_value,
            }
        )
        seen_values.add(value)
    return normalized


def _match_option(feature: dict[str, Any], value: Any) -> dict[str, Any] | None:
    if _is_missing_value(value):
        return None

    raw_string = str(value).strip()
    for option in feature.get("options", []):
        if raw_string == str(option["value"]) or raw_string == str(option["label"]):
            return option
        try:
            if float(value) == float(option["encoded_value"]):
                return option
        except (TypeError, ValueError):
            continue
    return None


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureSchemaError(f"Expected {field_name} to be numeric, received '{value}'.") from exc


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float):
        return math.isnan(value)
    return False
