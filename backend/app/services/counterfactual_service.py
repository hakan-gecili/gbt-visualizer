from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from app.core.session_store import session_store
from app.domain.session_types import SessionState
from app.services.cf_engine import build_counterfactual_engine
from app.services.dataset_service import extract_feature_vector_from_row
from app.services.prediction_service import predict_model

_engine_cache: dict[str, Any] = {}


def _serialize_feature_options(options: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for option in options or []:
        serialized.append(
            {
                "value": option.get("value"),
                "label": option.get("label"),
                "encoded_value": option.get("encoded_value"),
            }
        )
    return serialized


def build_counterfactual_schema(feature_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for feature in feature_metadata:
        feature_type = str(feature.get("type", "numeric")).lower()
        normalized_type = "continuous" if feature_type == "numeric" else feature_type
        if normalized_type == "continuous" and _is_integer_like_feature(feature):
            normalized_type = "integer"
        features.append(
            {
                "name": str(feature["name"]),
                "type": normalized_type,
                "missing_allowed": bool(feature.get("missing_allowed", True)),
                "min_value": feature.get("min_value"),
                "max_value": feature.get("max_value"),
                "default_value": feature.get("default_value"),
                "options": _serialize_feature_options(feature.get("options")),
            }
        )
    return {"features": features}


def _is_integer_like_feature(feature: dict[str, Any]) -> bool:
    name = str(feature.get("name", "")).lower()
    if name.endswith("-num") or name.endswith("_num") or name.endswith(" count") or name.endswith("_count"):
        return True

    numeric_values = [
        feature.get("min_value"),
        feature.get("max_value"),
        feature.get("default_value"),
    ]
    present_values = [value for value in numeric_values if value is not None]
    return bool(present_values) and all(_is_integer_value(value) for value in present_values)


def _is_integer_value(value: Any) -> bool:
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _build_engine_cache_key(
    model: Any,
    predictor: Any | None,
    dataset: pd.DataFrame,
    schema: dict[str, Any],
    key: str | None = None,
) -> str:
    if key is not None:
        return key
    schema_digest = hashlib.sha1(
        json.dumps(schema, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return ":".join(
        [
            str(id(predictor) if predictor is not None else id(model)),
            str(id(dataset)),
            schema_digest,
        ]
    )


def _build_session_engine_key(
    session_id: str,
    model: Any,
    predictor: Any | None,
    dataset: pd.DataFrame,
    schema: dict[str, Any],
) -> str:
    return f"{session_id}:{_build_engine_cache_key(model, predictor, dataset, schema)}"


def get_engine(
    model: Any,
    dataset: pd.DataFrame,
    schema: dict[str, Any],
    *,
    predictor: Any | None = None,
    key: str | None = None,
) -> Any:
    cache_key = _build_engine_cache_key(model, predictor, dataset, schema, key=key)
    if cache_key not in _engine_cache:
        engine_model = predictor if predictor is not None else model
        if (
            not hasattr(engine_model, "predict_proba")
            and not hasattr(engine_model, "booster_")
            and not hasattr(engine_model, "predict")
        ):
            raise TypeError("Counterfactual engine requires a LightGBM predictor with predict_proba() or booster_.")
        _engine_cache[cache_key] = build_counterfactual_engine(
            model=engine_model,
            dataset=dataset,
            schema=schema,
        )
    return _engine_cache[cache_key]


def generate_counterfactual(
    model: Any,
    dataset: pd.DataFrame,
    schema: dict[str, Any],
    row_index: int,
    threshold: float,
    target_class: int | None,
    max_steps: int = 3,
    *,
    predictor: Any | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    engine = get_engine(
        model=model,
        dataset=dataset,
        schema=schema,
        predictor=predictor,
        key=key,
    )
    return engine.generate_counterfactual_for_row(
        row_index=row_index,
        threshold=threshold,
        target_class=target_class,
        max_steps=max_steps,
    )


def _ensure_session_dataset(session: SessionState) -> pd.DataFrame:
    if session.dataset_frame is None:
        raise ValueError("No dataset has been uploaded for this session.")
    return session.dataset_frame


def get_session_counterfactual_engine(session_id: str) -> Any:
    session = session_store.get(session_id)
    dataset = _ensure_session_dataset(session)
    schema = build_counterfactual_schema(session.feature_metadata)
    return get_engine(
        model=session.model,
        predictor=session.predictor,
        dataset=dataset,
        schema=schema,
        key=_build_session_engine_key(session_id, session.model, session.predictor, dataset, schema),
    )


def generate_counterfactual_for_session(
    session_id: str,
    *,
    row_index: int,
    threshold: float,
    target_class: int | None,
    max_steps: int = 3,
) -> dict[str, Any]:
    session = session_store.get(session_id)
    dataset = _ensure_session_dataset(session)
    schema = build_counterfactual_schema(session.feature_metadata)
    result = generate_counterfactual(
        model=session.model,
        predictor=session.predictor,
        dataset=dataset,
        schema=schema,
        row_index=row_index,
        threshold=threshold,
        target_class=target_class,
        max_steps=max_steps,
        key=_build_session_engine_key(session_id, session.model, session.predictor, dataset, schema),
    )
    return _validate_counterfactual_result_with_app_prediction(
        session=session,
        dataset=dataset,
        result=result,
        row_index=row_index,
        threshold=threshold,
        target_class=target_class,
    )


def _validate_counterfactual_result_with_app_prediction(
    *,
    session: SessionState,
    dataset: pd.DataFrame,
    result: dict[str, Any],
    row_index: int,
    threshold: float,
    target_class: int | None,
) -> dict[str, Any]:
    raw_feature_vector = extract_feature_vector_from_row(dataset, int(row_index), session.feature_metadata)
    prepared_feature_vector, prediction = predict_model(
        session.model,
        session.predictor,
        session.feature_metadata,
        raw_feature_vector,
    )
    current_probability = float(prediction.probability)
    current_prediction = int(current_probability >= float(threshold))
    target = int(1 - current_prediction if target_class is None else target_class)

    validated_counterfactuals: list[dict[str, Any]] = []
    for counterfactual in result.get("counterfactuals", []):
        replay_vector = dict(prepared_feature_vector)
        for change in counterfactual.get("changes", []):
            replay_vector[str(change["feature"])] = change["new_value"]

        try:
            _, replay_prediction = predict_model(
                session.model,
                session.predictor,
                session.feature_metadata,
                replay_vector,
            )
        except Exception:
            continue

        replay_probability = float(replay_prediction.probability)
        replay_label = int(replay_probability >= float(threshold))
        if replay_label != target:
            continue

        validated = dict(counterfactual)
        validated["new_probability"] = replay_probability
        validated["new_prediction"] = replay_label
        validated_counterfactuals.append(validated)

    validated_result = dict(result)
    validated_result["current_probability"] = current_probability
    validated_result["current_prediction"] = current_prediction
    validated_result["threshold"] = float(threshold)
    validated_result["counterfactuals"] = validated_counterfactuals
    return validated_result
