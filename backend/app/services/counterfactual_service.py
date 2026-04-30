from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pandas as pd

from app.core.session_store import session_store
from app.domain.session_types import SessionState
from app.services.cf_engine import build_counterfactual_engine
from app.services.cf_engine.fast_prediction import select_fast_prediction_evaluator
from app.services.cf_engine.unified_counterfactual_engine import UnifiedPrediction, generate_unified_counterfactual
from app.services.dataset_service import extract_feature_vector_from_row
from app.services.feature_schema_service import FeatureValue, prepare_feature_vector
from app.services.prediction_service import predict_model
from app.services.xgboost_counterfactual_service import generate_xgboost_counterfactual_for_session as generate_xgboost_counterfactual

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


def build_lightgbm_counterfactual_metadata(model: Any) -> dict[str, float]:
    trees = list(getattr(model, "trees", []) or [])
    if not trees:
        return {"avg_leaves": 0.0, "avg_depth": 0.0}

    leaf_counts = [float(getattr(tree, "num_leaves", 0) or len(getattr(tree, "leaves", {}) or {})) for tree in trees]
    depths = [
        float(
            getattr(tree, "max_depth", 0)
            or max((getattr(leaf, "depth", 0) for leaf in getattr(tree, "leaves", {}).values()), default=0)
        )
        for tree in trees
    ]
    return {
        "avg_leaves": sum(leaf_counts) / len(leaf_counts),
        "avg_depth": sum(depths) / len(depths),
    }


def _get_lightgbm_counterfactual_metadata(session: SessionState) -> dict[str, float]:
    metadata = session.counterfactual_metadata
    if "avg_leaves" not in metadata:
        metadata.update(build_lightgbm_counterfactual_metadata(session.model))
    return {
        "avg_leaves": float(metadata.get("avg_leaves", 0.0)),
        "avg_depth": float(metadata.get("avg_depth", 0.0)),
    }


def _legacy_lightgbm_adaptive_max_steps(avg_leaves: float, margin: float) -> int:
    if avg_leaves <= 12.0:
        max_steps = 3
    elif avg_leaves <= 25.0:
        max_steps = 5
    else:
        max_steps = 10

    if margin < -2.0:
        max_steps += 2

    return min(max_steps, 12)


def _counterfactual_debug_enabled() -> bool:
    return str(os.getenv("COUNTERFACTUAL_DEBUG", "")).lower() in {"1", "true", "yes", "on"}


def _debug_legacy_lightgbm_max_steps(
    *,
    avg_leaves: float,
    margin: float | None,
    max_steps: int,
    manual_override: bool,
) -> None:
    if not _counterfactual_debug_enabled():
        return
    margin_text = "unknown" if margin is None else f"{margin:.6f}"
    print(
        "[counterfactual:legacy-lightgbm] "
        f"avg_leaves={avg_leaves:.3f} "
        f"original_margin={margin_text} "
        f"max_steps={int(max_steps)} "
        f"manual_override={manual_override}"
    )


def _resolve_legacy_lightgbm_max_steps(
    *,
    session: SessionState,
    dataset: pd.DataFrame,
    row_index: int,
    max_steps: int | None,
    feature_vector: dict[str, FeatureValue] | None = None,
) -> int:
    manual_override = max_steps is not None
    metadata = _get_lightgbm_counterfactual_metadata(session)
    avg_leaves = metadata["avg_leaves"]

    if manual_override:
        resolved_max_steps = int(max_steps)
        margin: float | None = None
        if _counterfactual_debug_enabled():
            raw_vector = feature_vector or extract_feature_vector_from_row(dataset, row_index, session.feature_metadata)
            _, prediction = predict_model(session.model, session.predictor, session.feature_metadata, raw_vector)
            margin = float(prediction.margin)
        _debug_legacy_lightgbm_max_steps(
            avg_leaves=avg_leaves,
            margin=margin,
            max_steps=resolved_max_steps,
            manual_override=True,
        )
        return resolved_max_steps

    raw_vector = feature_vector or extract_feature_vector_from_row(dataset, row_index, session.feature_metadata)
    _, prediction = predict_model(session.model, session.predictor, session.feature_metadata, raw_vector)
    margin = float(prediction.margin)
    resolved_max_steps = _legacy_lightgbm_adaptive_max_steps(avg_leaves, margin)
    _debug_legacy_lightgbm_max_steps(
        avg_leaves=avg_leaves,
        margin=margin,
        max_steps=resolved_max_steps,
        manual_override=False,
    )
    return resolved_max_steps


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
        if not _is_supported_lightgbm_counterfactual_model(model, engine_model):
            raise TypeError("Counterfactual generation is currently supported only for LightGBM models.")
        _engine_cache[cache_key] = build_counterfactual_engine(
            model=engine_model,
            dataset=dataset,
            schema=schema,
        )
    return _engine_cache[cache_key]


def _is_supported_lightgbm_counterfactual_model(model: Any, engine_model: Any) -> bool:
    model_type = str(getattr(model, "model_type", "")).lower()
    if model_type and "lightgbm" not in model_type:
        return False

    booster = engine_model.booster_ if hasattr(engine_model, "booster_") else engine_model
    return callable(getattr(booster, "feature_name", None)) and callable(getattr(booster, "dump_model", None))


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
    feature_vector: dict[str, FeatureValue] | None = None,
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
        feature_vector=feature_vector,
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
    feature_vector: dict[str, FeatureValue] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    session = session_store.get(session_id)
    dataset = _ensure_session_dataset(session)
    schema = build_counterfactual_schema(session.feature_metadata)
    prepared_feature_vector = (
        prepare_feature_vector(session.feature_metadata, feature_vector)
        if feature_vector is not None
        else None
    )
    if str(session.model.model_family).lower() == "xgboost":
        return generate_xgboost_counterfactual(
            model=session.model,
            predictor=session.predictor,
            dataset=dataset,
            feature_metadata=session.feature_metadata,
            row_index=row_index,
            threshold=threshold,
            target_class=target_class,
            max_steps=3 if max_steps is None else int(max_steps),
            feature_vector=prepared_feature_vector,
        )
    if str(session.model.model_family).lower() != "lightgbm":
        raise TypeError("Counterfactual generation is currently supported only for LightGBM and XGBoost models.")
    if _use_unified_counterfactual_for_lightgbm():
        return _generate_unified_lightgbm_counterfactual_for_session(
            session=session,
            dataset=dataset,
            row_index=row_index,
            threshold=threshold,
            target_class=target_class,
            max_steps=3 if max_steps is None else int(max_steps),
            feature_vector=prepared_feature_vector,
        )
    resolved_max_steps = _resolve_legacy_lightgbm_max_steps(
        session=session,
        dataset=dataset,
        row_index=row_index,
        max_steps=max_steps,
        feature_vector=prepared_feature_vector,
    )
    result = generate_counterfactual(
        model=session.model,
        predictor=session.predictor,
        dataset=dataset,
        schema=schema,
        row_index=row_index,
        threshold=threshold,
        target_class=target_class,
        max_steps=resolved_max_steps,
        key=_build_session_engine_key(session_id, session.model, session.predictor, dataset, schema),
        feature_vector=prepared_feature_vector,
    )
    return _validate_counterfactual_result_with_app_prediction(
        session=session,
        dataset=dataset,
        result=result,
        row_index=row_index,
        threshold=threshold,
        target_class=target_class,
        feature_vector=prepared_feature_vector,
    )


def _use_unified_counterfactual_for_lightgbm() -> bool:
    # Legacy LightGBM remains the default until unified parity gaps are reviewed.
    return str(os.getenv("USE_UNIFIED_CF_FOR_LIGHTGBM", "")).lower() in {"1", "true", "yes", "on"}


def _use_fast_counterfactual_evaluator() -> bool:
    # Enables ranking-only native batch prediction; predict_model remains the
    # final validator in the unified engine.
    return str(os.getenv("USE_FAST_CF_EVALUATOR", "")).lower() in {"1", "true", "yes", "on"}


def _generate_unified_lightgbm_counterfactual_for_session(
    *,
    session: SessionState,
    dataset: pd.DataFrame,
    row_index: int,
    threshold: float,
    target_class: int | None,
    max_steps: int,
    feature_vector: dict[str, FeatureValue] | None = None,
) -> dict[str, Any]:
    raw_feature_vector = feature_vector or extract_feature_vector_from_row(dataset, int(row_index), session.feature_metadata)
    original_vector, _ = predict_model(
        session.model,
        session.predictor,
        session.feature_metadata,
        raw_feature_vector,
    )

    def _predict(vector: dict[str, Any]) -> UnifiedPrediction:
        _, prediction = predict_model(
            session.model,
            session.predictor,
            session.feature_metadata,
            vector,
        )
        probability = float(prediction.probability)
        return UnifiedPrediction(
            probability=probability,
            margin=float(prediction.margin),
            label=int(probability >= float(threshold)),
        )

    use_fast_evaluator = _use_fast_counterfactual_evaluator()
    return generate_unified_counterfactual(
        model=session.model,
        feature_metadata=session.feature_metadata,
        original_vector=original_vector,
        prediction_evaluator=_predict,
        threshold=threshold,
        target_class=target_class,
        max_steps=max_steps,
        fast_evaluator=select_fast_prediction_evaluator(
            model=session.model,
            predictor=session.predictor,
            feature_metadata=session.feature_metadata,
            use_fast_evaluator=use_fast_evaluator,
        ),
        use_fast_evaluator=use_fast_evaluator,
        debug_label="lightgbm-unified",
    )


def _validate_counterfactual_result_with_app_prediction(
    *,
    session: SessionState,
    dataset: pd.DataFrame,
    result: dict[str, Any],
    row_index: int,
    threshold: float,
    target_class: int | None,
    feature_vector: dict[str, FeatureValue] | None = None,
) -> dict[str, Any]:
    raw_feature_vector = feature_vector or extract_feature_vector_from_row(dataset, int(row_index), session.feature_metadata)
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
        validated["new_margin"] = float(replay_prediction.margin)
        validated["new_prediction"] = replay_label
        validated_counterfactuals.append(validated)

    validated_result = dict(result)
    validated_result["current_probability"] = current_probability
    validated_result["current_prediction"] = current_prediction
    validated_result["threshold"] = float(threshold)
    validated_result["counterfactuals"] = validated_counterfactuals
    return validated_result
