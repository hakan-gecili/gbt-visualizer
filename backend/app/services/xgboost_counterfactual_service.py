from __future__ import annotations

from typing import Any

import pandas as pd

from app.domain.model_types import EnsembleModel
from app.services.cf_engine.fast_prediction import select_fast_prediction_evaluator
from app.services.cf_engine.unified_counterfactual_engine import UnifiedPrediction, generate_unified_counterfactual
from app.services.dataset_service import extract_feature_vector_from_row
from app.services.feature_schema_service import FeatureValue
from app.services.prediction_service import predict_model


def generate_xgboost_counterfactual_for_session(
    *,
    model: EnsembleModel,
    predictor: object | None,
    dataset: pd.DataFrame,
    feature_metadata: list[dict[str, Any]],
    row_index: int,
    threshold: float,
    target_class: int | None,
    max_steps: int = 3,
) -> dict[str, Any]:
    if not 0 <= int(row_index) < len(dataset):
        raise IndexError(f"row_index out of range; dataset has {len(dataset)} rows")

    raw_feature_vector = extract_feature_vector_from_row(dataset, int(row_index), feature_metadata)
    original_vector, _ = predict_model(model, predictor, feature_metadata, raw_feature_vector)

    def _predict(vector: dict[str, FeatureValue]) -> UnifiedPrediction:
        _, prediction = predict_model(model, predictor, feature_metadata, vector)
        probability = float(prediction.probability)
        return UnifiedPrediction(
            probability=probability,
            margin=float(prediction.margin),
            label=int(probability >= float(threshold)),
        )

    use_fast_evaluator = _use_fast_counterfactual_evaluator()
    return generate_unified_counterfactual(
        model=model,
        feature_metadata=feature_metadata,
        original_vector=original_vector,
        prediction_evaluator=_predict,
        threshold=threshold,
        target_class=target_class,
        max_steps=max_steps,
        fast_evaluator=select_fast_prediction_evaluator(
            model=model,
            predictor=predictor,
            feature_metadata=feature_metadata,
            use_fast_evaluator=use_fast_evaluator,
        ),
        use_fast_evaluator=use_fast_evaluator,
        debug_label="xgboost",
    )


def _use_fast_counterfactual_evaluator() -> bool:
    import os

    # Native XGBoost prediction is faster but ranking-only; final replay stays
    # on predict_model through generate_unified_counterfactual().
    return str(os.getenv("USE_FAST_CF_EVALUATOR", "")).lower() in {"1", "true", "yes", "on"}
