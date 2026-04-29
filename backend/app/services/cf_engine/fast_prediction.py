from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from app.domain.model_types import EnsembleModel, sigmoid
from app.services.feature_schema_service import FeatureValue, encode_feature_vector
from app.services.prediction_service import predict_model


RELIABILITY_EXACT_APP_SEMANTICS = "exact_app_semantics"
RELIABILITY_MODEL_NATIVE_RANKING_ONLY = "model_native_ranking_only"


@dataclass(frozen=True)
class BatchPrediction:
    margins: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    reliability: str


class FastPredictionEvaluator(Protocol):
    model_family: str
    supports_batch_prediction: bool
    reliability_notes: str
    final_validation_allowed: bool

    def predict_batch(self, rows: list[dict[str, FeatureValue]], threshold: float) -> BatchPrediction:
        ...

    def predict_margin_batch(self, rows: list[dict[str, FeatureValue]]) -> np.ndarray:
        ...

    def predict_probability_batch(self, rows: list[dict[str, FeatureValue]]) -> np.ndarray:
        ...

    def predict_label_batch(self, rows: list[dict[str, FeatureValue]], threshold: float) -> np.ndarray:
        ...


class AppPredictModelEvaluator:
    model_family = "app"
    supports_batch_prediction = False
    reliability_notes = RELIABILITY_EXACT_APP_SEMANTICS
    final_validation_allowed = True

    def __init__(
        self,
        *,
        model: EnsembleModel,
        predictor: object | None,
        feature_metadata: list[dict[str, Any]],
    ) -> None:
        self.model = model
        self.predictor = predictor
        self.feature_metadata = feature_metadata
        self.model_family = str(model.model_family)

    def predict_batch(self, rows: list[dict[str, FeatureValue]], threshold: float) -> BatchPrediction:
        margins: list[float] = []
        probabilities: list[float] = []
        labels: list[int] = []
        for row in rows:
            _, prediction = predict_model(self.model, self.predictor, self.feature_metadata, row)
            probability = float(prediction.probability)
            margins.append(float(prediction.margin))
            probabilities.append(probability)
            labels.append(int(probability >= float(threshold)))
        return BatchPrediction(
            margins=np.asarray(margins, dtype=float),
            probabilities=np.asarray(probabilities, dtype=float),
            labels=np.asarray(labels, dtype=int),
            reliability=self.reliability_notes,
        )

    def predict_margin_batch(self, rows: list[dict[str, FeatureValue]]) -> np.ndarray:
        return self.predict_batch(rows, threshold=float(self.model.decision_threshold)).margins

    def predict_probability_batch(self, rows: list[dict[str, FeatureValue]]) -> np.ndarray:
        return self.predict_batch(rows, threshold=float(self.model.decision_threshold)).probabilities

    def predict_label_batch(self, rows: list[dict[str, FeatureValue]], threshold: float) -> np.ndarray:
        return self.predict_batch(rows, threshold=threshold).labels


class XGBoostBoosterFastEvaluator:
    model_family = "xgboost"
    supports_batch_prediction = True
    reliability_notes = RELIABILITY_MODEL_NATIVE_RANKING_ONLY
    # Native XGBoost can differ from app traversal at split boundaries; use it
    # only to order candidates, never to approve a returned counterfactual.
    final_validation_allowed = False

    def __init__(
        self,
        *,
        model: EnsembleModel,
        booster: object,
        feature_metadata: list[dict[str, Any]],
    ) -> None:
        import xgboost as xgb

        if not isinstance(booster, xgb.Booster):
            if hasattr(booster, "get_booster"):
                booster = booster.get_booster()
            if not isinstance(booster, xgb.Booster):
                raise TypeError("XGBoostBoosterFastEvaluator requires an xgboost.Booster-compatible predictor.")

        self.model = model
        self.booster = booster
        self.feature_metadata = feature_metadata

    def predict_batch(self, rows: list[dict[str, FeatureValue]], threshold: float) -> BatchPrediction:
        import xgboost as xgb

        matrix = xgb.DMatrix(
            _encode_rows(self.feature_metadata, self.model.feature_names, rows),
            feature_names=self.model.feature_names,
        )
        margins = np.asarray(self.booster.predict(matrix, output_margin=True), dtype=float)
        probabilities = _sigmoid_array(margins)
        return BatchPrediction(
            margins=margins,
            probabilities=probabilities,
            labels=(probabilities >= float(threshold)).astype(int),
            reliability=self.reliability_notes,
        )

    def predict_margin_batch(self, rows: list[dict[str, FeatureValue]]) -> np.ndarray:
        return self.predict_batch(rows, threshold=float(self.model.decision_threshold)).margins

    def predict_probability_batch(self, rows: list[dict[str, FeatureValue]]) -> np.ndarray:
        return self.predict_batch(rows, threshold=float(self.model.decision_threshold)).probabilities

    def predict_label_batch(self, rows: list[dict[str, FeatureValue]], threshold: float) -> np.ndarray:
        return self.predict_batch(rows, threshold=threshold).labels


class LightGBMBoosterFastEvaluator:
    model_family = "lightgbm"
    supports_batch_prediction = True
    reliability_notes = RELIABILITY_MODEL_NATIVE_RANKING_ONLY
    # Keep native LightGBM behind the same ranking-only boundary until the
    # unified engine is promoted and validated more broadly.
    final_validation_allowed = False

    def __init__(
        self,
        *,
        model: EnsembleModel,
        booster: object,
        feature_metadata: list[dict[str, Any]],
    ) -> None:
        self.model = model
        self.booster = booster
        self.feature_metadata = feature_metadata

    def predict_batch(self, rows: list[dict[str, FeatureValue]], threshold: float) -> BatchPrediction:
        margins = np.asarray(
            self.booster.predict(
                _encode_rows(self.feature_metadata, self.model.feature_names, rows),
                raw_score=True,
            ),
            dtype=float,
        )
        probabilities = _sigmoid_array(margins)
        return BatchPrediction(
            margins=margins,
            probabilities=probabilities,
            labels=(probabilities >= float(threshold)).astype(int),
            reliability=self.reliability_notes,
        )

    def predict_margin_batch(self, rows: list[dict[str, FeatureValue]]) -> np.ndarray:
        return self.predict_batch(rows, threshold=float(self.model.decision_threshold)).margins

    def predict_probability_batch(self, rows: list[dict[str, FeatureValue]]) -> np.ndarray:
        return self.predict_batch(rows, threshold=float(self.model.decision_threshold)).probabilities

    def predict_label_batch(self, rows: list[dict[str, FeatureValue]], threshold: float) -> np.ndarray:
        return self.predict_batch(rows, threshold=threshold).labels


def _encode_rows(
    feature_metadata: list[dict[str, Any]],
    feature_names: list[str],
    rows: list[dict[str, FeatureValue]],
) -> np.ndarray:
    encoded_rows: list[list[float]] = []
    for row in rows:
        encoded = encode_feature_vector(feature_metadata, row)
        encoded_rows.append([encoded.get(feature_name, np.nan) for feature_name in feature_names])
    return np.asarray(encoded_rows, dtype=float)


def _sigmoid_array(margins: np.ndarray) -> np.ndarray:
    return np.asarray([sigmoid(float(margin)) for margin in margins], dtype=float)


def select_fast_prediction_evaluator(
    *,
    model: EnsembleModel,
    predictor: object | None,
    feature_metadata: list[dict[str, Any]],
    use_fast_evaluator: bool,
) -> FastPredictionEvaluator | None:
    # USE_FAST_CF_EVALUATOR gates native predictors globally; when disabled,
    # unified counterfactual search keeps its original exact replay ordering.
    if not use_fast_evaluator:
        return None

    model_family = str(model.model_family).lower()
    if predictor is not None and model_family == "xgboost":
        return XGBoostBoosterFastEvaluator(
            model=model,
            booster=predictor,
            feature_metadata=feature_metadata,
        )
    if predictor is not None and model_family == "lightgbm":
        return LightGBMBoosterFastEvaluator(
            model=model,
            booster=predictor,
            feature_metadata=feature_metadata,
        )
    return AppPredictModelEvaluator(
        model=model,
        predictor=predictor,
        feature_metadata=feature_metadata,
    )
