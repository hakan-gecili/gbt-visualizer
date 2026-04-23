from __future__ import annotations

from app.adapters.lightgbm_adapter import LightGBMModelAdapter
from app.domain.model_types import EnsembleModel, PredictionResult, sigmoid
from app.services.feature_schema_service import FeatureValue, encode_feature_vector, prepare_feature_vector
from app.services.traversal_service import traverse_tree


def predict_model(
    model: EnsembleModel,
    predictor: object | None,
    feature_metadata: list[dict[str, object]],
    raw_feature_vector: dict[str, FeatureValue],
) -> tuple[dict[str, FeatureValue], PredictionResult]:
    prepared_feature_vector = prepare_feature_vector(feature_metadata, raw_feature_vector)
    feature_vector = encode_feature_vector(feature_metadata, prepared_feature_vector)
    margin = model.base_score
    tree_results = []

    for tree in model.trees:
        traversal = traverse_tree(tree, feature_vector)
        tree_results.append(traversal)
        margin += traversal.leaf_value

    probability = sigmoid(margin)
    predicted_label = int(probability >= model.decision_threshold)
    local_feature_importance = (
        LightGBMModelAdapter.compute_local_feature_importance(predictor, model.feature_names, feature_vector)
        if predictor is not None
        else []
    )
    return prepared_feature_vector, PredictionResult(
        margin=margin,
        probability=probability,
        predicted_label=predicted_label,
        decision_threshold=model.decision_threshold,
        local_feature_importance=local_feature_importance,
        tree_results=tree_results,
    )
