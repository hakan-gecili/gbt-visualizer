from __future__ import annotations

from app.domain.model_types import PredictionResult, NormalizedModel, sigmoid
from app.services.feature_schema_service import FeatureValue, encode_feature_vector, prepare_feature_vector
from app.services.traversal_service import traverse_tree


def predict_model(
    model: NormalizedModel,
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
    return prepared_feature_vector, PredictionResult(
        margin=margin,
        probability=probability,
        tree_results=tree_results,
    )
