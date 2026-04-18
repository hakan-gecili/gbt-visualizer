from __future__ import annotations

from app.domain.model_types import PredictionResult, NormalizedModel, sigmoid
from app.services.traversal_service import prepare_feature_vector, traverse_tree


def predict_model(model: NormalizedModel, raw_feature_vector: dict[str, float]) -> tuple[dict[str, float], PredictionResult]:
    feature_vector = prepare_feature_vector(model, raw_feature_vector)
    margin = model.base_score
    tree_results = []

    for tree in model.trees:
        traversal = traverse_tree(tree, feature_vector)
        tree_results.append(traversal)
        margin += traversal.leaf_value

    probability = sigmoid(margin)
    return feature_vector, PredictionResult(
        margin=margin,
        probability=probability,
        tree_results=tree_results,
    )

