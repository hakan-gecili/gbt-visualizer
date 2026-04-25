from __future__ import annotations

from app.domain.model_types import EnsembleModel, FeatureImportanceEntry, PredictionResult


def serialize_feature_importance(entries: list[FeatureImportanceEntry]) -> list[dict]:
    return [
        {
            "feature_name": entry.feature_name,
            "value": entry.value,
        }
        for entry in entries
    ]


def serialize_model_summary(model: EnsembleModel) -> dict:
    return {
        "model_family": model.model_family,
        "model_type": model.model_type,
        "num_trees": model.num_trees,
        "num_features": len(model.feature_names),
        "feature_names": model.feature_names,
        "decision_threshold": model.decision_threshold,
        "global_feature_importance": serialize_feature_importance(model.metadata.global_feature_importance),
    }


def serialize_prediction(prediction_result: PredictionResult) -> dict:
    return {
        "margin": prediction_result.margin,
        "probability": prediction_result.probability,
        "predicted_label": prediction_result.predicted_label,
        "decision_threshold": prediction_result.decision_threshold,
        "local_feature_importance": serialize_feature_importance(prediction_result.local_feature_importance),
    }


def serialize_tree_predictions(prediction_result: PredictionResult) -> list[dict]:
    return [
        {
            "tree_index": item.tree_index,
            "selected_leaf_id": item.selected_leaf_id,
            "leaf_value": item.leaf_value,
            "contribution": item.leaf_value,
            "active_path_node_ids": item.path_node_ids,
        }
        for item in prediction_result.tree_results
    ]
