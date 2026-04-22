from __future__ import annotations

import math

from app.domain.model_types import NormalizedTree, TraversalResult, get_object


def _is_missing_branch(feature_value: float, missing_type: str) -> bool:
    if math.isnan(feature_value):
        return True
    return missing_type.lower() == "zero" and feature_value == 0.0


def _take_left_branch(feature_value: float, decision_type: str, threshold: str) -> bool:
    if decision_type == "<=":
        return feature_value <= float(threshold)
    if decision_type == "==":
        accepted_values = {float(item) for item in threshold.split("||") if item != ""}
        return feature_value in accepted_values
    raise ValueError(f"Unsupported LightGBM decision_type '{decision_type}'.")


def traverse_tree(tree: NormalizedTree, feature_vector: dict[str, float]) -> TraversalResult:
    current_id = tree.root_id
    path_node_ids: list[int] = []

    while True:
        obj = get_object(tree, current_id)
        if obj.is_leaf:
            return TraversalResult(
                tree_index=tree.tree_index,
                path_node_ids=path_node_ids,
                selected_leaf_id=obj.leaf_id,
                leaf_value=obj.value,
            )

        path_node_ids.append(obj.node_id)
        feature_value = feature_vector.get(obj.split_feature, math.nan)
        if _is_missing_branch(feature_value, obj.missing_type):
            current_id = obj.left_child_id if obj.default_left else obj.right_child_id
            continue

        if _take_left_branch(feature_value, obj.decision_type, obj.threshold):
            current_id = obj.left_child_id
            continue

        current_id = obj.right_child_id
