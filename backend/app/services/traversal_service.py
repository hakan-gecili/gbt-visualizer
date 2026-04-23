from __future__ import annotations

import math

from app.domain.model_types import SplitCondition, TraversalResult, Tree, get_object


def _is_missing_branch(feature_value: float, missing_value_strategy: str) -> bool:
    if math.isnan(feature_value):
        return True
    return missing_value_strategy.lower() == "zero" and feature_value == 0.0


def _take_true_branch(feature_value: float, condition: SplitCondition) -> bool:
    if condition.operator == "<=":
        if condition.threshold is None:
            raise ValueError(f"Node for feature '{condition.feature_name}' is missing a numeric threshold.")
        return feature_value <= float(condition.threshold)
    if condition.operator == "==":
        accepted_values = (
            set(condition.category_values)
            if condition.category_values
            else {float(item) for item in str(condition.threshold).split("||") if item != ""}
        )
        return feature_value in accepted_values
    raise ValueError(f"Unsupported split operator '{condition.operator}'.")


def traverse_tree(tree: Tree, feature_vector: dict[str, float]) -> TraversalResult:
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
        feature_value = feature_vector.get(obj.condition.feature_name, math.nan)
        if _is_missing_branch(feature_value, obj.condition.missing_value_strategy):
            current_id = obj.left_child_id if obj.default_left else obj.right_child_id
            continue

        if _take_true_branch(feature_value, obj.condition):
            current_id = obj.left_child_id
            continue

        current_id = obj.right_child_id
