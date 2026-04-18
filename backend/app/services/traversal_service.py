from __future__ import annotations

import math
from typing import Dict

from app.domain.model_types import NormalizedModel, NormalizedTree, TraversalResult, get_object


def prepare_feature_vector(model: NormalizedModel, feature_vector: Dict[str, float]) -> Dict[str, float]:
    prepared = {feature_name: 0.0 for feature_name in model.feature_names}
    for key, value in feature_vector.items():
        if key in prepared:
            prepared[key] = float(value)
    return prepared


def traverse_tree(tree: NormalizedTree, feature_vector: Dict[str, float]) -> TraversalResult:
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
        feature_value = feature_vector.get(obj.split_feature, 0.0)
        if math.isnan(feature_value) or feature_value <= obj.threshold:
            current_id = obj.left_child_id
        else:
            current_id = obj.right_child_id

