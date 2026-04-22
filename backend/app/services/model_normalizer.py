from __future__ import annotations

from typing import Any, Dict, Tuple
from uuid import uuid4

from app.domain.model_types import (
    NormalizedLeaf,
    NormalizedModel,
    NormalizedNode,
    NormalizedTree,
    RadialLayoutConfig,
    compute_model_radial_layout,
)
from app.services.feature_schema_service import build_feature_metadata


class LightGBMModelNormalizationError(ValueError):
    pass


def normalize_dumped_model(model_dump: Dict[str, Any]) -> NormalizedModel:
    objective = str(model_dump.get("objective", "")).lower()
    if "binary" not in objective:
        raise LightGBMModelNormalizationError("Only LightGBM binary classification models are supported.")
    if int(model_dump.get("num_class", 1)) != 1:
        raise LightGBMModelNormalizationError("Only binary classification models with one raw-score output are supported.")

    feature_names = [str(name) for name in model_dump.get("feature_names", [])]
    trees = [normalize_tree(tree_info, feature_names) for tree_info in model_dump.get("tree_info", [])]
    model = NormalizedModel(
        model_id=uuid4().hex,
        feature_names=feature_names,
        base_score=0.0,
        trees=trees,
    )
    compute_model_radial_layout(
        model,
        RadialLayoutConfig(inner_radius=0.08, outer_radius=1.0, depth_exponent=1.0),
    )
    return model


def normalize_tree(tree_info: Dict[str, Any], feature_names: list[str]) -> NormalizedTree:
    tree_index = int(tree_info["tree_index"])
    tree = NormalizedTree(tree_index=tree_index, root_id=-1)
    next_id = 0

    def _allocate_id() -> int:
        nonlocal next_id
        value = next_id
        next_id += 1
        return value

    def _walk(node_payload: Dict[str, Any], depth: int) -> int:
        object_id = _allocate_id()
        if "leaf_value" in node_payload:
            tree.leaves[object_id] = NormalizedLeaf(
                leaf_id=object_id,
                depth=depth,
                value=float(node_payload["leaf_value"]),
            )
            return object_id

        left_child_id = _walk(node_payload["left_child"], depth + 1)
        right_child_id = _walk(node_payload["right_child"], depth + 1)
        split_feature_index = int(node_payload["split_feature"])
        if split_feature_index < 0 or split_feature_index >= len(feature_names):
            raise LightGBMModelNormalizationError(
                f"Tree {tree_index} references split feature index {split_feature_index} outside feature_names."
            )
        tree.nodes[object_id] = NormalizedNode(
            node_id=object_id,
            depth=depth,
            split_feature=feature_names[split_feature_index],
            threshold=str(node_payload["threshold"]),
            decision_type=str(node_payload.get("decision_type", "<=")),
            default_left=bool(node_payload.get("default_left", True)),
            missing_type=str(node_payload.get("missing_type", "None")),
            left_child_id=left_child_id,
            right_child_id=right_child_id,
        )
        return object_id

    tree.root_id = _walk(tree_info["tree_structure"], 0)
    return tree
def summarize_layout(model: NormalizedModel) -> Tuple[int, int]:
    max_tree_depth = max((tree.max_depth for tree in model.trees), default=0)
    total_leaves = sum(tree.num_leaves for tree in model.trees)
    return max_tree_depth, total_leaves
