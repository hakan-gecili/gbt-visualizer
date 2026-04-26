from __future__ import annotations

from app.domain.model_types import EnsembleModel, extract_edges


def serialize_layout(model: EnsembleModel) -> dict:
    trees = []
    for tree in model.trees:
        trees.append(
            {
                "tree_index": tree.tree_index,
                "sector_start_angle": tree.sector_start_angle,
                "sector_end_angle": tree.sector_end_angle,
                "max_depth": tree.max_depth,
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "depth": node.depth,
                        "radius": node.radius,
                        "angle": node.angle,
                        "x": node.x,
                        "y": node.y,
                        "is_leaf": False,
                        "split_feature": node.split_feature,
                        "threshold": node.threshold,
                        "decision_type": node.decision_type,
                        "category_values": node.condition.category_values,
                        "left_child_id": node.left_child_id,
                        "right_child_id": node.right_child_id,
                        "subtree_leaf_count": node.subtree_leaf_count,
                    }
                    for node in sorted(tree.nodes.values(), key=lambda item: item.node_id)
                ],
                "edges": [
                    {"source_id": source_id, "target_id": target_id}
                    for source_id, target_id in extract_edges(tree)
                ],
                "leaves": [
                    {
                        "leaf_id": leaf.leaf_id,
                        "depth": leaf.depth,
                        "radius": leaf.radius,
                        "angle": leaf.angle,
                        "x": leaf.x,
                        "y": leaf.y,
                        "is_leaf": True,
                        "leaf_value": leaf.value,
                        "leaf_order": leaf.leaf_order,
                    }
                    for leaf in sorted(tree.leaves.values(), key=lambda item: item.leaf_id)
                ],
            }
        )
    return {"layout": {"trees": trees}}
