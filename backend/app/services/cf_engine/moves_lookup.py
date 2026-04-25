# counterfactuals/moves_lookup.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib

def is_leaf(node: Dict[str, Any]) -> bool:
    return "leaf_value" in node


def get_children(node: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return node["left_child"], node["right_child"]


def node_feature_idx(node: Dict[str, Any]) -> int:
    return int(node["split_feature"])


def node_threshold(node: Dict[str, Any]) -> Any:
    return node["threshold"]


def node_decision_type(node: Dict[str, Any]) -> str:
    return str(node.get("decision_type", "<="))


NodeKey = Tuple[str, int]


@dataclass(frozen=True)
class Move:
    tree_id: int
    from_leaf_id: int
    split_feature: int
    feature_name: str
    threshold: Any
    decision_type: str
    direction_required: str
    want_right: bool
    sibling_node_id: NodeKey
    subtree_max_leaf_value: float
    current_leaf_value: float
    delta_leaf_upper_bound: float
    split_index: Optional[int] = None


def _node_key(node: Dict[str, Any]) -> NodeKey:
    if is_leaf(node):
        return ("leaf", int(node["leaf_index"]))
    return ("split", int(node.get("split_index", -1)))


def _compute_subtree_max(node: Dict[str, Any], stats: Dict[NodeKey, float]) -> float:
    if is_leaf(node):
        val = float(node["leaf_value"])
        stats[_node_key(node)] = val
        return val

    left, right = get_children(node)
    left_max = _compute_subtree_max(left, stats)
    right_max = _compute_subtree_max(right, stats)
    val = float(max(left_max, right_max))
    stats[_node_key(node)] = val
    return val


def _collect_leaf_paths(
    node: Dict[str, Any],
    path: List[Tuple[Dict[str, Any], bool]],
    leaf_paths: Dict[int, List[Tuple[Dict[str, Any], bool]]],
    leaf_values: Dict[int, float],
) -> None:
    if is_leaf(node):
        leaf_id = int(node["leaf_index"])
        leaf_paths[leaf_id] = list(path)
        leaf_values[leaf_id] = float(node["leaf_value"])
        return

    left, right = get_children(node)
    _collect_leaf_paths(left, path + [(node, True)], leaf_paths, leaf_values)
    _collect_leaf_paths(right, path + [(node, False)], leaf_paths, leaf_values)


def build_moves_lookup(booster, top_m: int = 8) -> Dict[Tuple[int, int], List[Move]]:
    dump = booster.dump_model()
    feature_names = list(booster.feature_name())
    trees = [t["tree_structure"] for t in dump["tree_info"]]

    moves_lookup: Dict[Tuple[int, int], List[Move]] = {}

    for tree_id, tree_root in enumerate(trees):
        subtree_stats: Dict[NodeKey, float] = {}
        _compute_subtree_max(tree_root, subtree_stats)

        leaf_paths: Dict[int, List[Tuple[Dict[str, Any], bool]]] = {}
        leaf_values: Dict[int, float] = {}
        _collect_leaf_paths(tree_root, [], leaf_paths, leaf_values)

        for leaf_id, path in leaf_paths.items():
            leaf_value_cur = float(leaf_values[leaf_id])
            moves: List[Move] = []
            for node, went_left in path:
                if node_decision_type(node) not in {"<=", "=="}:
                    continue

                left_child, right_child = get_children(node)
                sibling = right_child if went_left else left_child

                subtree_max = float(subtree_stats[_node_key(sibling)])
                delta_ub = subtree_max - leaf_value_cur

                feature_idx = node_feature_idx(node)
                moves.append(
                    Move(
                        tree_id=int(tree_id),
                        from_leaf_id=int(leaf_id),
                        split_feature=int(feature_idx),
                        feature_name=str(feature_names[int(feature_idx)]),
                        threshold=node_threshold(node),
                        decision_type=node_decision_type(node),
                        direction_required="INCREASE" if went_left else "DECREASE",
                        want_right=bool(went_left),
                        sibling_node_id=_node_key(sibling),
                        subtree_max_leaf_value=subtree_max,
                        current_leaf_value=leaf_value_cur,
                        delta_leaf_upper_bound=float(delta_ub),
                        split_index=int(node.get("split_index", -1)),
                    )
                )

            if moves:
                moves.sort(key=lambda m: m.delta_leaf_upper_bound, reverse=True)
                moves_lookup[(int(tree_id), int(leaf_id))] = moves[: int(top_m)]

    return moves_lookup


def save_moves_lookup(moves_lookup: Dict[Tuple[int, int], List[Move]], out_path: str, meta: Optional[Dict[str, Any]] = None) -> None:
    payload = {"moves_lookup": moves_lookup, "meta": dict(meta or {})}
    joblib.dump(payload, out_path)


def load_moves_lookup(path: str) -> Dict[Tuple[int, int], List[Move]]:
    payload = joblib.load(path)
    return payload["moves_lookup"]
