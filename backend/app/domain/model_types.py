from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, exp, pi, sin
from typing import Dict, List, Optional, Tuple


@dataclass
class NormalizedNode:
    node_id: int
    depth: int
    split_feature: str
    threshold: str
    decision_type: str
    default_left: bool
    missing_type: str
    left_child_id: int
    right_child_id: int
    subtree_leaf_count: int = 0
    angle: float = 0.0
    radius: float = 0.0
    x: float = 0.0
    y: float = 0.0
    is_leaf: bool = False


@dataclass
class NormalizedLeaf:
    leaf_id: int
    depth: int
    value: float
    subtree_leaf_count: int = 1
    angle: float = 0.0
    radius: float = 0.0
    x: float = 0.0
    y: float = 0.0
    leaf_order: int = 0
    is_leaf: bool = True


@dataclass
class NormalizedTree:
    tree_index: int
    root_id: int
    nodes: Dict[int, NormalizedNode] = field(default_factory=dict)
    leaves: Dict[int, NormalizedLeaf] = field(default_factory=dict)
    max_depth: int = 0
    num_leaves: int = 0
    sector_start_angle: float = 0.0
    sector_end_angle: float = 0.0


@dataclass
class NormalizedModel:
    model_id: str
    feature_names: List[str]
    base_score: float
    trees: List[NormalizedTree]

    @property
    def num_trees(self) -> int:
        return len(self.trees)


@dataclass
class TraversalResult:
    tree_index: int
    path_node_ids: List[int]
    selected_leaf_id: int
    leaf_value: float


@dataclass
class PredictionResult:
    margin: float
    probability: float
    tree_results: List[TraversalResult]


@dataclass
class RadialLayoutConfig:
    inner_radius: float = 0.08
    outer_radius: float = 1.0
    depth_exponent: float = 1.0


def is_leaf_id(tree: NormalizedTree, object_id: int) -> bool:
    return object_id in tree.leaves


def get_object(tree: NormalizedTree, object_id: int) -> NormalizedNode | NormalizedLeaf:
    if object_id in tree.nodes:
        return tree.nodes[object_id]
    if object_id in tree.leaves:
        return tree.leaves[object_id]
    raise KeyError(f"Object ID {object_id} not found in tree {tree.tree_index}.")


def compute_subtree_leaf_counts(tree: NormalizedTree) -> int:
    def _count(object_id: int) -> int:
        obj = get_object(tree, object_id)
        if obj.is_leaf:
            return 1
        left_count = _count(obj.left_child_id)
        right_count = _count(obj.right_child_id)
        obj.subtree_leaf_count = left_count + right_count
        return obj.subtree_leaf_count

    total = _count(tree.root_id)
    tree.num_leaves = total
    return total


def compute_max_depth(tree: NormalizedTree) -> int:
    def _depth(object_id: int) -> int:
        obj = get_object(tree, object_id)
        if obj.is_leaf:
            return obj.depth
        return max(_depth(obj.left_child_id), _depth(obj.right_child_id))

    tree.max_depth = _depth(tree.root_id)
    return tree.max_depth


def assign_leaf_order(tree: NormalizedTree) -> None:
    current_index = 0

    def _dfs(object_id: int) -> None:
        nonlocal current_index
        obj = get_object(tree, object_id)
        if obj.is_leaf:
            obj.leaf_order = current_index
            current_index += 1
            return
        _dfs(obj.left_child_id)
        _dfs(obj.right_child_id)

    _dfs(tree.root_id)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def depth_to_radius(depth: int, max_depth: int, config: RadialLayoutConfig) -> float:
    if max_depth <= 0:
        return config.outer_radius
    frac = (depth / max_depth) ** config.depth_exponent
    return config.inner_radius + frac * (config.outer_radius - config.inner_radius)


def assign_tree_sectors(model: NormalizedModel) -> None:
    if model.num_trees == 0:
        return
    # SVG screen coordinates invert the Y axis, so visually rotating the ring
    # counterclockwise to the top requires a -pi/2 angular offset.
    angle_offset = -pi / 2
    delta = 2 * pi / model.num_trees
    for index, tree in enumerate(model.trees):
        tree.sector_start_angle = index * delta + angle_offset
        tree.sector_end_angle = (index + 1) * delta + angle_offset


def compute_tree_radial_layout(tree: NormalizedTree, config: RadialLayoutConfig) -> None:
    if tree.num_leaves == 0:
        compute_subtree_leaf_counts(tree)
    if tree.max_depth == 0:
        compute_max_depth(tree)

    def _subtree_count(object_id: int) -> int:
        obj = get_object(tree, object_id)
        return obj.subtree_leaf_count

    def _place(object_id: int, theta_left: float, theta_right: float) -> None:
        obj = get_object(tree, object_id)
        theta_mid = 0.5 * (theta_left + theta_right)
        radius = depth_to_radius(obj.depth, tree.max_depth, config)

        obj.angle = theta_mid
        obj.radius = radius
        obj.x = radius * cos(theta_mid)
        obj.y = radius * sin(theta_mid)

        if obj.is_leaf:
            return

        left_count = _subtree_count(obj.left_child_id)
        right_count = _subtree_count(obj.right_child_id)
        total = left_count + right_count
        if total <= 0:
            raise ValueError(f"Invalid subtree counts in tree {tree.tree_index}.")

        theta_split = theta_left + (left_count / total) * (theta_right - theta_left)
        _place(obj.left_child_id, theta_left, theta_split)
        _place(obj.right_child_id, theta_split, theta_right)

    _place(tree.root_id, tree.sector_start_angle, tree.sector_end_angle)


def compute_model_radial_layout(
    model: NormalizedModel,
    config: Optional[RadialLayoutConfig] = None,
) -> None:
    layout_config = config or RadialLayoutConfig()
    assign_tree_sectors(model)
    for tree in model.trees:
        compute_subtree_leaf_counts(tree)
        compute_max_depth(tree)
        assign_leaf_order(tree)
        compute_tree_radial_layout(tree, layout_config)


def extract_edges(tree: NormalizedTree) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for node in tree.nodes.values():
        edges.append((node.node_id, node.left_child_id))
        edges.append((node.node_id, node.right_child_id))
    return edges
