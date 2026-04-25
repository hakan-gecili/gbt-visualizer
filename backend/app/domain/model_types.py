from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, exp, pi, sin
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class SplitCondition:
    feature_name: str
    feature_index: int
    operator: str
    threshold: float | str | None = None
    category_values: List[float] = field(default_factory=list)
    default_child: str = "left"
    missing_value_strategy: str = "none"


@dataclass
class TreeNode:
    node_id: int
    depth: int
    condition: SplitCondition
    left_child_id: int
    right_child_id: int
    gain: float | None = None
    cover: float | None = None
    subtree_leaf_count: int = 0
    angle: float = 0.0
    radius: float = 0.0
    x: float = 0.0
    y: float = 0.0
    is_leaf: bool = False

    # Compatibility properties preserve the existing API serialization shape
    # while shared backend logic moves to the generic condition payload above.
    @property
    def split_feature(self) -> str:
        return self.condition.feature_name

    @property
    def threshold(self) -> float | str | None:
        return self.condition.threshold

    @property
    def decision_type(self) -> str:
        return self.condition.operator

    @property
    def default_left(self) -> bool:
        return self.condition.default_child == "left"

    @property
    def missing_type(self) -> str:
        return self.condition.missing_value_strategy


@dataclass
class TreeLeaf:
    leaf_id: int
    depth: int
    value: float
    cover: float | None = None
    subtree_leaf_count: int = 1
    angle: float = 0.0
    radius: float = 0.0
    x: float = 0.0
    y: float = 0.0
    leaf_order: int = 0
    is_leaf: bool = True


@dataclass
class Tree:
    tree_index: int
    root_id: int
    nodes: Dict[int, TreeNode] = field(default_factory=dict)
    leaves: Dict[int, TreeLeaf] = field(default_factory=dict)
    max_depth: int = 0
    num_leaves: int = 0
    sector_start_angle: float = 0.0
    sector_end_angle: float = 0.0


@dataclass
class EnsembleMetadata:
    model_family: str
    model_type: str
    feature_names: List[str]
    class_labels: List[int | str] = field(default_factory=lambda: [0, 1])
    decision_threshold: float = 0.5
    global_feature_importance: List["FeatureImportanceEntry"] = field(default_factory=list)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


@dataclass
class EnsembleModel:
    model_id: str
    metadata: EnsembleMetadata
    base_score: float
    trees: List[Tree]

    @property
    def num_trees(self) -> int:
        return len(self.trees)

    @property
    def feature_names(self) -> List[str]:
        return self.metadata.feature_names

    @property
    def model_type(self) -> str:
        return self.metadata.model_type

    @property
    def model_family(self) -> str:
        return self.metadata.model_family

    @property
    def decision_threshold(self) -> float:
        return self.metadata.decision_threshold


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
    predicted_label: int
    decision_threshold: float
    local_feature_importance: List["FeatureImportanceEntry"]
    tree_results: List[TraversalResult]


@dataclass
class FeatureImportanceEntry:
    feature_name: str
    value: float


@dataclass
class PathCondition:
    node_id: int
    feature_name: str
    operator: str
    threshold: float | str | None
    branch_direction: str


@dataclass
class RadialLayoutConfig:
    inner_radius: float = 0.08
    outer_radius: float = 1.0
    depth_exponent: float = 1.0


def is_leaf_id(tree: Tree, object_id: int) -> bool:
    return object_id in tree.leaves


def get_object(tree: Tree, object_id: int) -> TreeNode | TreeLeaf:
    if object_id in tree.nodes:
        return tree.nodes[object_id]
    if object_id in tree.leaves:
        return tree.leaves[object_id]
    raise KeyError(f"Object ID {object_id} not found in tree {tree.tree_index}.")


def compute_subtree_leaf_counts(tree: Tree) -> int:
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


def compute_max_depth(tree: Tree) -> int:
    def _depth(object_id: int) -> int:
        obj = get_object(tree, object_id)
        if obj.is_leaf:
            return obj.depth
        return max(_depth(obj.left_child_id), _depth(obj.right_child_id))

    tree.max_depth = _depth(tree.root_id)
    return tree.max_depth


def assign_leaf_order(tree: Tree) -> None:
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


def assign_tree_sectors(model: EnsembleModel) -> None:
    if model.num_trees == 0:
        return
    angle_offset = -pi / 2
    delta = 2 * pi / model.num_trees
    for index, tree in enumerate(model.trees):
        tree.sector_start_angle = index * delta + angle_offset
        tree.sector_end_angle = (index + 1) * delta + angle_offset


def compute_tree_radial_layout(tree: Tree, config: RadialLayoutConfig) -> None:
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
    model: EnsembleModel,
    config: Optional[RadialLayoutConfig] = None,
) -> None:
    layout_config = config or RadialLayoutConfig()
    assign_tree_sectors(model)
    for tree in model.trees:
        compute_subtree_leaf_counts(tree)
        compute_max_depth(tree)
        assign_leaf_order(tree)
        compute_tree_radial_layout(tree, layout_config)


def extract_edges(tree: Tree) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for node in tree.nodes.values():
        edges.append((node.node_id, node.left_child_id))
        edges.append((node.node_id, node.right_child_id))
    return edges


def summarize_layout(model: EnsembleModel) -> Tuple[int, int]:
    max_tree_depth = max((tree.max_depth for tree in model.trees), default=0)
    total_leaves = sum(tree.num_leaves for tree in model.trees)
    return max_tree_depth, total_leaves


def extract_path_conditions(
    tree: Tree,
    path_node_ids: Sequence[int],
    selected_leaf_id: int,
) -> List[PathCondition]:
    traversal_sequence = [*path_node_ids, selected_leaf_id]
    path_conditions: List[PathCondition] = []

    for index, node_id in enumerate(path_node_ids):
        node = tree.nodes.get(node_id)
        next_id = traversal_sequence[index + 1] if index + 1 < len(traversal_sequence) else None
        if node is None or next_id is None:
            continue

        if next_id == node.left_child_id:
            branch_direction = "left"
        elif next_id == node.right_child_id:
            branch_direction = "right"
        else:
            continue

        path_conditions.append(
            PathCondition(
                node_id=node.node_id,
                feature_name=node.condition.feature_name,
                operator=node.condition.operator,
                threshold=node.condition.threshold,
                branch_direction=branch_direction,
            )
        )

    return path_conditions
