from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

from app.domain.model_types import Tree, TreeLeaf, TreeNode, get_object


BranchName = str


@dataclass(frozen=True)
class PathStep:
    tree_index: int
    node: TreeNode
    branch: BranchName
    child_id: int


@dataclass(frozen=True)
class BranchAlternative:
    tree_index: int
    node: TreeNode
    current_branch: BranchName
    target_branch: BranchName
    current_child_id: int
    target_child_id: int


def is_missing_for_split(value: float, missing_value_strategy: str) -> bool:
    if math.isnan(float(value)):
        return True
    return str(missing_value_strategy).lower() == "zero" and float(value) == 0.0


def take_true_branch(node: TreeNode, encoded_value: float) -> bool:
    condition = node.condition
    if condition.operator == "<":
        if condition.threshold is None:
            raise ValueError(f"Node for feature '{condition.feature_name}' is missing a numeric threshold.")
        return float(encoded_value) < float(condition.threshold)
    if condition.operator == "<=":
        if condition.threshold is None:
            raise ValueError(f"Node for feature '{condition.feature_name}' is missing a numeric threshold.")
        return float(encoded_value) <= float(condition.threshold)
    if condition.operator == "==":
        return float(encoded_value) in category_split_values(node)
    raise ValueError(f"Unsupported split operator '{condition.operator}'.")


def branch_for_node(node: TreeNode, encoded_value: float) -> BranchName:
    if is_missing_for_split(encoded_value, node.condition.missing_value_strategy):
        return node.condition.default_child
    return "left" if take_true_branch(node, encoded_value) else "right"


def child_id_for_branch(node: TreeNode, branch: BranchName) -> int:
    if branch == "left":
        return int(node.left_child_id)
    if branch == "right":
        return int(node.right_child_id)
    raise ValueError(f"Unsupported branch '{branch}'.")


def sibling_branch(branch: BranchName) -> BranchName:
    if branch == "left":
        return "right"
    if branch == "right":
        return "left"
    raise ValueError(f"Unsupported branch '{branch}'.")


def category_split_values(node: TreeNode) -> set[float]:
    if node.condition.category_values:
        return {float(value) for value in node.condition.category_values}
    threshold = node.condition.threshold
    if threshold is None:
        return set()
    if isinstance(threshold, str):
        return {float(item) for item in threshold.split("||") if item != ""}
    return {float(threshold)}


def trace_tree_path(tree: Tree, encoded_vector: dict[str, float]) -> tuple[list[PathStep], TreeLeaf]:
    current_id = tree.root_id
    path: list[PathStep] = []

    while True:
        obj = get_object(tree, current_id)
        if obj.is_leaf:
            return path, obj

        node = obj
        encoded_value = encoded_vector.get(node.condition.feature_name, math.nan)
        branch = branch_for_node(node, encoded_value)
        child_id = child_id_for_branch(node, branch)
        path.append(
            PathStep(
                tree_index=int(tree.tree_index),
                node=node,
                branch=branch,
                child_id=child_id,
            )
        )
        current_id = child_id


def iter_branch_alternatives(tree: Tree, encoded_vector: dict[str, float]) -> Iterator[BranchAlternative]:
    path, _ = trace_tree_path(tree, encoded_vector)
    for step in path:
        target_branch = sibling_branch(step.branch)
        yield BranchAlternative(
            tree_index=step.tree_index,
            node=step.node,
            current_branch=step.branch,
            target_branch=target_branch,
            current_child_id=step.child_id,
            target_child_id=child_id_for_branch(step.node, target_branch),
        )
