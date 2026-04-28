from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from app.domain.model_types import EnsembleModel, Tree, TreeNode, get_object
from app.services.cf_engine.normalized_traversal import child_id_for_branch, trace_tree_path


@dataclass(frozen=True)
class SubtreeLeafSummary:
    leaf_count: int
    min_value: float
    max_value: float
    mean_value: float


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Any
    estimated_margin_delta: float
    target_directed_improvement: float
    score: float
    cost: float
    num_changes: int
    current_leaf_value: float
    target_leaf_summary: SubtreeLeafSummary


class SubtreeSummaryCache:
    def __init__(self, model: EnsembleModel):
        self._trees = {int(tree.tree_index): tree for tree in model.trees}
        self._cache: dict[tuple[int, int], SubtreeLeafSummary] = {}

    def tree(self, tree_index: int) -> Tree:
        return self._trees[int(tree_index)]

    def summary(self, tree_index: int, object_id: int) -> SubtreeLeafSummary:
        key = (int(tree_index), int(object_id))
        if key not in self._cache:
            values = list(_leaf_values(self.tree(tree_index), int(object_id)))
            if not values:
                self._cache[key] = SubtreeLeafSummary(
                    leaf_count=0,
                    min_value=0.0,
                    max_value=0.0,
                    mean_value=0.0,
                )
            else:
                self._cache[key] = SubtreeLeafSummary(
                    leaf_count=len(values),
                    min_value=min(values),
                    max_value=max(values),
                    mean_value=sum(values) / len(values),
                )
        return self._cache[key]


def rank_candidates(
    *,
    model: EnsembleModel,
    candidates: Sequence[Any],
    encoded_vector: dict[str, float],
    target_class: int,
    cache: SubtreeSummaryCache | None = None,
) -> list[ScoredCandidate]:
    summary_cache = cache or SubtreeSummaryCache(model)
    current_leaf_values = _current_leaf_values(model, encoded_vector)
    scored = [
        _score_candidate(
            candidate=candidate,
            target_class=int(target_class),
            summary_cache=summary_cache,
            current_leaf_values=current_leaf_values,
        )
        for candidate in candidates
    ]
    return sorted(
        scored,
        key=lambda item: (
            item.score,
            item.target_directed_improvement,
            -item.cost,
            -item.num_changes,
        ),
        reverse=True,
    )


def _score_candidate(
    *,
    candidate: Any,
    target_class: int,
    summary_cache: SubtreeSummaryCache,
    current_leaf_values: dict[int, float],
) -> ScoredCandidate:
    tree_index = int(getattr(candidate, "tree_index"))
    tree = summary_cache.tree(tree_index)
    node = tree.nodes[int(getattr(candidate, "node_id"))]
    target_branch = str(getattr(candidate, "target_branch"))
    target_child_id = int(getattr(candidate, "target_child_id", child_id_for_branch(node, target_branch)))
    summary = summary_cache.summary(tree_index, target_child_id)
    current_leaf_value = float(current_leaf_values.get(tree_index, summary.mean_value))

    target_leaf_value = summary.max_value if int(target_class) == 1 else summary.min_value
    estimated_margin_delta = target_leaf_value - current_leaf_value
    target_directed_improvement = estimated_margin_delta if int(target_class) == 1 else -estimated_margin_delta
    cost = max(float(getattr(candidate, "cost", 1.0)), 1e-12)
    num_changes = _num_candidate_changes(candidate)
    score = target_directed_improvement / cost

    if not math.isfinite(score):
        score = float("-inf")

    return ScoredCandidate(
        candidate=candidate,
        estimated_margin_delta=float(estimated_margin_delta),
        target_directed_improvement=float(target_directed_improvement),
        score=float(score),
        cost=cost,
        num_changes=num_changes,
        current_leaf_value=current_leaf_value,
        target_leaf_summary=summary,
    )


def _current_leaf_values(model: EnsembleModel, encoded_vector: dict[str, float]) -> dict[int, float]:
    values: dict[int, float] = {}
    for tree in model.trees:
        _, leaf = trace_tree_path(tree, encoded_vector)
        values[int(tree.tree_index)] = float(leaf.value)
    return values


def _leaf_values(tree: Tree, object_id: int) -> Iterable[float]:
    obj = get_object(tree, int(object_id))
    if obj.is_leaf:
        yield float(obj.value)
        return

    node = obj
    yield from _leaf_values(tree, node.left_child_id)
    yield from _leaf_values(tree, node.right_child_id)


def _num_candidate_changes(candidate: Any) -> int:
    updates = getattr(candidate, "updates", None)
    if isinstance(updates, dict):
        return max(len(updates), 1)
    return 1
