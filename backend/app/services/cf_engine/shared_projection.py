from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.domain.model_types import TreeNode
from app.services.cf_engine.normalized_traversal import branch_for_node
from app.services.feature_schema_service import FeatureValue, normalize_feature_value


@dataclass(frozen=True)
class FeatureProposal:
    feature: str
    new_value: FeatureValue
    cost: float
    proposal_type: str


def proposals_for_node_branch(
    node: TreeNode,
    feature: dict[str, Any],
    prepared_vector: dict[str, FeatureValue],
    target_branch: str,
) -> list[FeatureProposal]:
    feature_type = str(feature.get("type", "numeric"))
    if feature_type in {"categorical", "binary"}:
        return categorical_proposals_for_branch(node, feature, prepared_vector, target_branch)

    proposal = numeric_proposal_for_branch(node, feature, prepared_vector, target_branch)
    return [] if proposal is None else [proposal]


def numeric_proposal_for_branch(
    node: TreeNode,
    feature: dict[str, Any],
    prepared_vector: dict[str, FeatureValue],
    target_branch: str,
) -> FeatureProposal | None:
    feature_name = str(feature["name"])
    current = prepared_vector.get(feature_name)
    if _is_missing_value(current):
        return None
    if node.condition.threshold is None:
        return None

    threshold = float(node.condition.threshold)
    current_numeric = float(current)
    integer_like = is_integer_like_feature(feature)
    operator = str(node.condition.operator)

    if operator == "<":
        if integer_like:
            new_value = math.ceil(threshold) - 1 if target_branch == "left" else math.ceil(threshold)
        else:
            new_value = float(np.nextafter(threshold, -np.inf if target_branch == "left" else np.inf))
    elif operator == "<=":
        if integer_like:
            new_value = math.floor(threshold) if target_branch == "left" else math.floor(threshold) + 1
        else:
            new_value = float(np.nextafter(threshold, -np.inf if target_branch == "left" else np.inf))
    else:
        return None

    if not _numeric_value_matches_branch(operator, float(new_value), threshold, target_branch):
        return None
    if not _within_feature_bounds(feature, float(new_value)):
        return None
    if float(current_numeric) == float(new_value):
        return None

    normalized_value = normalize_feature_value(
        feature,
        int(new_value) if integer_like else float(new_value),
        source="counterfactual projection",
    )
    cost = abs(float(normalized_value) - current_numeric)
    return FeatureProposal(
        feature=feature_name,
        new_value=normalized_value,
        cost=max(float(cost), 1e-9),
        proposal_type="integer" if integer_like else "continuous",
    )


def categorical_proposals_for_branch(
    node: TreeNode,
    feature: dict[str, Any],
    prepared_vector: dict[str, FeatureValue],
    target_branch: str,
) -> list[FeatureProposal]:
    feature_name = str(feature["name"])
    current = prepared_vector.get(feature_name)
    if _is_missing_value(current):
        return []

    proposals: list[FeatureProposal] = []
    for option in feature.get("options", []) or []:
        value = option.get("value")
        if value == current:
            continue
        try:
            encoded = float(option["encoded_value"])
        except (KeyError, TypeError, ValueError):
            continue
        if branch_for_node(node, encoded) != target_branch:
            continue

        normalized_value = normalize_feature_value(feature, value, source="counterfactual projection")
        proposals.append(
            FeatureProposal(
                feature=feature_name,
                new_value=normalized_value,
                cost=1.0,
                proposal_type="categorical",
            )
        )
    return proposals


def is_integer_like_feature(feature: dict[str, Any]) -> bool:
    name = str(feature.get("name", "")).lower()
    if name.endswith("-num") or name.endswith("_num") or name.endswith(" count") or name.endswith("_count"):
        return True

    bounds = [feature.get("min_value"), feature.get("max_value")]
    present_bounds = [value for value in bounds if value is not None]
    if len(present_bounds) == 2 and all(_is_integer_value(value) for value in present_bounds):
        return True

    default_value = feature.get("default_value")
    return default_value is not None and _is_integer_value(default_value)


def _numeric_value_matches_branch(operator: str, value: float, threshold: float, target_branch: str) -> bool:
    if operator == "<":
        return value < threshold if target_branch == "left" else value >= threshold
    if operator == "<=":
        return value <= threshold if target_branch == "left" else value > threshold
    return False


def _within_feature_bounds(feature: dict[str, Any], value: float) -> bool:
    min_value = feature.get("min_value")
    max_value = feature.get("max_value")
    if min_value is not None and value < float(min_value):
        return False
    if max_value is not None and value > float(max_value):
        return False
    return True


def _is_integer_value(value: Any) -> bool:
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
