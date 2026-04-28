from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from app.domain.model_types import EnsembleModel
from app.services.cf_engine.normalized_traversal import iter_branch_alternatives
from app.services.cf_engine.shared_projection import proposals_for_node_branch
from app.services.cf_engine.shared_scoring import ScoredCandidate, SubtreeSummaryCache, rank_candidates
from app.services.feature_schema_service import FeatureValue, encode_feature_vector


@dataclass(frozen=True)
class UnifiedPrediction:
    probability: float
    margin: float
    label: int


@dataclass(frozen=True)
class UnifiedCandidate:
    feature: str
    new_value: FeatureValue
    cost: float
    tree_index: int
    node_id: int
    target_branch: str
    target_child_id: int
    proposal_type: str


PredictionEvaluator = Callable[[dict[str, FeatureValue]], UnifiedPrediction]


def generate_unified_counterfactual(
    *,
    model: EnsembleModel,
    feature_metadata: list[dict[str, Any]],
    original_vector: dict[str, FeatureValue],
    prediction_evaluator: PredictionEvaluator,
    threshold: float,
    target_class: int | None,
    max_steps: int = 3,
    exact_eval_top_n: int | None = None,
    debug_label: str = "unified",
) -> dict[str, Any]:
    started = time.perf_counter()
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    original_prediction = prediction_evaluator(dict(original_vector))
    current_probability = float(original_prediction.probability)
    current_margin = float(original_prediction.margin)
    current_prediction = int(current_probability >= float(threshold))
    target = int(1 - current_prediction if target_class is None else target_class)

    feature_by_name = {str(feature["name"]): feature for feature in feature_metadata}
    working_vector = dict(original_vector)
    steps: list[dict[str, Any]] = []
    used_updates: set[tuple[str, str]] = set()
    eval_top_n = int(exact_eval_top_n or _exact_eval_top_n())
    subtree_summary_cache = SubtreeSummaryCache(model)

    total_candidates_generated = 0
    total_candidates_scored = 0
    total_replay_validations = 0
    fallback_used = False
    scoring_seconds = 0.0
    replay_seconds = 0.0
    top_ranked_features: list[str] = []

    for step_index in range(int(max_steps)):
        prediction_before = prediction_evaluator(working_vector)
        probability_before = float(prediction_before.probability)
        margin_before = float(prediction_before.margin)
        label_before = int(probability_before >= float(threshold))
        if label_before == target:
            break

        encoded_vector = encode_feature_vector(feature_metadata, working_vector)
        candidates = _generate_candidates(model, feature_by_name, working_vector, encoded_vector)
        total_candidates_generated += len(candidates)

        scoring_started = time.perf_counter()
        ranked_candidates = rank_candidates(
            model=model,
            candidates=candidates,
            encoded_vector=encoded_vector,
            target_class=target,
            cache=subtree_summary_cache,
        )
        scoring_seconds += time.perf_counter() - scoring_started
        total_candidates_scored += len(ranked_candidates)
        if step_index == 0:
            top_ranked_features = [str(item.candidate.feature) for item in ranked_candidates[:10]]

        top_ranked = ranked_candidates[:eval_top_n]
        remaining_ranked = ranked_candidates[eval_top_n:]
        evaluated, replay_count, replay_elapsed = _evaluate_ranked_candidates(
            ranked_candidates=top_ranked,
            prediction_evaluator=prediction_evaluator,
            working_vector=working_vector,
            used_updates=used_updates,
            probability_before=probability_before,
            threshold=float(threshold),
            target=target,
        )
        total_replay_validations += replay_count
        replay_seconds += replay_elapsed

        top_ranked_reached_target = any(item[3]["label"] == target for item in evaluated)
        if remaining_ranked and not top_ranked_reached_target:
            fallback_used = True
            fallback_evaluated, replay_count, replay_elapsed = _evaluate_ranked_candidates(
                ranked_candidates=remaining_ranked,
                prediction_evaluator=prediction_evaluator,
                working_vector=working_vector,
                used_updates=used_updates,
                probability_before=probability_before,
                threshold=float(threshold),
                target=target,
            )
            evaluated.extend(fallback_evaluated)
            total_replay_validations += replay_count
            replay_seconds += replay_elapsed

        if not evaluated:
            break

        evaluated.sort(key=lambda item: (item[3]["label"] == target, item[0], item[1]), reverse=True)
        _, _, best, best_prediction = evaluated[0]
        old_value = working_vector.get(best.feature)
        working_vector[best.feature] = best.new_value
        used_updates.add((best.feature, repr(best.new_value)))
        steps.append(
            {
                "step": step_index + 1,
                "proposal_type": best.proposal_type,
                "picked_tree_index": int(best.tree_index),
                "node_id": int(best.node_id),
                "feature": best.feature,
                "target_branch": best.target_branch,
                "updates": {best.feature: best.new_value},
                "before": {best.feature: old_value},
                "cost": float(best.cost),
                "margin_before": margin_before,
                "margin_after": best_prediction["margin"],
                "delta_margin": best_prediction["margin"] - margin_before,
                "proba_after": best_prediction["probability"],
                "pred_before": label_before,
                "pred_after": best_prediction["label"],
            }
        )

        if best_prediction["label"] == target:
            break

    final_prediction = prediction_evaluator(working_vector)
    final_probability = float(final_prediction.probability)
    final_margin = float(final_prediction.margin)
    final_label = int(final_probability >= float(threshold))
    counterfactuals: list[dict[str, Any]] = []
    if steps and final_label == target:
        changes = [
            {"feature": feature, "old_value": original_vector.get(feature), "new_value": working_vector.get(feature)}
            for feature in model.feature_names
            if _values_differ(original_vector.get(feature), working_vector.get(feature))
        ]
        pruned_changes, removed_changes, pruned_vector = _prune_changes(
            prediction_evaluator=prediction_evaluator,
            original_vector=original_vector,
            changes=changes,
            target_class=target,
            threshold=float(threshold),
        )
        pruned_prediction = prediction_evaluator(pruned_vector)
        pruned_probability = float(pruned_prediction.probability)
        pruned_margin = float(pruned_prediction.margin)
        pruned_label = int(pruned_probability >= float(threshold))
        if pruned_label == target:
            counterfactuals.append(
                {
                    "new_probability": pruned_probability,
                    "new_margin": pruned_margin,
                    "new_prediction": pruned_label,
                    "cost": float(sum(float(step["cost"]) for step in steps)),
                    "changes": pruned_changes,
                    "steps": steps,
                    "original_num_changes": len(changes),
                    "pruned_num_changes": len(pruned_changes),
                    "removed_changes": removed_changes,
                    "is_minimal_after_pruning": _is_minimal(
                        prediction_evaluator=prediction_evaluator,
                        original_vector=original_vector,
                        changes=pruned_changes,
                        target_class=target,
                        threshold=float(threshold),
                    ),
                }
            )

    runtime_ms = 1000.0 * (time.perf_counter() - started)
    diagnostics = {
        "candidate_count": total_candidates_generated,
        "candidates_generated": total_candidates_generated,
        "candidates_scored": total_candidates_scored,
        "exact_eval_top_n": eval_top_n,
        "replay_validations": total_replay_validations,
        "replay_count": total_replay_validations,
        "fallback_used": fallback_used,
        "scoring_ms": 1000.0 * scoring_seconds,
        "replay_ms": 1000.0 * replay_seconds,
        "top_ranked_features": top_ranked_features,
    }
    if _counterfactual_debug_enabled():
        print(
            f"[counterfactual:{debug_label}] "
            f"candidates_generated={total_candidates_generated} "
            f"candidates_scored={total_candidates_scored} "
            f"exact_eval_top_n={eval_top_n} "
            f"replay_validations={total_replay_validations} "
            f"fallback_used={fallback_used} "
            f"scoring_ms={1000.0 * scoring_seconds:.3f} "
            f"replay_ms={1000.0 * replay_seconds:.3f} "
            f"total_ms={runtime_ms:.3f}"
        )

    return {
        "current_probability": current_probability,
        "current_margin": current_margin,
        "current_prediction": current_prediction,
        "threshold": float(threshold),
        "counterfactuals": counterfactuals,
        "runtime_ms": runtime_ms,
        "diagnostics": diagnostics,
    }


def _generate_candidates(
    model: EnsembleModel,
    feature_by_name: dict[str, dict[str, Any]],
    prepared_vector: dict[str, FeatureValue],
    encoded_vector: dict[str, float],
) -> list[UnifiedCandidate]:
    candidates: list[UnifiedCandidate] = []
    for tree in model.trees:
        for alternative in iter_branch_alternatives(tree, encoded_vector):
            feature = feature_by_name.get(alternative.node.condition.feature_name)
            if feature is None:
                continue
            for proposal in proposals_for_node_branch(
                alternative.node,
                feature,
                prepared_vector,
                alternative.target_branch,
            ):
                candidates.append(
                    UnifiedCandidate(
                        feature=proposal.feature,
                        new_value=proposal.new_value,
                        cost=proposal.cost,
                        tree_index=int(alternative.tree_index),
                        node_id=int(alternative.node.node_id),
                        target_branch=alternative.target_branch,
                        target_child_id=int(alternative.target_child_id),
                        proposal_type=proposal.proposal_type,
                    )
                )
    return candidates


def _evaluate_ranked_candidates(
    *,
    ranked_candidates: list[ScoredCandidate],
    prediction_evaluator: PredictionEvaluator,
    working_vector: dict[str, FeatureValue],
    used_updates: set[tuple[str, str]],
    probability_before: float,
    threshold: float,
    target: int,
) -> tuple[list[tuple[float, float, UnifiedCandidate, dict[str, Any]]], int, float]:
    started = time.perf_counter()
    replay_validations = 0
    evaluated: list[tuple[float, float, UnifiedCandidate, dict[str, Any]]] = []
    for scored_candidate in ranked_candidates:
        candidate = scored_candidate.candidate
        update_key = (candidate.feature, repr(candidate.new_value))
        if update_key in used_updates:
            continue
        trial_vector = dict(working_vector)
        trial_vector[candidate.feature] = candidate.new_value
        try:
            replay_validations += 1
            trial_prediction = prediction_evaluator(trial_vector)
        except Exception:
            continue

        trial_probability = float(trial_prediction.probability)
        delta_probability = trial_probability - probability_before
        direction = delta_probability if target == 1 else -delta_probability
        if direction <= 0:
            continue
        score = direction / max(float(candidate.cost), 1e-12)
        evaluated.append(
            (
                score,
                direction,
                candidate,
                {
                    "probability": trial_probability,
                    "margin": float(trial_prediction.margin),
                    "label": int(trial_probability >= float(threshold)),
                },
            )
        )
    return evaluated, replay_validations, time.perf_counter() - started


def _prune_changes(
    *,
    prediction_evaluator: PredictionEvaluator,
    original_vector: dict[str, FeatureValue],
    changes: list[dict[str, Any]],
    target_class: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, FeatureValue]]:
    active_changes = list(changes)
    removed: list[dict[str, Any]] = []
    index = 0
    while index < len(active_changes):
        trial_changes = active_changes[:index] + active_changes[index + 1 :]
        trial_vector = _apply_changes(original_vector, trial_changes)
        prediction = prediction_evaluator(trial_vector)
        probability = float(prediction.probability)
        label = int(probability >= float(threshold))
        if label == int(target_class):
            removed.append(active_changes[index])
            active_changes = trial_changes
            continue
        index += 1
    return active_changes, removed, _apply_changes(original_vector, active_changes)


def _is_minimal(
    *,
    prediction_evaluator: PredictionEvaluator,
    original_vector: dict[str, FeatureValue],
    changes: list[dict[str, Any]],
    target_class: int,
    threshold: float,
) -> bool:
    for index in range(len(changes)):
        trial_vector = _apply_changes(original_vector, changes[:index] + changes[index + 1 :])
        prediction = prediction_evaluator(trial_vector)
        if int(float(prediction.probability) >= float(threshold)) == int(target_class):
            return False
    return True


def _apply_changes(
    original_vector: dict[str, FeatureValue],
    changes: list[dict[str, Any]],
) -> dict[str, FeatureValue]:
    vector = dict(original_vector)
    for change in changes:
        vector[str(change["feature"])] = change["new_value"]
    return vector


def _values_differ(old_value: Any, new_value: Any) -> bool:
    if old_value is None and new_value is None:
        return False
    try:
        if pd.isna(old_value) and pd.isna(new_value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(old_value != new_value)


def _counterfactual_debug_enabled() -> bool:
    return str(os.getenv("COUNTERFACTUAL_DEBUG", "")).lower() in {"1", "true", "yes", "on"}


def _exact_eval_top_n() -> int:
    raw_value = os.getenv("COUNTERFACTUAL_EXACT_EVAL_TOP_N")
    if raw_value is None:
        return 64
    try:
        return max(int(raw_value), 1)
    except ValueError:
        return 64
