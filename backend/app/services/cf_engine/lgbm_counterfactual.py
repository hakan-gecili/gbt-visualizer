# counterfactuals/lgbm_counterfactual.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import pandas as pd

from .projection import ProjectionEngine, Proposal
from .moves_lookup import Move, load_moves_lookup


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = np.exp(-z)
        return float(1.0 / (1.0 + ez))
    else:
        ez = np.exp(z)
        return float(ez / (1.0 + ez))


def logit(p: float, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p / (1 - p)))


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


def node_default_left(node: Dict[str, Any]) -> bool:
    return bool(node.get("default_left", True))


def _parse_category_threshold(threshold: Any) -> set[float]:
    if isinstance(threshold, str):
        return {float(p) for p in threshold.split("||") if p != ""}
    return {float(threshold)}


def choose_direction_numeric(node: Dict[str, Any], x_val: Any) -> bool:
    """
    LightGBM numeric: x <= thr -> left, else right.
    True => left, False => right.
    """
    dtyp = node_decision_type(node)
    if pd.isna(x_val):
        return node_default_left(node)
    thr = node_threshold(node)
    if dtyp == "<=":
        return bool(float(x_val) <= float(thr))
    if dtyp == "==":
        return bool(float(x_val) in _parse_category_threshold(thr))
    raise NotImplementedError(f"Unsupported decision_type={dtyp}")


def trace_path(tree_root: Dict[str, Any], x: pd.Series, feature_names: List[str]) -> Tuple[List[Tuple[Dict[str, Any], bool]], Dict[str, Any]]:
    path: List[Tuple[Dict[str, Any], bool]] = []
    node = tree_root
    while not is_leaf(node):
        j = node_feature_idx(node)
        fname = feature_names[j]
        went_left = choose_direction_numeric(node, x[fname])
        path.append((node, went_left))
        left, right = get_children(node)
        node = left if went_left else right
    return path, node


def simulate_subtree(subtree_root: Dict[str, Any], x: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
    node = subtree_root
    while not is_leaf(node):
        j = node_feature_idx(node)
        fname = feature_names[j]
        went_left = choose_direction_numeric(node, x[fname])
        left, right = get_children(node)
        node = left if went_left else right
    return node


@dataclass
class Candidate:
    tree_index: int
    depth_from_root: int
    split_index: Optional[int]

    feature_name: str
    threshold: Any
    current_side: str

    updates: Dict[str, Any]
    cost: float
    proposal_type: str

    leaf_value_current: float
    leaf_value_switched: float
    delta_leaf: float

    margin_current: float
    delta_margin_tree: float
    proba_current: float
    proba_after_tree_switch_approx: float
    delta_proba_approx: float

    moves_toward_target: bool
    score: float


@dataclass
class LookupCandidate:
    tree_index: int
    leaf_index: int
    feature_name: str
    threshold: Any
    want_right: bool

    updates: Dict[str, Any]
    cost: float
    proposal_type: str

    leaf_value_current: float
    subtree_max_leaf_value: float
    delta_leaf_upper_bound: float

    margin_current: float
    delta_margin_est: float
    proba_current: float
    delta_proba_est: float

    moves_toward_target: bool
    score: float
    source: str


@dataclass(frozen=True)
class CFTriggerParams:
    delta_margin: float = 0.5
    top_risk_pct: float = 0.002


def should_generate_cf(
    margin: float,
    tau_margin: float,
    flags: Optional[Dict[str, Any]] = None,
    risk_rank: Optional[float] = None,
    params: Optional[CFTriggerParams] = None,
) -> Tuple[bool, str, str]:
    params = params or CFTriggerParams()
    flags = flags or {}
    rules_flag = bool(flags.get("rules_flag", False))
    top_risk = False
    if risk_rank is not None:
        top_risk = float(risk_rank) <= float(params.top_risk_pct)

    d = float(tau_margin - margin)

    if margin >= tau_margin:
        return True, "PRED_FRAUD", "SLA"
    if 0.0 < d <= float(params.delta_margin):
        return True, "NEAR_BOUNDARY", "SLA"
    if rules_flag or top_risk:
        return True, "RULES_FLAG" if rules_flag else "TOP_RISK", "SLA"
    return False, "SKIPPED", "NONE"


class LightGBMCounterfactualExplainer:
    def __init__(self, model, projection: ProjectionEngine, moves_lookup: Optional[Dict[Tuple[int, int], List[Move]]] = None):
        self.model = model
        self.booster = model.booster_
        self.dump = self.booster.dump_model()

        self.feature_names = list(self.booster.feature_name())
        self.trees = [t["tree_structure"] for t in self.dump["tree_info"]]

        self.projection = projection
        self.bias_: Optional[float] = None
        self.leaf_scale_: Optional[float] = None
        self.moves_lookup: Optional[Dict[Tuple[int, int], List[Move]]] = moves_lookup

    def load_moves_lookup(self, path: str) -> None:
        self.moves_lookup = load_moves_lookup(path)

    def _row_series(self, x_row_df: pd.DataFrame) -> pd.Series:
        if not isinstance(x_row_df, pd.DataFrame) or len(x_row_df) != 1:
            raise ValueError("x_row_df must be a pandas DataFrame with exactly 1 row.")
        missing = [c for c in self.feature_names if c not in x_row_df.columns]
        if missing:
            raise ValueError(f"x_row_df missing required columns (first 10): {missing[:10]}")
        return x_row_df.iloc[0][self.feature_names]

    def calibrate_margin_decomposition(self, x_row_df: pd.DataFrame) -> Tuple[float, float]:
        x = self._row_series(x_row_df)
        margin = float(self.model.predict(x_row_df, raw_score=True)[0])
        contrib = self.model.predict(x_row_df, pred_contrib=True)
        bias = float(contrib[0, -1])

        sum_leaf = 0.0
        for tree_root in self.trees:
            try:
                _, leaf = trace_path(tree_root, x, self.feature_names)
            except NotImplementedError:
                continue
            sum_leaf += float(leaf["leaf_value"])

        leaf_scale = 1.0 if abs(sum_leaf) < 1e-12 else float((margin - bias) / sum_leaf)
        self.bias_, self.leaf_scale_ = bias, leaf_scale
        return bias, leaf_scale

    def _ensure_calibrated(self, x_row_df: pd.DataFrame) -> None:
        if self.bias_ is None or self.leaf_scale_ is None:
            self.calibrate_margin_decomposition(x_row_df)

    def generate_candidates(
        self,
        x_row_df: pd.DataFrame,
        thr_proba: float,
        target_label: Optional[int] = None,
        eps_strategy: str = "nextafter",
        max_nodes_per_tree: Optional[int] = 10,
        top_k_per_tree: int = 5,
        exact_eval_top_n: int = 30,
    ) -> pd.DataFrame:
        self._ensure_calibrated(x_row_df)
        assert self.leaf_scale_ is not None

        x = self._row_series(x_row_df)

        boundary = logit(thr_proba)
        margin = float(self.model.predict(x_row_df, raw_score=True)[0])
        p = sigmoid(margin)
        current_pred = int(margin >= boundary)

        if target_label is None:
            target_label = 1 - current_pred
        target_label = int(target_label)

        desired_sign = +1.0 if target_label == 1 else -1.0

        cands: List[Candidate] = []

        for t_idx, tree_root in enumerate(self.trees):
            try:
                path, leaf_cur = trace_path(tree_root, x, self.feature_names)
            except NotImplementedError:
                continue

            leaf_value_cur = float(leaf_cur["leaf_value"])
            nodes_to_scan = list(reversed(path))
            if max_nodes_per_tree is not None:
                nodes_to_scan = nodes_to_scan[: int(max_nodes_per_tree)]

            per_tree: List[Candidate] = []

            for rev_depth, (node, went_left) in enumerate(nodes_to_scan):
                dtyp = node_decision_type(node)
                if dtyp not in {"<=", "=="}:
                    continue

                j = node_feature_idx(node)
                fname = self.feature_names[j]
                thr = node_threshold(node)

                want_right = went_left
                current_side = "left" if went_left else "right"

                proposal: Optional[Proposal] = self.projection.propose_for_split(
                    x=x,
                    feature=fname,
                    threshold=thr,
                    want_right=want_right,
                    eps_strategy=eps_strategy,
                    decision_type=dtyp,
                )
                if proposal is None:
                    continue

                x_mod = x.copy()
                for k, v in proposal.updates.items():
                    x_mod[k] = v

                left_child, right_child = get_children(node)
                sibling_root = right_child if want_right else left_child

                try:
                    leaf_sw = simulate_subtree(sibling_root, x_mod, self.feature_names)
                except NotImplementedError:
                    continue

                leaf_value_sw = float(leaf_sw["leaf_value"])
                delta_leaf = leaf_value_sw - leaf_value_cur

                delta_margin = float(self.leaf_scale_ * delta_leaf)
                p_after = sigmoid(margin + delta_margin)
                delta_p = p_after - p

                moves_toward_target = (desired_sign * delta_margin) > 0
                score = float((desired_sign * delta_margin) / max(proposal.cost, 1e-12))

                split_index = node.get("split_index", None)
                depth_from_root = len(path) - 1 - rev_depth

                per_tree.append(
                    Candidate(
                        tree_index=t_idx,
                        depth_from_root=int(depth_from_root),
                        split_index=int(split_index) if split_index is not None else None,
                        feature_name=fname,
                        threshold=thr,
                        current_side=current_side,
                        updates=dict(proposal.updates),
                        cost=float(proposal.cost),
                        proposal_type=str(proposal.meta.get("type", "unknown")),
                        leaf_value_current=leaf_value_cur,
                        leaf_value_switched=leaf_value_sw,
                        delta_leaf=delta_leaf,
                        margin_current=margin,
                        delta_margin_tree=delta_margin,
                        proba_current=p,
                        proba_after_tree_switch_approx=p_after,
                        delta_proba_approx=delta_p,
                        moves_toward_target=bool(moves_toward_target),
                        score=score,
                    )
                )

            if per_tree:
                per_tree.sort(key=lambda c: c.score, reverse=True)
                cands.extend(per_tree[: int(top_k_per_tree)])

        if not cands:
            return pd.DataFrame()

        df = pd.DataFrame([c.__dict__ for c in cands])
        df.sort_values(["moves_toward_target", "score", "cost"], ascending=[False, False, True], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Exact eval for top N candidates (apply updates and re-predict full model)
        if exact_eval_top_n and exact_eval_top_n > 0:
            top_n = min(int(exact_eval_top_n), len(df))
            margin_exact = []
            proba_exact = []
            delta_margin_exact = []
            delta_proba_exact = []
            for i in range(top_n):
                r = df.iloc[i]
                x2 = x_row_df.copy()
                for k, v in r["updates"].items():
                    x2.loc[x2.index[0], k] = v
                m2 = float(self.model.predict(x2, raw_score=True)[0])
                p2 = float(self.model.predict_proba(x2)[0, 1])
                margin_exact.append(m2)
                proba_exact.append(p2)
                delta_margin_exact.append(m2 - margin)
                delta_proba_exact.append(p2 - float(df.loc[i, "proba_current"]))
            df.loc[: top_n - 1, "margin_after_exact"] = margin_exact
            df.loc[: top_n - 1, "proba_after_exact"] = proba_exact
            df.loc[: top_n - 1, "delta_margin_exact"] = delta_margin_exact
            df.loc[: top_n - 1, "delta_proba_exact"] = delta_proba_exact

        df["thr_proba"] = float(thr_proba)
        df["boundary_margin"] = float(boundary)
        df["current_pred_deployed"] = int(current_pred)
        df["target_label"] = int(target_label)
        df["leaf_scale_factor"] = float(self.leaf_scale_)
        df["bias_estimate"] = float(self.bias_ if self.bias_ is not None else 0.0)

        return df

    def generate_candidates_lookup(
        self,
        x_row_df: pd.DataFrame,
        thr_proba: float,
        target_label: Optional[int] = None,
        max_candidates_per_row: int = 2000,
    ) -> pd.DataFrame:
        if not self.moves_lookup:
            return pd.DataFrame()

        self._ensure_calibrated(x_row_df)
        assert self.leaf_scale_ is not None

        x = self._row_series(x_row_df)

        boundary = logit(thr_proba)
        margin = float(self.model.predict(x_row_df, raw_score=True)[0])
        p = sigmoid(margin)
        current_pred = int(margin >= boundary)

        if target_label is None:
            target_label = 1 - current_pred
        target_label = int(target_label)
        desired_sign = +1.0 if target_label == 1 else -1.0

        leaf_ids = np.asarray(self.model.predict(x_row_df, pred_leaf=True))
        cands: List[LookupCandidate] = []

        for t_idx in range(leaf_ids.shape[1]):
            leaf_id = int(leaf_ids[0, t_idx])
            moves = self.moves_lookup.get((int(t_idx), int(leaf_id)), [])
            if not moves:
                continue

            for move in moves:
                proposal = self.projection.propose_for_split(
                    x=x,
                    feature=move.feature_name,
                    threshold=move.threshold,
                    want_right=move.want_right,
                    decision_type=getattr(move, "decision_type", "<="),
                )
                if proposal is None:
                    continue

                delta_leaf = float(move.delta_leaf_upper_bound)
                delta_margin_est = float(self.leaf_scale_ * delta_leaf)
                delta_p_est = float(p * (1.0 - p) * delta_margin_est)

                moves_toward_target = (desired_sign * delta_margin_est) > 0
                score = float((desired_sign * delta_margin_est) / max(proposal.cost, 1e-12))

                cands.append(
                    LookupCandidate(
                        tree_index=int(t_idx),
                        leaf_index=int(leaf_id),
                        feature_name=move.feature_name,
                        threshold=move.threshold,
                        want_right=bool(move.want_right),
                        updates=dict(proposal.updates),
                        cost=float(proposal.cost),
                        proposal_type=str(proposal.meta.get("type", "lookup")),
                        leaf_value_current=float(move.current_leaf_value),
                        subtree_max_leaf_value=float(move.subtree_max_leaf_value),
                        delta_leaf_upper_bound=delta_leaf,
                        margin_current=margin,
                        delta_margin_est=delta_margin_est,
                        proba_current=p,
                        delta_proba_est=delta_p_est,
                        moves_toward_target=bool(moves_toward_target),
                        score=score,
                        source="lookup",
                    )
                )

        if not cands:
            return pd.DataFrame()

        df = pd.DataFrame([c.__dict__ for c in cands])
        df.sort_values(["moves_toward_target", "score", "cost"], ascending=[False, False, True], inplace=True)
        df.reset_index(drop=True, inplace=True)

        if max_candidates_per_row and len(df) > int(max_candidates_per_row):
            df = df.iloc[: int(max_candidates_per_row)].reset_index(drop=True)

        df["thr_proba"] = float(thr_proba)
        df["boundary_margin"] = float(boundary)
        df["current_pred_deployed"] = int(current_pred)
        df["target_label"] = int(target_label)
        df["leaf_scale_factor"] = float(self.leaf_scale_)
        df["bias_estimate"] = float(self.bias_ if self.bias_ is not None else 0.0)

        return df

    def greedy_counterfactual(
        self,
        x_row_df: pd.DataFrame,
        thr_proba: float,
        target_label: Optional[int] = None,
        max_steps: int = 5,
        margin_safety: float = 1e-3,
        top_k_per_tree: int = 5,
        exact_eval_top_n: int = 30,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._ensure_calibrated(x_row_df)
        boundary = logit(thr_proba)

        x_cf = x_row_df.copy()
        steps: List[Dict[str, Any]] = []

        for step in range(int(max_steps)):
            margin = float(self.model.predict(x_cf, raw_score=True)[0])
            pred = int(margin >= boundary)
            tgt = (1 - pred) if target_label is None else int(target_label)

            if tgt == 1 and margin >= boundary + margin_safety:
                break
            if tgt == 0 and margin <= boundary - margin_safety:
                break

            cand_df = self.generate_candidates(
                x_cf, thr_proba=thr_proba, target_label=tgt,
                top_k_per_tree=top_k_per_tree,
                exact_eval_top_n=exact_eval_top_n,
            )
            if cand_df.empty:
                break

            # Filter to only candidates that move toward target
            # Use exact evaluation if available (prioritize accuracy over approximation)
            if "delta_margin_exact" in cand_df.columns:
                # Only consider rows with exact evaluation (non-NaN)
                with_exact = cand_df[cand_df["delta_margin_exact"].notna()].copy()
                if not with_exact.empty:
                    if tgt == 1:
                        actually_moving = with_exact[with_exact["delta_margin_exact"] > 0]
                    else:
                        actually_moving = with_exact[with_exact["delta_margin_exact"] < 0]
                    if not actually_moving.empty:
                        # Re-sort by exact delta and cost
                        actually_moving = actually_moving.copy()
                        actually_moving["exact_score"] = actually_moving["delta_margin_exact"].abs() / actually_moving["cost"].clip(lower=1e-12)
                        moving_toward = actually_moving.sort_values("exact_score", ascending=False)
                    else:
                        # No exact candidates move toward target - stop
                        break
                else:
                    # No exact evaluations available, fall back to approximation
                    moving_toward = cand_df[cand_df["moves_toward_target"]]
            else:
                moving_toward = cand_df[cand_df["moves_toward_target"]]

            if moving_toward.empty:
                # No candidates move toward target, stop greedy search
                break

            best = moving_toward.iloc[0]
            updates: Dict[str, Any] = dict(best["updates"])

            before_vals = {k: x_cf.loc[x_cf.index[0], k] for k in updates.keys()}
            for k, v in updates.items():
                x_cf.loc[x_cf.index[0], k] = v

            margin_new = float(self.model.predict(x_cf, raw_score=True)[0])
            p_new = float(self.model.predict_proba(x_cf)[0, 1])
            pred_new = int(margin_new >= boundary)

            steps.append({
                "step": step + 1,
                "proposal_type": best["proposal_type"],
                "picked_tree_index": int(best["tree_index"]),
                "feature": best["feature_name"],
                "updates": updates,
                "before": before_vals,
                "cost": float(best["cost"]),
                "margin_before": margin,
                "margin_after": margin_new,
                "delta_margin": margin_new - margin,
                "proba_after": p_new,
                "pred_before": pred,
                "pred_after": pred_new,
            })

            if tgt == 1 and margin_new >= boundary + margin_safety:
                break
            if tgt == 0 and margin_new <= boundary - margin_safety:
                break

        return x_cf, pd.DataFrame(steps)

    def generate_counterfactuals(
        self,
        x_row_df: pd.DataFrame,
        thr_proba: float,
        target_label: Optional[int] = None,
        k: int = 1,
        max_candidates_per_row: int = 2000,
        max_finalists_to_verify: int = 50,
        beam_width: int = 5,
        max_depth: int = 3,
        budget_ms: float = 200.0,
        diversity_threshold: float = 0.3,
        trigger_params: Optional[CFTriggerParams] = None,
        flags: Optional[Dict[str, Any]] = None,
        risk_rank: Optional[float] = None,
    ) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
        start = time.perf_counter()
        metrics = {
            "runtime_ms": 0.0,
            "candidates_considered": 0,
            "verification_calls": 0,
            "success": False,
            "trigger_reason": None,
            "mode": None,
        }

        boundary = logit(thr_proba)
        margin = float(self.model.predict(x_row_df, raw_score=True)[0])
        trigger, reason, mode = should_generate_cf(margin, boundary, flags=flags, risk_rank=risk_rank, params=trigger_params)
        metrics["trigger_reason"] = reason
        metrics["mode"] = mode
        if not trigger:
            metrics["runtime_ms"] = 1000.0 * (time.perf_counter() - start)
            return [], metrics

        if target_label is None:
            target_label = int(1 - int(margin >= boundary))
        target_label = int(target_label)
        desired_sign = +1.0 if target_label == 1 else -1.0

        cand_df = self.generate_candidates_lookup(
            x_row_df,
            thr_proba=thr_proba,
            target_label=target_label,
            max_candidates_per_row=max_candidates_per_row,
        )
        if cand_df.empty:
            cand_df = self.generate_candidates(
                x_row_df,
                thr_proba=thr_proba,
                target_label=target_label,
                top_k_per_tree=5,
                exact_eval_top_n=0,
            )

        if cand_df.empty:
            metrics["runtime_ms"] = 1000.0 * (time.perf_counter() - start)
            return [], metrics

        candidates = []
        for _, row in cand_df.iterrows():
            delta_est = float(row["delta_margin_est"]) if "delta_margin_est" in row else float(row["delta_margin_tree"])
            candidates.append({
                "updates": dict(row["updates"]),
                "cost": float(row["cost"]),
                "delta_margin_est": delta_est,
                "score": float(row["score"]),
                "features": set(dict(row["updates"]).keys()),
            })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        if max_candidates_per_row and len(candidates) > int(max_candidates_per_row):
            candidates = candidates[: int(max_candidates_per_row)]

        metrics["candidates_considered"] = int(len(candidates))
        results: List[pd.DataFrame] = []

        def meets_target(m: float) -> bool:
            return bool(desired_sign * (m - boundary) >= 0.0)

        def within_budget() -> bool:
            return (time.perf_counter() - start) * 1000.0 <= float(budget_ms)

        if k <= 1:
            for cand in candidates[: int(max_finalists_to_verify)]:
                if not within_budget():
                    break
                x2 = x_row_df.copy()
                for k_feat, v in cand["updates"].items():
                    x2.loc[x2.index[0], k_feat] = v
                m2 = float(self.model.predict(x2, raw_score=True)[0])
                metrics["verification_calls"] += 1
                if meets_target(m2):
                    results.append(x2)
                    metrics["success"] = True
                    break

            metrics["runtime_ms"] = 1000.0 * (time.perf_counter() - start)
            return results, metrics

        def jaccard(a: set[str], b: set[str]) -> float:
            if not a and not b:
                return 0.0
            inter = len(a & b)
            union = len(a | b)
            return 1.0 - float(inter / max(union, 1))

        beam = [{
            "x": x_row_df.copy(),
            "cost": 0.0,
            "features": set(),
            "est_gain": 0.0,
            "steps": [],
        }]

        for _depth in range(int(max_depth)):
            if not within_budget():
                break
            next_beam = []
            for state in beam:
                if not within_budget():
                    break
                for cand in candidates[: int(max_finalists_to_verify)]:
                    if not within_budget():
                        break
                    if state["features"] & cand["features"]:
                        continue

                    x2 = state["x"].copy()
                    for k_feat, v in cand["updates"].items():
                        x2.loc[x2.index[0], k_feat] = v

                    m2 = float(self.model.predict(x2, raw_score=True)[0])
                    metrics["verification_calls"] += 1

                    new_features = state["features"] | cand["features"]
                    new_cost = float(state["cost"] + cand["cost"])
                    new_est_gain = float(state["est_gain"] + cand["delta_margin_est"])

                    if meets_target(m2):
                        too_close = False
                        for cf in results:
                            cf_feats = set(cf.columns[cf.iloc[0] != x_row_df.iloc[0]])
                            if jaccard(new_features, cf_feats) < float(diversity_threshold):
                                too_close = True
                                break
                        if not too_close:
                            results.append(x2)
                            metrics["success"] = True
                            if len(results) >= int(k):
                                metrics["runtime_ms"] = 1000.0 * (time.perf_counter() - start)
                                return results, metrics
                        continue

                    score = float(abs(new_est_gain) / max(new_cost, 1e-12))
                    next_beam.append({
                        "x": x2,
                        "cost": new_cost,
                        "features": new_features,
                        "est_gain": new_est_gain,
                        "score": score,
                    })

            if not next_beam:
                break

            next_beam.sort(key=lambda s: s["score"], reverse=True)
            beam = next_beam[: int(beam_width)]

        metrics["runtime_ms"] = 1000.0 * (time.perf_counter() - start)
        return results, metrics
