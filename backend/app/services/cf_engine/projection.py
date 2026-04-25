# counterfactuals/projection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set
import numpy as np
import pandas as pd


def next_after_up(x: float) -> float:
    return float(np.nextafter(float(x), np.inf))


def next_after_down(x: float) -> float:
    return float(np.nextafter(float(x), -np.inf))


@dataclass(frozen=True)
class IntSpec:
    min_val: Optional[int] = None
    max_val: Optional[int] = None
    step: int = 1
    weight: float = 1.0  # multiplies step-based cost


@dataclass(frozen=True)
class OneHotGroupSpec:
    cols: List[str]
    allow_all_zero: bool = False
    switch_cost: float = 1.0
    fallback_col: Optional[str] = None
    immutable: bool = False


@dataclass(frozen=True)
class CategoricalSpec:
    encoded_values: List[float]
    switch_cost: float = 1.0
    immutable: bool = False


@dataclass(frozen=True)
class Proposal:
    updates: Dict[str, Any]  # may include multiple columns for one-hot
    cost: float
    meta: Dict[str, Any]


class ProjectionEngine:
    def __init__(
        self,
        feature_scales: Optional[Dict[str, float]] = None,
        min_scale: float = 1e-12,
        int_specs: Optional[Dict[str, IntSpec]] = None,
        onehot_groups: Optional[Dict[str, OneHotGroupSpec]] = None,
        categorical_specs: Optional[Dict[str, CategoricalSpec]] = None,
        immutable_features: Optional[List[str]] = None,
        default_cont_weight: float = 1.0,
    ):
        self.feature_scales = feature_scales or {}
        self.min_scale = float(min_scale)
        self.int_specs = int_specs or {}
        self.onehot_groups = onehot_groups or {}
        self.categorical_specs = categorical_specs or {}
        self.immutable_features = set(immutable_features or [])
        self.default_cont_weight = float(default_cont_weight)

        self.onehot_col_to_group: Dict[str, str] = {}
        for g, spec in self.onehot_groups.items():
            for c in spec.cols:
                self.onehot_col_to_group[c] = g

    def _scale(self, feat: str) -> float:
        s = float(self.feature_scales.get(feat, 1.0))
        return max(s, self.min_scale)

    def propose_for_split(
        self,
        x: pd.Series,
        feature: str,
        threshold: float,
        want_right: bool,
        eps_strategy: str = "nextafter",
        decision_type: str = "<=",
    ) -> Optional[Proposal]:
        if feature in self.immutable_features:
            return None

        if feature in self.onehot_col_to_group:
            return self._propose_onehot(x, feature, threshold, want_right)

        if feature in self.categorical_specs:
            return self._propose_categorical(x, feature, threshold, want_right, decision_type)

        if feature in self.int_specs:
            return self._propose_integer(x, feature, threshold, want_right)

        return self._propose_continuous(x, feature, threshold, want_right, eps_strategy)

    def _parse_category_threshold(self, threshold: Any) -> Set[float]:
        if isinstance(threshold, str):
            parts = [p for p in threshold.split("||") if p != ""]
            return {float(p) for p in parts}
        return {float(threshold)}

    def _propose_categorical(self, x: pd.Series, feature: str, threshold: Any, want_right: bool, decision_type: str) -> Optional[Proposal]:
        spec = self.categorical_specs[feature]
        if spec.immutable:
            return None

        cur = x.get(feature, np.nan)
        if pd.isna(cur):
            return None
        cur_val = float(cur)
        values = [float(v) for v in spec.encoded_values]
        if not values:
            return None

        dtyp = str(decision_type)
        if dtyp == "==":
            threshold_values = self._parse_category_threshold(threshold)
            allowed = [v for v in values if v in threshold_values] if not want_right else [v for v in values if v not in threshold_values]
        elif dtyp == "<=":
            thr = float(threshold)
            allowed = [v for v in values if v > thr] if want_right else [v for v in values if v <= thr]
        else:
            return None

        if not allowed:
            return None

        if any(abs(cur_val - v) < 1e-12 for v in allowed):
            return None

        new_val = allowed[0]
        return Proposal(
            updates={feature: new_val},
            cost=float(spec.switch_cost),
            meta={"type": "categorical", "cur": cur_val, "new": new_val},
        )

    def _propose_continuous(self, x: pd.Series, feature: str, threshold: float, want_right: bool, eps_strategy: str) -> Optional[Proposal]:
        cur = x.get(feature, np.nan)
        if pd.isna(cur):
            return None
        cur = float(cur)

        if want_right:
            new = next_after_up(threshold) if eps_strategy == "nextafter" else float(threshold + 1e-12)
            if not (new > threshold):
                return None
        else:
            new = next_after_down(threshold) if eps_strategy == "nextafter" else float(threshold - 1e-12)
            if not (new <= threshold):
                return None

        if new == cur:
            return None

        delta = new - cur
        cost = self.default_cont_weight * abs(delta) / self._scale(feature)

        return Proposal(updates={feature: new}, cost=float(cost), meta={"type": "continuous", "delta": float(delta)})

    def _propose_integer(self, x: pd.Series, feature: str, threshold: float, want_right: bool) -> Optional[Proposal]:
        spec = self.int_specs[feature]
        cur = x.get(feature, np.nan)
        if pd.isna(cur):
            return None
        cur_int = int(round(float(cur)))

        thr = float(threshold)
        step = int(spec.step)
        if step <= 0:
            raise ValueError(f"Invalid step for {feature}: {step}")

        if want_right:
            new_int = int(np.floor(thr)) + 1
        else:
            new_int = int(np.floor(thr))

        if spec.min_val is not None:
            new_int = max(new_int, int(spec.min_val))
        if spec.max_val is not None:
            new_int = min(new_int, int(spec.max_val))

        if want_right and not (new_int > thr):
            return None
        if (not want_right) and not (new_int <= thr):
            return None

        if new_int == cur_int:
            return None

        step_delta = abs(new_int - cur_int) / step
        cost = spec.weight * step_delta

        return Proposal(
            updates={feature: int(new_int)},
            cost=float(cost),
            meta={"type": "integer", "cur": cur_int, "new": int(new_int), "steps": float(step_delta)},
        )

    def _propose_onehot(self, x: pd.Series, col: str, threshold: float, want_right: bool) -> Optional[Proposal]:
        group = self.onehot_col_to_group[col]
        spec = self.onehot_groups[group]
        if spec.immutable:
            return None

        thr = float(threshold)
        desired = 1 if want_right else 0

        if want_right and not (desired > thr):
            return None
        if (not want_right) and not (desired <= thr):
            return None

        cols = list(spec.cols)
        vals = {c: float(x.get(c, 0.0)) for c in cols}
        active_cols = [c for c in cols if vals.get(c, 0.0) >= 0.5]
        active = active_cols[0] if active_cols else None

        cur_val = int(1 if vals.get(col, 0.0) >= 0.5 else 0)
        if desired == cur_val:
            return None

        updates: Dict[str, Any] = {}

        if desired == 1:
            for c in cols:
                updates[c] = 1 if c == col else 0
            cost = float(spec.switch_cost) if active != col else 0.0
            if cost == 0.0:
                return None
            return Proposal(updates=updates, cost=cost, meta={"type": "onehot", "group": group, "to": col, "from": active})

        # desired == 0
        if active != col:
            return None

        if spec.allow_all_zero:
            for c in cols:
                updates[c] = 0
            return Proposal(updates=updates, cost=float(spec.switch_cost), meta={"type": "onehot", "group": group, "to": None, "from": active})

        # must switch to another category
        if spec.fallback_col is not None and spec.fallback_col in cols and spec.fallback_col != col:
            repl = spec.fallback_col
        else:
            others = [c for c in cols if c != col]
            if not others:
                return None
            repl = others[0]

        for c in cols:
            updates[c] = 1 if c == repl else 0

        return Proposal(updates=updates, cost=float(spec.switch_cost), meta={"type": "onehot", "group": group, "to": repl, "from": active})
