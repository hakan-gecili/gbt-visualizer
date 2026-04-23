from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import lightgbm as lgb

from app.domain.model_types import (
    EnsembleMetadata,
    EnsembleModel,
    FeatureImportanceEntry,
    RadialLayoutConfig,
    SplitCondition,
    Tree,
    TreeLeaf,
    TreeNode,
    compute_model_radial_layout,
)


class LightGBMAdapterError(ValueError):
    pass


class LightGBMModelAdapter:
    """Convert raw LightGBM model dumps into the app's generic ensemble schema.

    Future model families such as XGBoost should implement the same boundary:
    parse the model-specific dump once here, then hand the rest of the app a
    populated `EnsembleModel`.
    """

    model_family = "lightgbm"
    model_type = "lightgbm_binary_classifier"
    supported_extensions = (".txt",)

    @classmethod
    def load_artifacts_from_path(cls, model_path: str | Path) -> tuple[EnsembleModel, lgb.Booster]:
        booster = lgb.Booster(model_file=str(model_path))
        return cls.from_booster(booster), booster

    @classmethod
    def load_from_path(cls, model_path: str | Path) -> EnsembleModel:
        model, _ = cls.load_artifacts_from_path(model_path)
        return model

    @classmethod
    def from_booster(cls, booster: lgb.Booster) -> EnsembleModel:
        return cls.from_dumped_model(booster.dump_model(), booster)

    @classmethod
    def from_dumped_model(cls, model_dump: dict[str, Any], booster: lgb.Booster | None = None) -> EnsembleModel:
        objective = str(model_dump.get("objective", "")).lower()
        if "binary" not in objective:
            raise LightGBMAdapterError("Only LightGBM binary classification models are supported.")
        if int(model_dump.get("num_class", 1)) != 1:
            raise LightGBMAdapterError("Only binary classification models with one raw-score output are supported.")

        feature_names = [str(name) for name in model_dump.get("feature_names", [])]
        trees = [cls._normalize_tree(tree_info, feature_names) for tree_info in model_dump.get("tree_info", [])]
        model = EnsembleModel(
            model_id=uuid4().hex,
            metadata=EnsembleMetadata(
                model_type=cls.model_type,
                feature_names=feature_names,
                class_labels=[0, 1],
                decision_threshold=0.5,
                global_feature_importance=cls._extract_global_feature_importance(booster, feature_names),
            ),
            base_score=0.0,
            trees=trees,
        )
        compute_model_radial_layout(
            model,
            RadialLayoutConfig(inner_radius=0.08, outer_radius=1.0, depth_exponent=1.0),
        )
        return model

    @staticmethod
    def compute_local_feature_importance(
        booster: lgb.Booster,
        feature_names: list[str],
        feature_vector: dict[str, float],
    ) -> list[FeatureImportanceEntry]:
        ordered_row = np.array([[feature_vector.get(feature_name, np.nan) for feature_name in feature_names]], dtype=float)
        contributions = booster.predict(ordered_row, pred_contrib=True)
        contribution_values = contributions[0] if contributions.ndim > 1 else contributions
        return [
            FeatureImportanceEntry(feature_name=feature_name, value=float(contribution_values[index]))
            for index, feature_name in enumerate(feature_names)
        ]

    @classmethod
    def _normalize_tree(cls, tree_info: dict[str, Any], feature_names: list[str]) -> Tree:
        tree_index = int(tree_info["tree_index"])
        tree = Tree(tree_index=tree_index, root_id=-1)
        next_id = 0

        def _allocate_id() -> int:
            nonlocal next_id
            value = next_id
            next_id += 1
            return value

        def _walk(node_payload: dict[str, Any], depth: int) -> int:
            object_id = _allocate_id()
            if "leaf_value" in node_payload:
                tree.leaves[object_id] = TreeLeaf(
                    leaf_id=object_id,
                    depth=depth,
                    value=float(node_payload["leaf_value"]),
                )
                return object_id

            left_child_id = _walk(node_payload["left_child"], depth + 1)
            right_child_id = _walk(node_payload["right_child"], depth + 1)
            split_feature_index = int(node_payload["split_feature"])
            if split_feature_index < 0 or split_feature_index >= len(feature_names):
                raise LightGBMAdapterError(
                    f"Tree {tree_index} references split feature index {split_feature_index} outside feature_names."
                )

            threshold_value = cls._parse_threshold(node_payload.get("threshold"))
            tree.nodes[object_id] = TreeNode(
                node_id=object_id,
                depth=depth,
                condition=SplitCondition(
                    feature_name=feature_names[split_feature_index],
                    feature_index=split_feature_index,
                    operator=str(node_payload.get("decision_type", "<=")),
                    threshold=threshold_value,
                    category_values=cls._parse_category_values(
                        str(node_payload.get("decision_type", "<=")),
                        threshold_value,
                    ),
                    default_child="left" if bool(node_payload.get("default_left", True)) else "right",
                    missing_value_strategy=str(node_payload.get("missing_type", "None")),
                ),
                left_child_id=left_child_id,
                right_child_id=right_child_id,
            )
            return object_id

        tree.root_id = _walk(tree_info["tree_structure"], 0)
        return tree

    @staticmethod
    def _parse_threshold(raw_threshold: Any) -> float | str | None:
        if raw_threshold is None:
            return None
        if isinstance(raw_threshold, (int, float)):
            return float(raw_threshold)

        threshold_text = str(raw_threshold).strip()
        if not threshold_text:
            return threshold_text

        try:
            return float(threshold_text)
        except ValueError:
            return threshold_text

    @staticmethod
    def _parse_category_values(decision_type: str, threshold: float | str | None) -> list[float]:
        if decision_type != "==" or not isinstance(threshold, str):
            return []
        return [float(item) for item in threshold.split("||") if item != ""]

    @staticmethod
    def _extract_global_feature_importance(
        booster: lgb.Booster | None,
        feature_names: list[str],
    ) -> list[FeatureImportanceEntry]:
        if booster is None:
            return []

        gain_importance = booster.feature_importance(importance_type="gain")
        return [
            FeatureImportanceEntry(feature_name=feature_name, value=float(gain_importance[index]))
            for index, feature_name in enumerate(feature_names)
        ]
