from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import xgboost as xgb

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


class XGBoostAdapterError(ValueError):
    pass


class XGBoostModelAdapter:
    """Convert XGBoost binary trees into the app's generic ensemble schema."""

    model_family = "xgboost"
    model_type = "xgboost_binary_classifier"
    supported_extensions = (".json", ".ubj", ".xgb", ".bst", ".model", ".pkl", ".pickle", ".joblib")

    @classmethod
    def load_artifacts_from_path(cls, model_path: str | Path) -> tuple[EnsembleModel, xgb.Booster]:
        booster = cls._load_booster(model_path)
        return cls.from_booster(booster), booster

    @classmethod
    def load_from_path(cls, model_path: str | Path) -> EnsembleModel:
        model, _ = cls.load_artifacts_from_path(model_path)
        return model

    @classmethod
    def from_booster(cls, booster: xgb.Booster) -> EnsembleModel:
        config = cls._load_config(booster)
        cls._validate_supported_binary_classifier(config)
        feature_names = cls._extract_feature_names(booster, config)
        dumped_trees = [json.loads(raw_tree) for raw_tree in booster.get_dump(dump_format="json", with_stats=True)]
        trees = [cls._normalize_tree(tree_payload, index, feature_names) for index, tree_payload in enumerate(dumped_trees)]

        model = EnsembleModel(
            model_id=uuid4().hex,
            metadata=EnsembleMetadata(
                model_family=cls.model_family,
                model_type=cls.model_type,
                feature_names=feature_names,
                class_labels=[0, 1],
                decision_threshold=0.5,
                global_feature_importance=cls._extract_global_feature_importance(booster, feature_names),
            ),
            base_score=cls._extract_base_margin(config),
            trees=trees,
        )
        compute_model_radial_layout(
            model,
            RadialLayoutConfig(inner_radius=0.08, outer_radius=1.0, depth_exponent=1.0),
        )
        return model

    @staticmethod
    def compute_local_feature_importance(
        booster: xgb.Booster,
        feature_names: list[str],
        feature_vector: dict[str, float],
    ) -> list[FeatureImportanceEntry]:
        ordered_row = np.array([[feature_vector.get(feature_name, np.nan) for feature_name in feature_names]], dtype=float)
        matrix = xgb.DMatrix(ordered_row, feature_names=feature_names)
        contributions = booster.predict(matrix, pred_contribs=True)
        contribution_values = contributions[0] if contributions.ndim > 1 else contributions
        return [
            FeatureImportanceEntry(feature_name=feature_name, value=float(contribution_values[index]))
            for index, feature_name in enumerate(feature_names)
        ]

    @classmethod
    def _load_booster(cls, model_path: str | Path) -> xgb.Booster:
        path = Path(model_path)
        booster = xgb.Booster()
        try:
            booster.load_model(str(path))
            return booster
        except xgb.core.XGBoostError:
            pass

        loaded_object = cls._load_pickled_object(path)
        if isinstance(loaded_object, xgb.Booster):
            return loaded_object
        if hasattr(loaded_object, "get_booster"):
            booster = loaded_object.get_booster()
            if isinstance(booster, xgb.Booster):
                return booster
        raise XGBoostAdapterError("Expected an XGBoost Booster or XGBClassifier-compatible object.")

    @staticmethod
    def _load_pickled_object(path: Path) -> Any:
        try:
            import joblib

            return joblib.load(path)
        except ModuleNotFoundError:
            with path.open("rb") as file:
                return pickle.load(file)
        except Exception as joblib_exc:
            try:
                with path.open("rb") as file:
                    return pickle.load(file)
            except Exception as pickle_exc:
                raise XGBoostAdapterError(
                    f"Failed to load XGBoost model file as native model or pickle/joblib artifact: {pickle_exc}"
                ) from joblib_exc

    @staticmethod
    def _load_config(booster: xgb.Booster) -> dict[str, Any]:
        try:
            return json.loads(booster.save_config())
        except (json.JSONDecodeError, xgb.core.XGBoostError) as exc:
            raise XGBoostAdapterError(f"Failed to read XGBoost model config: {exc}") from exc

    @classmethod
    def _validate_supported_binary_classifier(cls, config: dict[str, Any]) -> None:
        learner = config.get("learner", {})
        objective = str(learner.get("objective", {}).get("name", "")).lower()
        if objective not in {"binary:logistic", "binary:logitraw"}:
            raise XGBoostAdapterError("Only XGBoost binary classification models are supported.")

        learner_model_param = learner.get("learner_model_param", {})
        num_class = int(cls._parse_numeric_text(learner_model_param.get("num_class"), default=0))
        if num_class not in {0, 1}:
            raise XGBoostAdapterError("Only binary XGBoost models with one class output are supported.")

        booster_name = str(learner.get("gradient_booster", {}).get("name", "")).lower()
        if booster_name == "dart":
            raise XGBoostAdapterError("XGBoost DART models are out of scope for this adapter.")

    @classmethod
    def _extract_feature_names(cls, booster: xgb.Booster, config: dict[str, Any]) -> list[str]:
        if booster.feature_names:
            return [str(name) for name in booster.feature_names]

        learner_model_param = config.get("learner", {}).get("learner_model_param", {})
        num_feature = int(cls._parse_numeric_text(learner_model_param.get("num_feature"), default=booster.num_features()))
        if num_feature <= 0:
            num_feature = booster.num_features()
        return [f"f{index}" for index in range(num_feature)]

    @classmethod
    def _extract_base_margin(cls, config: dict[str, Any]) -> float:
        learner_model_param = config.get("learner", {}).get("learner_model_param", {})
        base_score = cls._parse_numeric_text(learner_model_param.get("base_score"), default=0.5)
        base_score = min(max(base_score, 1e-15), 1.0 - 1e-15)
        return math.log(base_score / (1.0 - base_score))

    @classmethod
    def _normalize_tree(cls, tree_payload: dict[str, Any], tree_index: int, feature_names: list[str]) -> Tree:
        tree = Tree(tree_index=tree_index, root_id=int(tree_payload["nodeid"]))

        def _walk(node_payload: dict[str, Any], depth: int) -> int:
            node_id = int(node_payload["nodeid"])
            if "leaf" in node_payload:
                tree.leaves[node_id] = TreeLeaf(
                    leaf_id=node_id,
                    depth=depth,
                    value=float(node_payload["leaf"]),
                    cover=cls._optional_float(node_payload.get("cover")),
                )
                return node_id

            children_by_id = {int(child["nodeid"]): child for child in node_payload.get("children", [])}
            yes_child_id = int(node_payload["yes"])
            no_child_id = int(node_payload["no"])
            missing_child_id = int(node_payload["missing"])
            if yes_child_id not in children_by_id or no_child_id not in children_by_id:
                raise XGBoostAdapterError(f"Tree {tree_index} has a split node with missing yes/no children.")

            _walk(children_by_id[yes_child_id], depth + 1)
            _walk(children_by_id[no_child_id], depth + 1)

            feature_name = str(node_payload["split"])
            feature_index = cls._feature_index(feature_name, feature_names)
            tree.nodes[node_id] = TreeNode(
                node_id=node_id,
                depth=depth,
                condition=SplitCondition(
                    feature_name=feature_name,
                    feature_index=feature_index,
                    operator="<",
                    threshold=float(node_payload["split_condition"]),
                    default_child="left" if missing_child_id == yes_child_id else "right",
                    missing_value_strategy="nan",
                ),
                left_child_id=yes_child_id,
                right_child_id=no_child_id,
                gain=cls._optional_float(node_payload.get("gain")),
                cover=cls._optional_float(node_payload.get("cover")),
            )
            return node_id

        _walk(tree_payload, 0)
        return tree

    @staticmethod
    def _feature_index(feature_name: str, feature_names: list[str]) -> int:
        if feature_name in feature_names:
            return feature_names.index(feature_name)
        if feature_name.startswith("f") and feature_name[1:].isdigit():
            return int(feature_name[1:])
        raise XGBoostAdapterError(f"XGBoost tree references unknown feature '{feature_name}'.")

    @staticmethod
    def _extract_global_feature_importance(
        booster: xgb.Booster,
        feature_names: list[str],
    ) -> list[FeatureImportanceEntry]:
        scores = booster.get_score(importance_type="gain")
        return [
            FeatureImportanceEntry(feature_name=feature_name, value=float(scores.get(feature_name, 0.0)))
            for feature_name in feature_names
        ]

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _parse_numeric_text(value: Any, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)

        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
        if match is None:
            return default
        return float(match.group(0))
