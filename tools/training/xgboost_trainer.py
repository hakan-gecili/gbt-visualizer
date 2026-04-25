from __future__ import annotations

from pathlib import Path
from typing import Any

import xgboost as xgb

from training.common import (
    build_export_from_config,
    build_training_inputs,
    encode_feature_frame_for_numeric_training,
    prepare_numeric_features_for_training,
    split_train_validation_indices,
    validate_output_dir,
    write_common_artifacts,
)


MODEL_FILENAME = "model.json"


def train_xgboost(csv_path: str | Path, target_column: str, output_dir: str | Path, config: Any) -> dict[str, Any]:
    output_path = Path(output_dir)
    validate_output_dir(output_path, MODEL_FILENAME)
    inputs = build_training_inputs(csv_path, target_column, config)
    training_features = encode_feature_frame_for_numeric_training(
        inputs["feature_frame"],
        inputs["feature_definitions"],
    )

    model, metrics = _train_model(training_features, inputs["target"], config)
    output_path.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(output_path / MODEL_FILENAME))

    export_frame = build_export_from_config(
        inputs["feature_frame"],
        inputs["target"],
        inputs["target_column"],
        config,
    )
    metadata = {
        "model_family": "xgboost",
        "model_file": MODEL_FILENAME,
        "training_parameters": _training_params(config),
        "metrics": metrics,
        "target_mapping": inputs["target_mapping"],
    }
    write_common_artifacts(output_path, inputs["schema"], export_frame, metadata)
    return {
        "schema": inputs["schema"],
        "export_frame": export_frame,
        "target_column": inputs["target_column"],
        "target_mapping": inputs["target_mapping"],
        "model_filename": MODEL_FILENAME,
        "metadata": metadata,
    }


def _train_model(features, target, config: Any) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    training_features = prepare_numeric_features_for_training(features)
    train_indices, validation_indices = split_train_validation_indices(
        target,
        getattr(config, "test_size", 0.2),
        getattr(config, "random_state", 42),
    )
    model = xgb.XGBClassifier(**_training_params(config))
    eval_set = None
    if validation_indices:
        eval_set = [
            (training_features.loc[train_indices], target.loc[train_indices]),
            (training_features.loc[validation_indices], target.loc[validation_indices]),
        ]
    model.fit(
        training_features.loc[train_indices],
        target.loc[train_indices],
        eval_set=eval_set,
        verbose=False,
    )
    return model, _collect_metrics(model)


def _training_params(config: Any) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": getattr(config, "n_estimators", 100),
        "max_depth": None if getattr(config, "max_depth", -1) == -1 else getattr(config, "max_depth", -1),
        "learning_rate": getattr(config, "learning_rate", 0.1),
        "random_state": getattr(config, "random_state", 42),
        "n_jobs": 1,
    }


def _collect_metrics(model: xgb.XGBClassifier) -> dict[str, Any]:
    try:
        return model.evals_result()
    except (AttributeError, xgb.core.XGBoostError):
        return {}
