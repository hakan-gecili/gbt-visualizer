from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb

from schema_utils import FEATURE_TYPE_NUMERIC
from training.common import (
    build_export_from_config,
    build_training_inputs,
    split_train_validation_indices,
    validate_output_dir,
    write_common_artifacts,
)


MODEL_FILENAME = "model.txt"


def train_lightgbm(csv_path: str | Path, target_column: str, output_dir: str | Path, config: Any) -> dict[str, Any]:
    output_path = Path(output_dir)
    validate_output_dir(output_path, MODEL_FILENAME)
    inputs = build_training_inputs(csv_path, target_column, config)

    booster = _train_model(
        inputs["feature_frame"],
        inputs["target"],
        inputs["feature_definitions"],
        config,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path / MODEL_FILENAME))

    export_frame = build_export_from_config(
        inputs["feature_frame"],
        inputs["target"],
        inputs["target_column"],
        config,
    )
    metadata = {
        "model_family": "lightgbm",
        "model_file": MODEL_FILENAME,
        "training_parameters": _training_params(config),
        "metrics": {},
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


def _train_model(
    features,
    target,
    feature_definitions: list[tuple[str, dict[str, Any]]],
    config: Any,
) -> lgb.Booster:
    train_indices, validation_indices = split_train_validation_indices(
        target,
        getattr(config, "test_size", 0.2),
        getattr(config, "random_state", 42),
    )
    categorical_features = [
        str(feature["name"])
        for _, feature in feature_definitions
        if str(feature["type"]) != FEATURE_TYPE_NUMERIC
    ]

    train_dataset = lgb.Dataset(
        features.loc[train_indices],
        label=target.loc[train_indices],
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    valid_sets = [train_dataset]
    valid_names = ["train"]
    if validation_indices:
        validation_dataset = lgb.Dataset(
            features.loc[validation_indices],
            label=target.loc[validation_indices],
            categorical_feature=categorical_features,
            reference=train_dataset,
            free_raw_data=False,
        )
        valid_sets.append(validation_dataset)
        valid_names.append("validation")

    return lgb.train(
        params=_training_params(config),
        train_set=train_dataset,
        num_boost_round=getattr(config, "n_estimators", 100),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.log_evaluation(period=0)],
    )


def _training_params(config: Any) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": getattr(config, "learning_rate", 0.1),
        "num_leaves": getattr(config, "num_leaves", 31),
        "max_depth": getattr(config, "max_depth", -1),
        "seed": getattr(config, "random_state", 42),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
