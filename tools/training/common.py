from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from schema_utils import (
    FEATURE_TYPE_NUMERIC,
    SchemaGenerationError,
    build_schema,
    iter_feature_definitions,
    load_overrides,
    resolve_target_column,
    stringify_value,
    unique_non_null_values,
)


class ExampleBuildError(ValueError):
    pass


def parse_drop_columns(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def merged_overrides(raw_overrides: dict[str, Any], drop_columns: list[str], target_column: str) -> dict[str, Any]:
    merged = dict(raw_overrides)
    merged["features"] = dict(raw_overrides.get("features", {}))
    merged["exclude_columns"] = [*raw_overrides.get("exclude_columns", []), *drop_columns]
    if "target_column" not in merged:
        merged["target_column"] = target_column
    return merged


def validate_output_dir(output_dir: Path, model_filename: str) -> None:
    if not output_dir.exists():
        return

    allowed_names = {model_filename, "dataset.csv", "feature_schema.json", "metadata.json"}
    extra_artifacts = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix in {".txt", ".csv", ".json"} and path.name not in allowed_names
    )
    if extra_artifacts:
        raise ExampleBuildError(
            "Output directory contains extra .txt/.csv/.json files that would make the example ambiguous: "
            f"{extra_artifacts}"
        )


def read_csv(data_path: Path) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(data_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ExampleBuildError(f"Failed to read CSV '{data_path}': {exc}") from exc
    if dataframe.empty:
        raise ExampleBuildError(f"CSV dataset has no rows: {data_path}")
    return dataframe


def validate_target(dataframe: pd.DataFrame, target_column: str) -> dict[str, Any]:
    if target_column not in dataframe.columns:
        raise ExampleBuildError(
            f"Target column '{target_column}' was not found in dataset columns: {list(map(str, dataframe.columns))}"
        )

    target_series = dataframe[target_column]
    if target_series.isna().any():
        raise ExampleBuildError(f"Target column '{target_column}' contains missing values.")

    target_values = unique_non_null_values(target_series)
    if len(target_values) != 2:
        raise ExampleBuildError(
            f"Binary classification requires exactly 2 distinct target values, found {len(target_values)}: "
            f"{[stringify_value(value) for value in target_values]}"
        )

    target_mapping = {stringify_value(value): index for index, value in enumerate(target_values)}
    encoded_target = target_series.map(lambda value: target_mapping[stringify_value(value)]).astype(int)
    return {
        "series": encoded_target,
        "mapping": target_mapping,
    }


def ensure_unique_feature_names(feature_definitions: list[tuple[str, dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for raw_column, feature in feature_definitions:
        feature_name = str(feature["name"])
        original = seen.get(feature_name)
        if original is not None:
            raise ExampleBuildError(
                f"Feature name collision after overrides: '{raw_column}' and '{original}' both map to '{feature_name}'."
            )
        seen[feature_name] = raw_column


def prepare_feature_frame(
    dataframe: pd.DataFrame,
    feature_definitions: list[tuple[str, dict[str, Any]]],
) -> pd.DataFrame:
    prepared_columns: dict[str, pd.Series] = {}

    for raw_column, feature in feature_definitions:
        feature_name = str(feature["name"])
        feature_type = str(feature["type"])
        series = dataframe[raw_column]

        if feature_type == FEATURE_TYPE_NUMERIC:
            prepared_columns[feature_name] = pd.to_numeric(series, errors="coerce")
            continue

        categories = [str(option["value"]) for option in feature.get("options", [])]
        normalized = series.map(lambda value: np.nan if pd.isna(value) else stringify_value(value))
        invalid_values = sorted({value for value in normalized.dropna().unique() if value not in set(categories)})
        if invalid_values:
            raise ExampleBuildError(
                f"Column '{raw_column}' contains values not covered by schema options: {invalid_values}"
            )
        prepared_columns[feature_name] = pd.Series(
            pd.Categorical(normalized, categories=categories),
            index=series.index,
            name=feature_name,
        )

    return pd.DataFrame(prepared_columns)


def encode_feature_frame_for_numeric_training(
    features: pd.DataFrame,
    feature_definitions: list[tuple[str, dict[str, Any]]],
) -> pd.DataFrame:
    encoded_columns: dict[str, pd.Series] = {}
    for _, feature in feature_definitions:
        feature_name = str(feature["name"])
        if str(feature["type"]) == FEATURE_TYPE_NUMERIC:
            encoded_columns[feature_name] = pd.to_numeric(features[feature_name], errors="coerce")
            continue

        encoded_by_value = {
            str(option["value"]): float(option["encoded_value"])
            for option in feature.get("options", [])
        }
        encoded_columns[feature_name] = features[feature_name].map(
            lambda value: np.nan if pd.isna(value) else encoded_by_value[str(value)]
        )
    return pd.DataFrame(encoded_columns, index=features.index)


def prepare_numeric_features_for_training(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()

    for col in features.columns:
        if str(features[col].dtype) == "category":
            features[col] = features[col].cat.codes.astype(float)
        elif features[col].dtype == "object":
            features[col] = features[col].astype("category").cat.codes.astype(float)

    return features


def split_train_validation_indices(
    target: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    if not 0 <= test_size < 1:
        raise ExampleBuildError(f"--test-size must be between 0 and 1, received {test_size}")
    if test_size == 0 or len(target) < 3:
        return list(target.index), []

    rng = random.Random(random_state)
    validation_indices: list[int] = []

    for class_value in sorted(target.unique()):
        class_indices = [int(index) for index in target.index[target == class_value].tolist()]
        if len(class_indices) < 2:
            return list(target.index), []
        rng.shuffle(class_indices)
        requested = int(round(len(class_indices) * test_size))
        holdout_count = max(1, requested)
        holdout_count = min(holdout_count, len(class_indices) - 1)
        validation_indices.extend(class_indices[:holdout_count])

    validation_set = set(validation_indices)
    training_indices = [int(index) for index in target.index.tolist() if int(index) not in validation_set]
    if len(set(target.loc[training_indices])) < 2:
        return list(target.index), []
    return training_indices, sorted(validation_indices)


def build_export_frame(
    features: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if max_rows <= 0:
        raise ExampleBuildError(f"--max-rows must be positive, received {max_rows}")

    export_frame = features.copy()
    for column in export_frame.columns:
        if isinstance(export_frame[column].dtype, pd.CategoricalDtype):
            export_frame[column] = export_frame[column].astype(object)
    export_frame[target_column] = target.astype(int)

    if len(export_frame) <= max_rows:
        return export_frame

    rng = random.Random(random_state)
    selected_indices: list[int] = []

    for class_value in sorted(target.unique()):
        class_indices = [int(index) for index in target.index[target == class_value].tolist()]
        if class_indices and len(selected_indices) < max_rows:
            selected_indices.append(rng.choice(class_indices))

    remaining = [int(index) for index in export_frame.index.tolist() if int(index) not in set(selected_indices)]
    rng.shuffle(remaining)
    selected_indices.extend(remaining[: max_rows - len(selected_indices)])
    selected_indices = sorted(dict.fromkeys(selected_indices))
    return export_frame.loc[selected_indices].reset_index(drop=True)


def inject_missing_values(export_frame: pd.DataFrame, target_column: str, random_state: int) -> pd.DataFrame:
    feature_columns = [column for column in export_frame.columns if column != target_column]
    if export_frame.empty or not feature_columns:
        return export_frame

    candidates = [
        (row_index, column_name)
        for row_index in range(len(export_frame))
        for column_name in feature_columns
        if not pd.isna(export_frame.at[row_index, column_name])
    ]
    if not candidates:
        return export_frame

    rng = random.Random(random_state)
    injection_count = min(3, max(1, len(feature_columns) // 2), len(candidates))
    for row_index, column_name in rng.sample(candidates, injection_count):
        export_frame.at[row_index, column_name] = np.nan
    return export_frame


def build_training_inputs(csv_path: str | Path, target_column: str, config: Any) -> dict[str, Any]:
    data_path = Path(csv_path)
    if not data_path.exists():
        raise ExampleBuildError(f"CSV dataset was not found: {data_path}")

    overrides = merged_overrides(
        load_overrides(getattr(config, "schema_overrides", None)),
        parse_drop_columns(getattr(config, "drop_columns", None)),
        target_column,
    )
    resolved_target = resolve_target_column(target_column, overrides)
    if resolved_target is None:
        raise ExampleBuildError("A target column is required.")

    dataframe = read_csv(data_path)
    target_payload = validate_target(dataframe, resolved_target)
    schema = build_schema(dataframe, resolved_target, overrides)
    schema["target_column"] = resolved_target
    feature_definitions = iter_feature_definitions(dataframe, resolved_target, overrides)
    ensure_unique_feature_names(feature_definitions)

    if not feature_definitions:
        raise ExampleBuildError("No feature columns remain after applying target and dropped columns.")

    feature_frame = prepare_feature_frame(dataframe, feature_definitions)
    return {
        "schema": schema,
        "feature_definitions": feature_definitions,
        "feature_frame": feature_frame,
        "target": target_payload["series"],
        "target_column": resolved_target,
        "target_mapping": target_payload["mapping"],
    }


def write_common_artifacts(
    output_dir: str | Path,
    schema: dict[str, Any],
    export_frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    export_frame.to_csv(output_path / "dataset.csv", index=False)
    with (output_path / "feature_schema.json").open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)
        handle.write("\n")
    with (output_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def build_export_from_config(
    features: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    config: Any,
) -> pd.DataFrame:
    export_frame = build_export_frame(
        features,
        target,
        target_column,
        getattr(config, "max_rows", 50),
        getattr(config, "random_state", 42),
    )
    if getattr(config, "inject_missing", False):
        export_frame = inject_missing_values(export_frame, target_column, getattr(config, "random_state", 42))
    return export_frame


def print_summary(
    schema: dict[str, Any],
    export_frame: pd.DataFrame,
    target_column: str,
    target_mapping: dict[str, int],
    output_dir: Path,
    model_filename: str,
) -> None:
    features = schema["features"]
    feature_types: dict[str, int] = {}
    for feature in features:
        feature_type = str(feature["type"])
        feature_types[feature_type] = feature_types.get(feature_type, 0) + 1

    class_distribution = export_frame[target_column].value_counts().sort_index().to_dict()

    print(f"Generated example with {len(features)} features")
    print(f"Feature types: {feature_types}")
    print(f"Target mapping: {target_mapping}")
    print(f"Exported dataset class distribution: {class_distribution}")
    print(f"{model_filename}: {output_dir / model_filename}")
    print(f"dataset.csv: {output_dir / 'dataset.csv'}")
    print(f"feature_schema.json: {output_dir / 'feature_schema.json'}")
    print(f"metadata.json: {output_dir / 'metadata.json'}")


__all__ = [
    "ExampleBuildError",
    "SchemaGenerationError",
    "build_export_from_config",
    "build_training_inputs",
    "encode_feature_frame_for_numeric_training",
    "prepare_numeric_features_for_training",
    "print_summary",
    "split_train_validation_indices",
    "validate_output_dir",
    "write_common_artifacts",
]
