#!/usr/bin/env python3
"""Train a LightGBM binary classifier from CSV and export app example artifacts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import lightgbm as lgb
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build model.txt, dataset.csv, and feature_schema.json from a CSV file."
    )
    parser.add_argument("--data", required=True, help="Path to the input CSV file.")
    parser.add_argument("--target-column", required=True, help="Binary target column name.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for the generated example.")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of boosting rounds.")
    parser.add_argument("--max-depth", type=int, default=-1, help="Maximum tree depth. Use -1 for no limit.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Boosting learning rate.")
    parser.add_argument("--num-leaves", type=int, default=31, help="Maximum leaves per tree.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction reserved for validation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-rows", type=int, default=50, help="Maximum rows to export in dataset.csv.")
    parser.add_argument(
        "--schema-overrides",
        help="Optional JSON file with the same override format used by generate_feature_schema.py.",
    )
    parser.add_argument(
        "--drop-columns",
        help="Optional comma-separated column names to exclude from the feature set.",
    )
    parser.add_argument(
        "--inject-missing",
        action="store_true",
        help="Inject a few missing values into the exported dataset.csv for app testing.",
    )
    return parser.parse_args()


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


def validate_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return

    allowed_names = {"model.txt", "dataset.csv", "feature_schema.json"}
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


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    feature_definitions: list[tuple[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> lgb.Booster:
    train_indices, validation_indices = split_train_validation_indices(target, args.test_size, args.random_state)
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

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "seed": args.random_state,
        "feature_pre_filter": False,
        "verbosity": -1,
    }

    return lgb.train(
        params=params,
        train_set=train_dataset,
        num_boost_round=args.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.log_evaluation(period=0)],
    )


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


def print_summary(
    schema: dict[str, Any],
    export_frame: pd.DataFrame,
    target_column: str,
    target_mapping: dict[str, int],
    output_dir: Path,
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
    print(f"model.txt: {output_dir / 'model.txt'}")
    print(f"dataset.csv: {output_dir / 'dataset.csv'}")
    print(f"feature_schema.json: {output_dir / 'feature_schema.json'}")


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)

    try:
        if not data_path.exists():
            raise ExampleBuildError(f"CSV dataset was not found: {data_path}")

        overrides = merged_overrides(
            load_overrides(args.schema_overrides),
            parse_drop_columns(args.drop_columns),
            args.target_column,
        )
        target_column = resolve_target_column(args.target_column, overrides)
        if target_column is None:
            raise ExampleBuildError("A target column is required.")

        validate_output_dir(output_dir)
        dataframe = read_csv(data_path)
        target_payload = validate_target(dataframe, target_column)

        schema = build_schema(dataframe, target_column, overrides)
        schema["target_column"] = target_column
        feature_definitions = iter_feature_definitions(dataframe, target_column, overrides)
        ensure_unique_feature_names(feature_definitions)

        if not feature_definitions:
            raise ExampleBuildError("No feature columns remain after applying target and dropped columns.")

        feature_frame = prepare_feature_frame(dataframe, feature_definitions)
        booster = train_model(feature_frame, target_payload["series"], feature_definitions, args)

        export_frame = build_export_frame(
            feature_frame,
            target_payload["series"],
            target_column,
            args.max_rows,
            args.random_state,
        )
        if args.inject_missing:
            export_frame = inject_missing_values(export_frame, target_column, args.random_state)

        output_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(output_dir / "model.txt"))
        export_frame.to_csv(output_dir / "dataset.csv", index=False)
        with (output_dir / "feature_schema.json").open("w", encoding="utf-8") as handle:
            json.dump(schema, handle, indent=2)
            handle.write("\n")

        print_summary(schema, export_frame, target_column, target_payload["mapping"], output_dir)
        return 0
    except (SchemaGenerationError, ExampleBuildError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
