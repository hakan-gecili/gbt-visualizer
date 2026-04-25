#!/usr/bin/env python3
"""Train a binary classifier from CSV and export app example artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.common import ExampleBuildError, SchemaGenerationError, print_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build model, dataset.csv, feature_schema.json, and metadata.json from a CSV file."
    )
    parser.add_argument("--csv", "--data", dest="csv_path", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--target",
        "--target-column",
        dest="target_column",
        required=True,
        help="Binary target column name.",
    )
    parser.add_argument("--output-dir", required=True, help="Destination directory for the generated example.")
    parser.add_argument(
        "--model-family",
        choices=("lightgbm", "xgboost"),
        default="lightgbm",
        help="Model family to train.",
    )
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of boosting rounds.")
    parser.add_argument("--max-depth", type=int, default=-1, help="Maximum tree depth. Use -1 for no limit.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Boosting learning rate.")
    parser.add_argument("--num-leaves", type=int, default=31, help="Maximum LightGBM leaves per tree.")
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


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    try:
        if args.model_family == "lightgbm":
            from training.lightgbm_trainer import train_lightgbm

            result = train_lightgbm(args.csv_path, args.target_column, output_dir, args)
        elif args.model_family == "xgboost":
            from training.xgboost_trainer import train_xgboost

            result = train_xgboost(args.csv_path, args.target_column, output_dir, args)
        else:
            raise ExampleBuildError(f"Unsupported model family: {args.model_family}")

        print_summary(
            result["schema"],
            result["export_frame"],
            result["target_column"],
            result["target_mapping"],
            output_dir,
            result["model_filename"],
        )
        return 0
    except (SchemaGenerationError, ExampleBuildError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
