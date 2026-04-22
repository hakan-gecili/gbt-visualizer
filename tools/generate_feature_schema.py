#!/usr/bin/env python3
"""Generate a backend-compatible feature_schema.json from a CSV dataset.

Usage:
    python tools/generate_feature_schema.py \
        --data train.csv \
        --output feature_schema.json \
        --target-column target

Optional overrides:
    python tools/generate_feature_schema.py \
        --data train.csv \
        --output feature_schema.json \
        --overrides schema_overrides.json

Override file format:
{
  "target_column": "target",
  "exclude_columns": ["id"],
  "features": {
    "sex": {
      "type": "binary",
      "labels": {
        "0": "Female",
        "1": "Male"
      }
    },
    "mean radius": {
      "name": "mean_radius"
    },
    "education_code": {
      "type": "categorical"
    },
    "income": {
      "default_value": "medium"
    },
    "user_id": {
      "exclude": true
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from schema_utils import SchemaGenerationError, build_schema, load_overrides, resolve_target_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a CSV dataset and generate a backend-compatible feature_schema.json "
            "for the LightGBM visualizer."
        )
    )
    parser.add_argument("--data", required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--output", required=True, help="Path to the generated feature_schema.json.")
    parser.add_argument(
        "--target-column",
        help="Optional target column to exclude from generated feature definitions.",
    )
    parser.add_argument(
        "--overrides",
        help="Optional JSON file with small human-editable overrides.",
    )
    return parser.parse_args()


def print_summary(schema: dict[str, object], output_path: Path) -> None:
    features = schema["features"]
    print(f"Generated schema for {len(features)} features")
    for feature in features:
        print(f"  - {feature['name']}: {feature['type']}")
    print(f"Wrote schema to {output_path}")


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    data_path = Path(args.data)

    try:
        if not data_path.exists():
            raise SchemaGenerationError(f"CSV dataset was not found: {data_path}")

        overrides = load_overrides(args.overrides)
        target_column = resolve_target_column(args.target_column, overrides)
        dataframe = pd.read_csv(data_path)

        if dataframe.empty:
            raise SchemaGenerationError(f"CSV dataset has no rows: {data_path}")
        if target_column is not None and target_column not in dataframe.columns:
            raise SchemaGenerationError(
                f"Target column '{target_column}' was not found in dataset columns: {list(map(str, dataframe.columns))}"
            )

        schema = build_schema(dataframe, target_column, overrides)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(schema, handle, indent=2)
            handle.write("\n")

        print_summary(schema, output_path)
        return 0
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"Error: failed to read CSV '{data_path}': {exc}")
        return 1
    except SchemaGenerationError as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
