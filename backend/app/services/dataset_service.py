from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import UploadFile


def _python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


async def load_dataset(upload_file: UploadFile) -> pd.DataFrame:
    return pd.read_csv(upload_file.file)


def _canonicalize_feature_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _dataset_column_map(dataframe: pd.DataFrame) -> dict[str, str]:
    column_map: dict[str, str] = {}
    for column in dataframe.columns:
        canonical = _canonicalize_feature_name(str(column))
        column_map.setdefault(canonical, str(column))
    return column_map


def summarize_dataset(dataframe: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
    columns = [str(column) for column in dataframe.columns]
    column_map = _dataset_column_map(dataframe)
    matched_feature_count = sum(1 for feature in feature_names if _canonicalize_feature_name(feature) in column_map)
    unmatched_model_features = [
        feature for feature in feature_names if _canonicalize_feature_name(feature) not in column_map
    ]
    feature_keys = {_canonicalize_feature_name(feature) for feature in feature_names}
    extra_dataset_columns = [column for column in columns if _canonicalize_feature_name(column) not in feature_keys]
    return {
        "is_loaded": True,
        "num_rows": int(len(dataframe)),
        "num_columns": int(len(columns)),
        "matched_feature_count": matched_feature_count,
        "unmatched_model_features": unmatched_model_features,
        "extra_dataset_columns": extra_dataset_columns,
    }


def build_preview(dataframe: pd.DataFrame, limit: int = 50) -> dict[str, Any]:
    preview_frame = dataframe.head(limit)
    return {
        "columns": [str(column) for column in preview_frame.columns],
        "rows": [[_python_value(value) for value in row] for row in preview_frame.to_numpy().tolist()],
    }


def apply_dataset_ranges(feature_metadata: list[dict[str, Any]], dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    updated = []
    column_map = _dataset_column_map(dataframe)
    for item in feature_metadata:
        feature_name = item["name"]
        next_item = dict(item)
        dataset_column = column_map.get(_canonicalize_feature_name(feature_name))
        if dataset_column is not None:
            series = pd.to_numeric(dataframe[dataset_column], errors="coerce")
            numeric = series.dropna()
            if not numeric.empty:
                next_item["min_value"] = float(numeric.min())
                next_item["max_value"] = float(numeric.max())
                next_item["default_value"] = float(numeric.mean())
        updated.append(next_item)
    return updated


def extract_feature_vector_from_row(dataframe: pd.DataFrame, row_index: int, feature_names: list[str]) -> dict[str, float]:
    if row_index < 0 or row_index >= len(dataframe):
        raise IndexError(f"Row index {row_index} is outside dataset bounds.")
    row = dataframe.iloc[row_index]
    column_map = _dataset_column_map(dataframe)
    feature_vector: dict[str, float] = {}
    for feature_name in feature_names:
        dataset_column = column_map.get(_canonicalize_feature_name(feature_name))
        value = row[dataset_column] if dataset_column is not None else 0.0
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        feature_vector[feature_name] = 0.0 if pd.isna(numeric) else float(numeric)
    return feature_vector
