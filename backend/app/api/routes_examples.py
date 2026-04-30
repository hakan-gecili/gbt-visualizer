from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.adapters.registry import ModelAdapterResolutionError, supported_model_extensions
from app.core.session_store import session_store
from app.domain.session_types import SessionState
from app.schemas.responses import ExamplesListResponse, LoadExampleResponse
from app.services.dataset_service import (
    apply_dataset_ranges,
    build_preview,
    load_dataset_from_path,
    summarize_dataset,
)
from app.services.feature_schema_service import FeatureSchemaError, build_feature_metadata, parse_feature_schema_json
from app.services.counterfactual_service import build_lightgbm_counterfactual_metadata
from app.services.model_loader import load_ensemble_model_from_path
from app.services.model_normalizer import LightGBMModelNormalizationError, summarize_layout
from app.services.serialization_service import serialize_model_summary

router = APIRouter(prefix="/api/examples", tags=["examples"])
logger = logging.getLogger(__name__)

EXAMPLES_ROOT = Path(__file__).resolve().parents[3] / "examples"
MODEL_FILE_FAMILIES = {
    "model.txt": "lightgbm",
    "model.json": "xgboost",
}


@dataclass(frozen=True)
class ExampleVariantInfo:
    id: str
    dataset_name: str
    model_family: str
    path: Path
    model_path: Path
    dataset_path: Path | None
    schema_path: Path | None
    metadata: dict[str, Any]

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(EXAMPLES_ROOT.parent).as_posix()

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_family": self.model_family,
            "path": self.relative_path,
            "model_file": self.model_path.name,
            "has_dataset": self.dataset_path is not None,
            "has_schema": self.schema_path is not None,
            "metadata": self.metadata,
        }


def _example_directories() -> list[Path]:
    if not EXAMPLES_ROOT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Examples directory was not found at runtime: {EXAMPLES_ROOT}",
        )
    return sorted([path for path in EXAMPLES_ROOT.iterdir() if path.is_dir()], key=lambda path: path.name)


def _load_metadata(example_dir: Path) -> dict[str, Any]:
    metadata_path = example_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to parse example metadata: %s", metadata_path)
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_dataset_path(example_dir: Path, dataset_dir: Path) -> Path | None:
    for candidate in (example_dir / "dataset.csv", dataset_dir / "dataset.csv"):
        if candidate.exists():
            return candidate
    dataset_files = sorted(example_dir.glob("*.csv")) or sorted(dataset_dir.glob("*.csv"))
    return dataset_files[0] if len(dataset_files) == 1 else None


def _find_schema_path(example_dir: Path, dataset_dir: Path) -> Path | None:
    for candidate in (example_dir / "feature_schema.json", dataset_dir / "feature_schema.json"):
        if candidate.exists():
            return candidate
    return None


def _infer_family_from_model_file(model_path: Path) -> str | None:
    if model_path.name == "model.txt" or model_path.suffix.lower() == ".txt":
        return "lightgbm"
    if model_path.name == "model.json" or model_path.suffix.lower() == ".json":
        return "xgboost"
    return None


def _variant_from_dir(dataset_dir: Path, example_dir: Path, model_path: Path, model_family: str) -> ExampleVariantInfo:
    dataset_name = dataset_dir.name
    return ExampleVariantInfo(
        id=f"{dataset_name}__{model_family}",
        dataset_name=dataset_name,
        model_family=model_family,
        path=example_dir,
        model_path=model_path,
        dataset_path=_find_dataset_path(example_dir, dataset_dir),
        schema_path=_find_schema_path(example_dir, dataset_dir),
        metadata=_load_metadata(example_dir),
    )


def _discover_dataset_variants(dataset_dir: Path) -> list[ExampleVariantInfo]:
    variants: list[ExampleVariantInfo] = []

    for family_dir in sorted([path for path in dataset_dir.iterdir() if path.is_dir()], key=lambda path: path.name):
        model_path = None
        model_family = MODEL_FILE_FAMILIES.get("model.txt") if (family_dir / "model.txt").exists() else None
        if model_family is not None:
            model_path = family_dir / "model.txt"
        elif (family_dir / "model.json").exists():
            model_family = MODEL_FILE_FAMILIES["model.json"]
            model_path = family_dir / "model.json"

        if model_path is None or model_family is None:
            continue
        variants.append(_variant_from_dir(dataset_dir, family_dir, model_path, model_family))

    direct_model_files = sorted(
        path
        for extension in supported_model_extensions()
        for path in dataset_dir.glob(f"*{extension}")
        if path.is_file()
    )
    if direct_model_files:
        if len(direct_model_files) != 1:
            logger.warning("Skipping legacy example '%s' with ambiguous model files: %s", dataset_dir.name, direct_model_files)
        else:
            model_family = _infer_family_from_model_file(direct_model_files[0])
            if model_family is not None:
                variants.append(_variant_from_dir(dataset_dir, dataset_dir, direct_model_files[0], model_family))

    return sorted(variants, key=lambda variant: variant.model_family)


def _discover_examples() -> list[ExampleVariantInfo]:
    variants: list[ExampleVariantInfo] = []
    for dataset_dir in _example_directories():
        variants.extend(_discover_dataset_variants(dataset_dir))
    return variants


def _group_examples(variants: list[ExampleVariantInfo]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ExampleVariantInfo]] = {}
    for variant in variants:
        if variant.dataset_path is None:
            logger.warning("Skipping example variant '%s' because no dataset CSV was found.", variant.id)
            continue
        grouped.setdefault(variant.dataset_name, []).append(variant)

    return [
        {
            "dataset_name": dataset_name,
            "variants": [variant.to_payload() for variant in sorted(dataset_variants, key=lambda item: item.model_family)],
        }
        for dataset_name, dataset_variants in sorted(grouped.items())
    ]


def _resolve_example_variant(example_id: str) -> ExampleVariantInfo:
    for variant in _discover_examples():
        if variant.id == example_id:
            return variant
        if "__" not in example_id and variant.path == EXAMPLES_ROOT / example_id:
            return variant
    raise HTTPException(status_code=404, detail=f"Example '{example_id}' was not found.")


def _load_example_schema(schema_path: Path | None) -> list[dict] | None:
    if schema_path is None:
        return None
    with schema_path.open("r", encoding="utf-8") as handle:
        return parse_feature_schema_json(handle.read())


def _normalize_feature_key(feature_name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", feature_name).strip("_").lower()


def _align_schema_and_dataframe_to_model_features(
    schema_overrides: list[dict] | None,
    dataframe,
    model_feature_names: list[str],
) -> tuple[list[dict] | None, Any]:
    normalized_model_names = {
        _normalize_feature_key(feature_name): feature_name
        for feature_name in model_feature_names
    }
    rename_columns: dict[str, str] = {}
    aligned_schema: list[dict] = []

    for feature in schema_overrides or []:
        feature_name = str(feature.get("name", ""))
        model_feature_name = feature_name
        if feature_name not in model_feature_names:
            model_feature_name = normalized_model_names.get(_normalize_feature_key(feature_name), feature_name)
        if model_feature_name != feature_name:
            rename_columns[feature_name] = model_feature_name
            next_feature = dict(feature)
            next_feature["name"] = model_feature_name
            aligned_schema.append(next_feature)
        else:
            aligned_schema.append(feature)

    if schema_overrides is None:
        for column_name in dataframe.columns:
            if column_name not in model_feature_names:
                model_feature_name = normalized_model_names.get(_normalize_feature_key(str(column_name)))
                if model_feature_name is not None:
                    rename_columns[str(column_name)] = model_feature_name

    if rename_columns:
        dataframe = dataframe.rename(columns=rename_columns)
    return (aligned_schema if schema_overrides is not None else None), dataframe


@router.get("", response_model=ExamplesListResponse)
async def list_examples() -> ExamplesListResponse:
    logger.info("Listing examples from %s", EXAMPLES_ROOT)
    grouped_examples = _group_examples(_discover_examples())
    logger.info("Discovered %d example dataset groups.", len(grouped_examples))
    return ExamplesListResponse(examples=grouped_examples)


@router.post("/{example_id}/load", response_model=LoadExampleResponse)
async def load_example(example_id: str) -> LoadExampleResponse:
    try:
        example = _resolve_example_variant(example_id)
        if example.dataset_path is None:
            raise HTTPException(status_code=400, detail=f"Example '{example_id}' does not include a dataset CSV.")
        logger.info(
            "Loading example '%s' with model=%s dataset=%s schema=%s",
            example_id,
            example.model_path,
            example.dataset_path,
            example.schema_path,
        )
        model, predictor = load_ensemble_model_from_path(str(example.model_path), model_family=example.model_family)
        schema_overrides = _load_example_schema(example.schema_path)
        dataframe = load_dataset_from_path(str(example.dataset_path))
        schema_overrides, dataframe = _align_schema_and_dataframe_to_model_features(
            schema_overrides,
            dataframe,
            model.feature_names,
        )
        feature_metadata = build_feature_metadata(model.feature_names, schema_overrides)
        dataset_summary = summarize_dataset(dataframe, model.feature_names)
        feature_metadata = apply_dataset_ranges(feature_metadata, dataframe)
        preview = build_preview(dataframe)

        session = SessionState(
            session_id=model.model_id,
            model=model,
            predictor=predictor,
            feature_metadata=feature_metadata,
            dataset_frame=dataframe,
            dataset_summary=dataset_summary,
            preview=preview,
            counterfactual_metadata=build_lightgbm_counterfactual_metadata(model)
            if str(model.model_family).lower() == "lightgbm"
            else {},
        )
        session_store.save(session)
        max_tree_depth, total_leaves = summarize_layout(model)

        return LoadExampleResponse(
            session_id=session.session_id,
            example_name=example.id,
            model_summary=serialize_model_summary(model),
            feature_metadata=feature_metadata,
            layout_summary={
                "max_tree_depth": max_tree_depth,
                "total_leaves": total_leaves,
            },
            dataset_summary=dataset_summary,
            preview=preview,
        )
    except HTTPException:
        raise
    except ModelAdapterResolutionError as exc:
        logger.exception("Example '%s' failed adapter resolution", example_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LightGBMModelNormalizationError as exc:
        logger.exception("Example '%s' failed model normalization", example_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FeatureSchemaError as exc:
        logger.exception("Example '%s' failed feature schema validation", example_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Example '%s' failed to load", example_id)
        raise HTTPException(status_code=400, detail=f"Failed to load example '{example_id}': {exc}") from exc
