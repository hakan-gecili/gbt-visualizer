from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.session_store import session_store
from app.domain.session_types import SessionState
from app.schemas.responses import ExamplesListResponse, LoadExampleResponse
from app.services.dataset_service import (
    apply_dataset_ranges,
    build_preview,
    load_dataset_from_path,
    summarize_dataset,
)
from app.services.model_loader import load_normalized_model_from_path
from app.services.model_normalizer import (
    LightGBMModelNormalizationError,
    build_feature_metadata,
    summarize_layout,
)

router = APIRouter(prefix="/api/examples", tags=["examples"])
logger = logging.getLogger(__name__)

EXAMPLES_ROOT = Path(__file__).resolve().parents[3] / "examples"


def _example_directories() -> list[Path]:
    if not EXAMPLES_ROOT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Examples directory was not found at runtime: {EXAMPLES_ROOT}",
        )
    return sorted([path for path in EXAMPLES_ROOT.iterdir() if path.is_dir()], key=lambda path: path.name)


def _resolve_example_files(example_name: str) -> tuple[Path, Path]:
    example_dir = EXAMPLES_ROOT / example_name
    if not example_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Example '{example_name}' was not found.")

    model_files = sorted(example_dir.glob("*.txt"))
    dataset_files = sorted(example_dir.glob("*.csv"))

    if len(model_files) != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Example '{example_name}' must contain exactly one .txt model file. "
                f"Found {len(model_files)}."
            ),
        )
    if len(dataset_files) != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Example '{example_name}' must contain exactly one .csv dataset file. "
                f"Found {len(dataset_files)}."
            ),
        )

    return model_files[0], dataset_files[0]


@router.get("", response_model=ExamplesListResponse)
async def list_examples() -> ExamplesListResponse:
    logger.info("Listing examples from %s", EXAMPLES_ROOT)
    examples = []
    for example_dir in _example_directories():
        try:
            _resolve_example_files(example_dir.name)
        except HTTPException:
            logger.exception("Skipping invalid example directory: %s", example_dir)
            continue
        examples.append(example_dir.name)
    logger.info("Discovered %d valid examples: %s", len(examples), examples)
    return ExamplesListResponse(examples=examples)


@router.post("/{example_name}/load", response_model=LoadExampleResponse)
async def load_example(example_name: str) -> LoadExampleResponse:
    try:
        model_path, dataset_path = _resolve_example_files(example_name)
        logger.info(
            "Loading example '%s' with model=%s dataset=%s",
            example_name,
            model_path,
            dataset_path,
        )
        model = load_normalized_model_from_path(str(model_path))
        feature_metadata = build_feature_metadata(model.feature_names)
        dataframe = load_dataset_from_path(str(dataset_path))
        dataset_summary = summarize_dataset(dataframe, model.feature_names)
        feature_metadata = apply_dataset_ranges(feature_metadata, dataframe)
        preview = build_preview(dataframe)

        session = SessionState(
            session_id=model.model_id,
            model=model,
            feature_metadata=feature_metadata,
            dataset_frame=dataframe,
            dataset_summary=dataset_summary,
            preview=preview,
        )
        session_store.save(session)
        max_tree_depth, total_leaves = summarize_layout(model)

        return LoadExampleResponse(
            session_id=session.session_id,
            example_name=example_name,
            model_summary={
                "model_type": "lightgbm_binary_classifier",
                "num_trees": model.num_trees,
                "num_features": len(model.feature_names),
                "feature_names": model.feature_names,
            },
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
    except LightGBMModelNormalizationError as exc:
        logger.exception("Example '%s' failed model normalization", example_name)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Example '%s' failed to load", example_name)
        raise HTTPException(status_code=400, detail=f"Failed to load example '{example_name}': {exc}") from exc
