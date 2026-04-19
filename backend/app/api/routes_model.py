from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.session_store import session_store
from app.domain.session_types import SessionState
from app.schemas.responses import ModelUploadResponse
from app.services.model_loader import load_normalized_model
from app.services.model_normalizer import (
    LightGBMModelNormalizationError,
    build_feature_metadata,
    summarize_layout,
)

router = APIRouter(prefix="/api/model", tags=["model"])
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=ModelUploadResponse)
async def upload_model(model_file: UploadFile = File(...)) -> ModelUploadResponse:
    try:
        logger.info("Received model upload: filename=%s content_type=%s", model_file.filename, model_file.content_type)
        model = await load_normalized_model(model_file)
    except LightGBMModelNormalizationError as exc:
        logger.exception("Model upload failed normalization: filename=%s", model_file.filename)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Model upload failed parsing: filename=%s", model_file.filename)
        raise HTTPException(status_code=400, detail=f"Failed to parse model: {exc}") from exc

    feature_metadata = build_feature_metadata(model.feature_names)
    session = SessionState(
        session_id=model.model_id,
        model=model,
        feature_metadata=feature_metadata,
    )
    session_store.save(session)
    max_tree_depth, total_leaves = summarize_layout(model)
    return ModelUploadResponse(
        session_id=session.session_id,
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
    )
