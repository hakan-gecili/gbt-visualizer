from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.session_store import session_store
from app.schemas.responses import LayoutResponse, SessionMetadataResponse
from app.services.layout_service import serialize_layout

router = APIRouter(prefix="/api/session", tags=["session"])


@router.get("/{session_id}/layout", response_model=LayoutResponse)
async def get_layout(session_id: str) -> LayoutResponse:
    try:
        session = session_store.get(session_id)
        return LayoutResponse(**serialize_layout(session.model))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{session_id}/metadata", response_model=SessionMetadataResponse)
async def get_metadata(session_id: str) -> SessionMetadataResponse:
    try:
        session = session_store.get(session_id)
        return SessionMetadataResponse(
            model_summary={
                "model_type": "lightgbm_binary_classifier",
                "num_trees": session.model.num_trees,
                "num_features": len(session.model.feature_names),
                "feature_names": session.model.feature_names,
            },
            feature_metadata=session.feature_metadata,
            dataset_summary=session.dataset_summary,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

