from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.session_store import session_store
from app.schemas.requests import PredictRequest
from app.schemas.responses import PredictResponse
from app.services.prediction_service import predict_model

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        session = session_store.get(request.session_id)
        _, prediction_result = predict_model(session.model, request.feature_vector)
        return PredictResponse(
            prediction={
                "margin": prediction_result.margin,
                "probability": prediction_result.probability,
                "predicted_label": 1 if prediction_result.probability >= 0.5 else 0,
            },
            tree_results=[
                {
                    "tree_index": item.tree_index,
                    "selected_leaf_id": item.selected_leaf_id,
                    "leaf_value": item.leaf_value,
                    "contribution": item.leaf_value,
                    "active_path_node_ids": item.path_node_ids,
                }
                for item in prediction_result.tree_results
            ],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

