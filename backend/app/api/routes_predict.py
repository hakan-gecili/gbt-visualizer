from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.session_store import session_store
from app.schemas.requests import PredictRequest
from app.schemas.responses import PredictResponse
from app.services.prediction_service import predict_model
from app.services.serialization_service import serialize_prediction, serialize_tree_predictions

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        session = session_store.get(request.session_id)
        _, prediction_result = predict_model(
            session.model,
            session.predictor,
            session.feature_metadata,
            request.feature_vector,
        )
        return PredictResponse(
            prediction=serialize_prediction(prediction_result),
            tree_results=serialize_tree_predictions(prediction_result),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
