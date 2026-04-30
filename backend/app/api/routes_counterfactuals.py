from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.requests import CounterfactualRequest
from app.schemas.responses import CounterfactualResponse
from app.services.counterfactual_service import generate_counterfactual_for_session

router = APIRouter(prefix="/api", tags=["counterfactuals"])


@router.post("/counterfactuals", response_model=CounterfactualResponse)
async def generate_counterfactual(request: CounterfactualRequest) -> CounterfactualResponse:
    try:
        result = generate_counterfactual_for_session(
            request.session_id,
            row_index=request.row_index,
            threshold=request.threshold,
            target_class=request.target_class,
            feature_vector=request.feature_vector,
            max_steps=request.max_steps,
        )
        return CounterfactualResponse(**result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (IndexError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Counterfactual generation failed: {exc}") from exc
