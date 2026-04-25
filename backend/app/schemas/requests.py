from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

from app.schemas.common import FeatureValue


class PredictRequest(BaseModel):
    session_id: str
    feature_vector: Dict[str, FeatureValue] = Field(default_factory=dict)


class SelectRowRequest(BaseModel):
    session_id: str
    row_index: int


class CounterfactualRequest(BaseModel):
    session_id: str
    row_index: int = Field(ge=0)
    threshold: float = Field(gt=0.0, lt=1.0)
    target_class: Optional[int] = Field(default=None, ge=0, le=1)
    max_steps: int = Field(default=3, ge=1)
