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
    threshold: float = Field(ge=0.0, le=1.0)
    target_class: Optional[int] = Field(default=None, ge=0, le=1)
    feature_vector: Optional[Dict[str, FeatureValue]] = None
    max_steps: Optional[int] = Field(default=None, ge=1)
