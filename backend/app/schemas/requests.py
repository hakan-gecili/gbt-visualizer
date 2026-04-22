from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field

from app.schemas.common import FeatureValue


class PredictRequest(BaseModel):
    session_id: str
    feature_vector: Dict[str, FeatureValue] = Field(default_factory=dict)


class SelectRowRequest(BaseModel):
    session_id: str
    row_index: int
