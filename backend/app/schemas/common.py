from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FeatureMetadata(BaseModel):
    name: str
    short_name: str
    type: str = "numeric"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: float = 0.0


class PredictionSummary(BaseModel):
    margin: float
    probability: float
    predicted_label: int


class ApiErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ApiErrorResponse(BaseModel):
    error: ApiErrorDetail


class LayoutEdge(BaseModel):
    source_id: int
    target_id: int


class PreviewPayload(BaseModel):
    columns: List[str]
    rows: List[List[Any]]

