from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


FeatureValue = float | str | None


class FeatureOption(BaseModel):
    value: str
    label: str
    encoded_value: float


class FeatureMetadata(BaseModel):
    name: str
    short_name: str
    type: Literal["numeric", "binary", "categorical"] = "numeric"
    missing_allowed: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: FeatureValue = 0.0
    options: List[FeatureOption] = []


class FeatureImportanceEntry(BaseModel):
    feature_name: str
    value: float


class PredictionSummary(BaseModel):
    margin: float
    probability: float
    predicted_label: int
    decision_threshold: float = 0.5
    local_feature_importance: List[FeatureImportanceEntry] = []


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
