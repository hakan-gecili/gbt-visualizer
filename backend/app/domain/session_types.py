from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from app.domain.model_types import NormalizedModel


@dataclass
class SessionState:
    session_id: str
    model: NormalizedModel
    feature_metadata: List[Dict[str, Any]]
    dataset_frame: Optional[pd.DataFrame] = None
    dataset_summary: Dict[str, Any] = field(
        default_factory=lambda: {
            "is_loaded": False,
            "num_rows": 0,
            "num_columns": 0,
            "matched_feature_count": 0,
            "unmatched_model_features": [],
            "extra_dataset_columns": [],
        }
    )
    preview: Optional[Dict[str, Any]] = None

