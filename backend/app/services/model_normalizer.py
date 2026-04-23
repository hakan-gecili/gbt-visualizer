from __future__ import annotations

from app.adapters.lightgbm_adapter import LightGBMAdapterError, LightGBMModelAdapter
from app.domain.model_types import EnsembleModel, summarize_layout


# Compatibility aliases keep the existing service import surface stable while
# LightGBM-specific normalization moves behind the dedicated adapter module.
LightGBMModelNormalizationError = LightGBMAdapterError


def normalize_dumped_model(model_dump: dict) -> EnsembleModel:
    return LightGBMModelAdapter.from_dumped_model(model_dump)


__all__ = [
    "LightGBMModelNormalizationError",
    "normalize_dumped_model",
    "summarize_layout",
]
