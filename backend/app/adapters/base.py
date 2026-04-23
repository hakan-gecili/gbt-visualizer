from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.domain.model_types import EnsembleModel


@runtime_checkable
class ModelAdapter(Protocol):
    """Common loading surface for model-family adapters.

    A future XGBoost adapter should implement the same class attributes and
    `load_from_path()` method, then register itself in `registry.py`.
    """

    model_family: str
    model_type: str
    supported_extensions: tuple[str, ...]

    @classmethod
    def load_from_path(cls, model_path: str | Path) -> EnsembleModel:
        ...

    @classmethod
    def load_artifacts_from_path(cls, model_path: str | Path) -> tuple[EnsembleModel, Any]:
        ...
