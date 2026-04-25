from __future__ import annotations

from pathlib import Path

from app.adapters.base import ModelAdapter
from app.adapters.lightgbm_adapter import LightGBMModelAdapter
from app.adapters.xgboost_adapter import XGBoostModelAdapter


class ModelAdapterResolutionError(ValueError):
    pass


_REGISTERED_ADAPTERS: tuple[type[ModelAdapter], ...] = (
    LightGBMModelAdapter,
    XGBoostModelAdapter,
)

_ADAPTERS_BY_FAMILY: dict[str, type[ModelAdapter]] = {
    adapter.model_family: adapter for adapter in _REGISTERED_ADAPTERS
}

_ADAPTERS_BY_EXTENSION: dict[str, type[ModelAdapter]] = {
    extension.lower(): adapter
    for adapter in _REGISTERED_ADAPTERS
    for extension in adapter.supported_extensions
}


def supported_model_families() -> tuple[str, ...]:
    return tuple(_ADAPTERS_BY_FAMILY.keys())


def supported_model_extensions() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS_BY_EXTENSION.keys()))


def resolve_model_adapter(
    *,
    model_path: str | Path | None = None,
    model_family: str | None = None,
) -> type[ModelAdapter]:
    if model_family:
        adapter = _ADAPTERS_BY_FAMILY.get(model_family.lower())
        if adapter is not None:
            return adapter
        raise ModelAdapterResolutionError(
            f"Unsupported model family '{model_family}'. Supported families: {', '.join(supported_model_families())}."
        )

    if model_path is not None:
        extension = Path(model_path).suffix.lower()
        adapter = _ADAPTERS_BY_EXTENSION.get(extension)
        if adapter is not None:
            return adapter
        raise ModelAdapterResolutionError(
            f"Unsupported model file extension '{extension or '<none>'}'. "
            f"Supported extensions: {', '.join(supported_model_extensions())}."
        )

    raise ModelAdapterResolutionError("Model adapter resolution requires either model_family or model_path.")
