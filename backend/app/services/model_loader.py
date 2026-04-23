from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import UploadFile

from app.adapters.registry import resolve_model_adapter


async def load_ensemble_model(upload_file: UploadFile, model_family: str | None = None):
    suffix = os.path.splitext(upload_file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        temp_file.write(await upload_file.read())

    try:
        adapter = resolve_model_adapter(model_path=temp_path if not model_family else None, model_family=model_family)
        return adapter.load_artifacts_from_path(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def load_ensemble_model_from_path(model_path: str | Path, model_family: str | None = None):
    adapter = resolve_model_adapter(model_path=model_path if not model_family else None, model_family=model_family)
    return adapter.load_artifacts_from_path(model_path)


# Compatibility aliases keep existing imports working during the transition to
# generic ensemble naming.
load_normalized_model = load_ensemble_model
load_normalized_model_from_path = load_ensemble_model_from_path
