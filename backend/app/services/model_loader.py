from __future__ import annotations

import os
import tempfile

import lightgbm as lgb
from fastapi import UploadFile

from app.services.model_normalizer import normalize_dumped_model


async def load_normalized_model(upload_file: UploadFile):
    suffix = os.path.splitext(upload_file.filename or "")[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        temp_file.write(await upload_file.read())

    try:
        booster = lgb.Booster(model_file=temp_path)
        return normalize_dumped_model(booster.dump_model())
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

