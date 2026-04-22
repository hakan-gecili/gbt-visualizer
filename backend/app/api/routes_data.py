from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.session_store import session_store
from app.schemas.requests import SelectRowRequest
from app.schemas.responses import DatasetUploadResponse, SchemaUploadResponse, SelectRowResponse
from app.services.dataset_service import (
    apply_dataset_ranges,
    build_preview,
    extract_feature_vector_from_row,
    load_dataset,
    summarize_dataset,
)
from app.services.feature_schema_service import FeatureSchemaError, build_feature_metadata, parse_feature_schema_json
from app.services.prediction_service import predict_model

router = APIRouter(prefix="/api", tags=["data"])
logger = logging.getLogger(__name__)


def _prediction_payload(prediction_result):
    return {
        "margin": prediction_result.margin,
        "probability": prediction_result.probability,
        "predicted_label": 1 if prediction_result.probability >= 0.5 else 0,
    }


def _tree_payloads(prediction_result):
    return [
        {
            "tree_index": item.tree_index,
            "selected_leaf_id": item.selected_leaf_id,
            "leaf_value": item.leaf_value,
            "contribution": item.leaf_value,
            "active_path_node_ids": item.path_node_ids,
        }
        for item in prediction_result.tree_results
    ]


@router.post("/data/upload", response_model=DatasetUploadResponse)
async def upload_data(
    session_id: str = Form(...),
    data_file: UploadFile = File(...),
) -> DatasetUploadResponse:
    try:
        logger.info(
            "Received dataset upload: session_id=%s filename=%s content_type=%s",
            session_id,
            data_file.filename,
            data_file.content_type,
        )
        session = session_store.get(session_id)
        dataframe = await load_dataset(data_file)
        dataset_summary = summarize_dataset(dataframe, session.model.feature_names)
        feature_metadata = apply_dataset_ranges(session.feature_metadata, dataframe)
        preview = build_preview(dataframe)
        session.dataset_frame = dataframe
        session.dataset_summary = dataset_summary
        session.feature_metadata = feature_metadata
        session.preview = preview
        session_store.save(session)
        return DatasetUploadResponse(
            dataset_summary=dataset_summary,
            feature_metadata=feature_metadata,
            preview=preview,
        )
    except KeyError as exc:
        logger.exception("Dataset upload failed: missing session_id=%s", session_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Dataset upload failed: session_id=%s filename=%s", session_id, data_file.filename)
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {exc}") from exc


@router.post("/schema/upload", response_model=SchemaUploadResponse)
async def upload_schema(
    session_id: str = Form(...),
    schema_file: UploadFile = File(...),
) -> SchemaUploadResponse:
    try:
        logger.info(
            "Received schema upload: session_id=%s filename=%s content_type=%s",
            session_id,
            schema_file.filename,
            schema_file.content_type,
        )
        session = session_store.get(session_id)
        raw_schema = await schema_file.read()
        schema_overrides = parse_feature_schema_json(raw_schema.decode("utf-8"))
        feature_metadata = build_feature_metadata(session.model.feature_names, schema_overrides)
        if session.dataset_frame is not None:
            feature_metadata = apply_dataset_ranges(feature_metadata, session.dataset_frame)
        session.feature_metadata = feature_metadata
        session_store.save(session)
        return SchemaUploadResponse(
            dataset_summary=session.dataset_summary,
            feature_metadata=feature_metadata,
            preview=session.preview,
        )
    except KeyError as exc:
        logger.exception("Schema upload failed: missing session_id=%s", session_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnicodeDecodeError as exc:
        logger.exception("Schema upload failed decoding: session_id=%s filename=%s", session_id, schema_file.filename)
        raise HTTPException(status_code=400, detail="Feature schema file must be UTF-8 encoded JSON.") from exc
    except FeatureSchemaError as exc:
        logger.exception("Schema upload failed validation: session_id=%s filename=%s", session_id, schema_file.filename)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Schema upload failed: session_id=%s filename=%s", session_id, schema_file.filename)
        raise HTTPException(status_code=400, detail=f"Failed to load feature schema: {exc}") from exc


@router.post("/sample/select-row", response_model=SelectRowResponse)
async def select_row(request: SelectRowRequest) -> SelectRowResponse:
    try:
        session = session_store.get(request.session_id)
        if session.dataset_frame is None:
            raise HTTPException(status_code=400, detail="No dataset has been uploaded for this session.")
        feature_vector = extract_feature_vector_from_row(
            session.dataset_frame,
            request.row_index,
            session.feature_metadata,
        )
        prepared_feature_vector, prediction_result = predict_model(
            session.model,
            session.feature_metadata,
            feature_vector,
        )
        return SelectRowResponse(
            sample={
                "source": "dataset_row",
                "row_index": request.row_index,
                "feature_vector": prepared_feature_vector,
            },
            prediction=_prediction_payload(prediction_result),
            tree_results=_tree_payloads(prediction_result),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
