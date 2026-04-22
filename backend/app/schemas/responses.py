from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.schemas.common import FeatureMetadata, FeatureValue, LayoutEdge, PredictionSummary, PreviewPayload


class ModelSummary(BaseModel):
    model_type: str
    num_trees: int
    num_features: int
    feature_names: List[str]


class LayoutSummary(BaseModel):
    max_tree_depth: int
    total_leaves: int


class ModelUploadResponse(BaseModel):
    session_id: str
    model_summary: ModelSummary
    feature_metadata: List[FeatureMetadata]
    layout_summary: LayoutSummary


class TreeLayoutNode(BaseModel):
    node_id: int
    depth: int
    radius: float
    angle: float
    x: float
    y: float
    is_leaf: bool
    split_feature: str
    threshold: float | str
    left_child_id: int
    right_child_id: int
    subtree_leaf_count: int


class TreeLayoutLeaf(BaseModel):
    leaf_id: int
    depth: int
    radius: float
    angle: float
    x: float
    y: float
    is_leaf: bool
    leaf_value: float
    leaf_order: int


class TreeLayoutPayload(BaseModel):
    tree_index: int
    sector_start_angle: float
    sector_end_angle: float
    max_depth: int
    nodes: List[TreeLayoutNode]
    edges: List[LayoutEdge]
    leaves: List[TreeLayoutLeaf]


class LayoutEnvelope(BaseModel):
    trees: List[TreeLayoutPayload]


class LayoutResponse(BaseModel):
    layout: LayoutEnvelope


class TreePredictionResult(BaseModel):
    tree_index: int
    selected_leaf_id: int
    leaf_value: float
    contribution: float
    active_path_node_ids: List[int]


class PredictResponse(BaseModel):
    prediction: PredictionSummary
    tree_results: List[TreePredictionResult]


class SamplePayload(BaseModel):
    source: str
    row_index: int
    feature_vector: Dict[str, FeatureValue]


class SelectRowResponse(BaseModel):
    sample: SamplePayload
    prediction: PredictionSummary
    tree_results: List[TreePredictionResult]


class DatasetSummary(BaseModel):
    is_loaded: bool = True
    num_rows: int
    num_columns: int
    matched_feature_count: int
    unmatched_model_features: List[str]
    extra_dataset_columns: List[str]


class DatasetUploadResponse(BaseModel):
    dataset_summary: DatasetSummary
    feature_metadata: List[FeatureMetadata]
    preview: PreviewPayload


class ExamplesListResponse(BaseModel):
    examples: List[str]


class LoadExampleResponse(BaseModel):
    session_id: str
    example_name: str
    model_summary: ModelSummary
    feature_metadata: List[FeatureMetadata]
    layout_summary: LayoutSummary
    dataset_summary: DatasetSummary
    preview: PreviewPayload


class MetadataDatasetSummary(BaseModel):
    is_loaded: bool
    num_rows: int
    num_columns: int
    matched_feature_count: int = 0
    unmatched_model_features: List[str] = []
    extra_dataset_columns: List[str] = []


class SessionMetadataResponse(BaseModel):
    model_summary: ModelSummary
    feature_metadata: List[FeatureMetadata]
    dataset_summary: MetadataDatasetSummary


class SchemaUploadResponse(BaseModel):
    dataset_summary: MetadataDatasetSummary
    feature_metadata: List[FeatureMetadata]
    preview: Optional[PreviewPayload] = None
