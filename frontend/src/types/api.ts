export type FeatureValue = number | string | null

export type FeatureOption = {
  value: string
  label: string
  encoded_value: number
}

export type FeatureMetadata = {
  name: string
  short_name: string
  type: 'numeric' | 'binary' | 'categorical'
  missing_allowed: boolean
  min_value: number | null
  max_value: number | null
  default_value: FeatureValue
  options: FeatureOption[]
}

export type FeatureImportanceEntry = {
  feature_name: string
  value: number
}

export type ModelSummary = {
  model_family: string
  model_type: string
  num_trees: number
  num_features: number
  feature_names: string[]
  decision_threshold: number
  global_feature_importance: FeatureImportanceEntry[]
}

export type LayoutSummary = {
  max_tree_depth: number
  total_leaves: number
}

export type ModelUploadResponse = {
  session_id: string
  model_summary: ModelSummary
  feature_metadata: FeatureMetadata[]
  layout_summary: LayoutSummary
}

export type DatasetSummary = {
  is_loaded?: boolean
  num_rows: number
  num_columns: number
  matched_feature_count: number
  unmatched_model_features: string[]
  extra_dataset_columns: string[]
}

export type PreviewPayload = {
  columns: string[]
  rows: Array<Array<string | number | null>>
}

export type DatasetUploadResponse = {
  dataset_summary: DatasetSummary
  feature_metadata: FeatureMetadata[]
  preview: PreviewPayload
}

export type SchemaUploadResponse = {
  dataset_summary: DatasetSummary & { is_loaded: boolean }
  feature_metadata: FeatureMetadata[]
  preview: PreviewPayload | null
}

export type ExampleVariant = {
  id: string
  model_family: string
  path: string
  model_file: string
  has_dataset: boolean
  has_schema: boolean
  metadata: Record<string, unknown>
}

export type ExampleDatasetGroup = {
  dataset_name: string
  variants: ExampleVariant[]
}

export type ExamplesListResponse = {
  examples: ExampleDatasetGroup[]
}

export type LoadExampleResponse = {
  session_id: string
  example_name: string
  model_summary: ModelSummary
  feature_metadata: FeatureMetadata[]
  layout_summary: LayoutSummary
  dataset_summary: DatasetSummary
  preview: PreviewPayload
}

export type TreeLayoutNode = {
  node_id: number
  depth: number
  radius: number
  angle: number
  x: number
  y: number
  is_leaf: false
  split_feature: string
  threshold: number | string
  decision_type: string
  left_child_id: number
  right_child_id: number
  subtree_leaf_count: number
}

export type TreeLayoutLeaf = {
  leaf_id: number
  depth: number
  radius: number
  angle: number
  x: number
  y: number
  is_leaf: true
  leaf_value: number
  leaf_order: number
}

export type LayoutEdge = {
  source_id: number
  target_id: number
}

export type TreeLayout = {
  tree_index: number
  sector_start_angle: number
  sector_end_angle: number
  max_depth: number
  nodes: TreeLayoutNode[]
  edges: LayoutEdge[]
  leaves: TreeLayoutLeaf[]
}

export type LayoutResponse = {
  layout: {
    trees: TreeLayout[]
  }
}

export type PredictionSummary = {
  margin: number
  probability: number
  predicted_label: number
  decision_threshold: number
  local_feature_importance: FeatureImportanceEntry[]
}

export type TreePredictionResult = {
  tree_index: number
  selected_leaf_id: number
  leaf_value: number
  contribution: number
  active_path_node_ids: number[]
}

export type PredictResponse = {
  prediction: PredictionSummary
  tree_results: TreePredictionResult[]
}

export type SelectRowResponse = {
  sample: {
    source: string
    row_index: number
    feature_vector: Record<string, FeatureValue>
  }
  prediction: PredictionSummary
  tree_results: TreePredictionResult[]
}

export type SessionMetadataResponse = {
  model_summary: ModelSummary
  feature_metadata: FeatureMetadata[]
  dataset_summary: DatasetSummary & { is_loaded: boolean }
}
