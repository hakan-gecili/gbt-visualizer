import { apiFetch, createFormData } from './apiClient'
import type {
  CounterfactualResponse,
  DatasetUploadResponse,
  ExamplesListResponse,
  LayoutResponse,
  LoadExampleResponse,
  ModelUploadResponse,
  PredictResponse,
  SchemaUploadResponse,
  SelectRowResponse,
  SessionMetadataResponse,
  FeatureValue,
} from '../types/api'

export function uploadModel(modelFile: File, modelFamily?: string) {
  const formData = createFormData({ model_file: modelFile })
  if (modelFamily) {
    formData.append('model_family', modelFamily)
  }
  return apiFetch<ModelUploadResponse>('/api/model/upload', {
    method: 'POST',
    body: formData,
  })
}

export function fetchLayout(sessionId: string) {
  return apiFetch<LayoutResponse>(`/api/session/${sessionId}/layout`)
}

export function fetchMetadata(sessionId: string) {
  return apiFetch<SessionMetadataResponse>(`/api/session/${sessionId}/metadata`)
}

export function uploadDataset(sessionId: string, dataFile: File) {
  return apiFetch<DatasetUploadResponse>('/api/data/upload', {
    method: 'POST',
    body: createFormData({
      session_id: sessionId,
      data_file: dataFile,
    }),
  })
}

export function uploadFeatureSchema(sessionId: string, schemaFile: File) {
  return apiFetch<SchemaUploadResponse>('/api/schema/upload', {
    method: 'POST',
    body: createFormData({
      session_id: sessionId,
      schema_file: schemaFile,
    }),
  })
}

export function fetchExamples() {
  return apiFetch<ExamplesListResponse>('/api/examples')
}

export function loadExample(exampleName: string) {
  return apiFetch<LoadExampleResponse>(`/api/examples/${encodeURIComponent(exampleName)}/load`, {
    method: 'POST',
  })
}

export function predict(sessionId: string, featureVector: Record<string, FeatureValue>) {
  return apiFetch<PredictResponse>('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      feature_vector: featureVector,
    }),
  })
}

export function selectDatasetRow(sessionId: string, rowIndex: number) {
  return apiFetch<SelectRowResponse>('/api/sample/select-row', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      row_index: rowIndex,
    }),
  })
}

export function generateCounterfactual(
  sessionId: string,
  rowIndex: number,
  threshold: number,
  targetClass: number,
  featureVector?: Record<string, FeatureValue>,
  maxSteps?: number,
) {
  const body: {
    session_id: string
    row_index: number
    threshold: number
    target_class: number
    feature_vector?: Record<string, FeatureValue>
    max_steps?: number
  } = {
    session_id: sessionId,
    row_index: rowIndex,
    threshold,
    target_class: targetClass,
  }

  if (featureVector !== undefined) {
    body.feature_vector = featureVector
  }

  if (maxSteps !== undefined) {
    body.max_steps = maxSteps
  }

  return apiFetch<CounterfactualResponse>('/api/counterfactuals', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}
