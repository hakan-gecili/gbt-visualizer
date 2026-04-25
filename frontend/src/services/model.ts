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

export function uploadModel(modelFile: File) {
  return apiFetch<ModelUploadResponse>('/api/model/upload', {
    method: 'POST',
    body: createFormData({ model_file: modelFile }),
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
  maxSteps = 3,
) {
  return apiFetch<CounterfactualResponse>('/api/counterfactuals', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      row_index: rowIndex,
      threshold,
      target_class: targetClass,
      max_steps: maxSteps,
    }),
  })
}
