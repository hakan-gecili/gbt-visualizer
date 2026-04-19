import { useEffect, useRef, useState } from 'react'

import { ContributionChartPanel } from './components/ContributionChartPanel'
import { DatasetTable } from './components/DatasetTable'
import { ExamplesPanel } from './components/ExamplesPanel'
import { FeatureControlPanel } from './components/FeatureControlPanel'
import { PredictionSummary } from './components/PredictionSummary'
import { RadialTreeView } from './components/RadialTreeView'
import { UploadPanel } from './components/UploadPanel'
import { useDebouncedValue } from './hooks/useDebouncedValue'
import { fetchExamples, fetchLayout, loadExample, predict, selectDatasetRow, uploadDataset, uploadModel } from './services/model'
import type {
  DatasetSummary,
  FeatureMetadata,
  PredictionSummary as PredictionSummaryType,
  PreviewPayload,
  TreeLayout,
  TreePredictionResult,
} from './types/api'

function buildDefaultFeatureVector(featureMetadata: FeatureMetadata[]) {
  return Object.fromEntries(
    featureMetadata.map((feature) => [feature.name, feature.default_value ?? 0]),
  ) as Record<string, number>
}

function serializeFeatureVector(featureVector: Record<string, number>) {
  return JSON.stringify(
    Object.entries(featureVector).sort(([left], [right]) => left.localeCompare(right)),
  )
}

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [availableExamples, setAvailableExamples] = useState<string[]>([])
  const [selectedExample, setSelectedExample] = useState('')
  const [featureMetadata, setFeatureMetadata] = useState<FeatureMetadata[]>([])
  const [featureVector, setFeatureVector] = useState<Record<string, number>>({})
  const [layoutTrees, setLayoutTrees] = useState<TreeLayout[]>([])
  const [datasetPreview, setDatasetPreview] = useState<PreviewPayload | null>(null)
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null)
  const [prediction, setPrediction] = useState<PredictionSummaryType | null>(null)
  const [treeResults, setTreeResults] = useState<TreePredictionResult[]>([])
  const [panelScale, setPanelScale] = useState(1)
  const [hoveredTreeIndex, setHoveredTreeIndex] = useState<number | null>(null)
  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const predictionRequestIdRef = useRef(0)
  const appliedFeatureVectorKeyRef = useRef('')

  const debouncedFeatureVector = useDebouncedValue(featureVector, 150)

  useEffect(() => {
    async function loadAvailableExamples() {
      try {
        const response = await fetchExamples()
        setAvailableExamples(response.examples)
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : 'Failed to load examples.')
      }
    }

    void loadAvailableExamples()
  }, [])

  useEffect(() => {
    async function runPrediction() {
      if (!sessionId || !Object.keys(debouncedFeatureVector).length) {
        return
      }
      const featureVectorKey = serializeFeatureVector(debouncedFeatureVector)
      if (featureVectorKey === appliedFeatureVectorKeyRef.current) {
        return
      }

      try {
        const requestId = predictionRequestIdRef.current + 1
        predictionRequestIdRef.current = requestId
        const response = await predict(sessionId, debouncedFeatureVector)
        if (requestId !== predictionRequestIdRef.current) {
          return
        }
        appliedFeatureVectorKeyRef.current = featureVectorKey
        setErrorMessage(null)
        setPrediction(response.prediction)
        setTreeResults(response.tree_results)
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : 'Prediction request failed.')
      }
    }

    void runPrediction()
  }, [debouncedFeatureVector, sessionId])

  async function handleModelUpload(file: File) {
    setBusy(true)
    setErrorMessage(null)

    try {
      const modelResponse = await uploadModel(file)
      const layoutResponse = await fetchLayout(modelResponse.session_id)
      const nextFeatureVector = buildDefaultFeatureVector(modelResponse.feature_metadata)

      setSelectedExample('')
      setSessionId(modelResponse.session_id)
      setFeatureMetadata(modelResponse.feature_metadata)
      setFeatureVector(nextFeatureVector)
      setLayoutTrees(layoutResponse.layout.trees)
      setDatasetPreview(null)
      setDatasetSummary(null)
      setPrediction(null)
      setTreeResults([])
      setHoveredTreeIndex(null)
      setSelectedRowIndex(null)
      predictionRequestIdRef.current = 0
      appliedFeatureVectorKeyRef.current = ''
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Model upload failed.')
    } finally {
      setBusy(false)
    }
  }

  async function handleDatasetUpload(file: File) {
    if (!sessionId) {
      return
    }

    setBusy(true)
    setErrorMessage(null)

    try {
      const response = await uploadDataset(sessionId, file)
      setDatasetPreview(response.preview)
      setDatasetSummary(response.dataset_summary)
      setFeatureMetadata(response.feature_metadata)
      const nextFeatureVector = buildDefaultFeatureVector(response.feature_metadata)
      setFeatureVector(nextFeatureVector)
      setSelectedRowIndex(null)
      setHoveredTreeIndex(null)
      appliedFeatureVectorKeyRef.current = ''
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Dataset upload failed.')
    } finally {
      setBusy(false)
    }
  }

  async function handleExampleSelect(exampleName: string) {
    setSelectedExample(exampleName)
    if (!exampleName) {
      return
    }

    setBusy(true)
    setErrorMessage(null)

    try {
      const response = await loadExample(exampleName)
      const layoutResponse = await fetchLayout(response.session_id)
      const nextFeatureVector = buildDefaultFeatureVector(response.feature_metadata)

      setSessionId(response.session_id)
      setFeatureMetadata(response.feature_metadata)
      setFeatureVector(nextFeatureVector)
      setLayoutTrees(layoutResponse.layout.trees)
      setDatasetPreview(response.preview)
      setDatasetSummary(response.dataset_summary)
      setPrediction(null)
      setTreeResults([])
      setHoveredTreeIndex(null)
      setSelectedRowIndex(null)
      predictionRequestIdRef.current = 0
      appliedFeatureVectorKeyRef.current = ''
    } catch (error) {
      setSelectedExample('')
      setErrorMessage(error instanceof Error ? error.message : 'Example load failed.')
    } finally {
      setBusy(false)
    }
  }

  async function handleSelectRow(rowIndex: number) {
    if (!sessionId) {
      return
    }

    setErrorMessage(null)

    try {
      const requestId = predictionRequestIdRef.current + 1
      predictionRequestIdRef.current = requestId
      const response = await selectDatasetRow(sessionId, rowIndex)
      if (requestId !== predictionRequestIdRef.current) {
        return
      }
      setSelectedRowIndex(rowIndex)
      setFeatureVector(response.sample.feature_vector)
      appliedFeatureVectorKeyRef.current = serializeFeatureVector(response.sample.feature_vector)
      setErrorMessage(null)
      setPrediction(response.prediction)
      setTreeResults(response.tree_results)
      setHoveredTreeIndex(null)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Row selection failed.')
    }
  }

  function handleFeatureChange(featureName: string, value: number) {
    setSelectedRowIndex(null)
    setHoveredTreeIndex(null)
    setFeatureVector((current) => ({
      ...current,
      [featureName]: Number.isFinite(value) ? value : 0,
    }))
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <UploadPanel
          sessionId={sessionId}
          busy={busy}
          onModelUpload={handleModelUpload}
          onDatasetUpload={handleDatasetUpload}
        />
        <ExamplesPanel
          examples={availableExamples}
          selectedExample={selectedExample}
          busy={busy}
          onSelectExample={handleExampleSelect}
        />
        <FeatureControlPanel
          featureMetadata={featureMetadata}
          featureVector={featureVector}
          onFeatureChange={handleFeatureChange}
        />
      </aside>

      <section className="main-stage">
        {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}
        <PredictionSummary
          prediction={prediction}
          treeResults={treeResults}
        />
        <ContributionChartPanel
          treeResults={treeResults}
          panelScale={panelScale}
          hoveredTreeIndex={hoveredTreeIndex}
          onHoverTree={setHoveredTreeIndex}
        />
        <RadialTreeView
          trees={layoutTrees}
          treeResults={treeResults}
          probability={prediction?.probability ?? null}
          panelScale={panelScale}
          onPanelScaleChange={setPanelScale}
          hoveredTreeIndex={hoveredTreeIndex}
          onHoverTree={setHoveredTreeIndex}
        />
        <DatasetTable
          preview={datasetPreview}
          selectedRowIndex={selectedRowIndex}
          onSelectRow={handleSelectRow}
        />
        <footer className="status-bar">
          <span>{sessionId ? `${layoutTrees.length} tree sectors loaded` : 'Awaiting model upload'}</span>
          <span>
            {datasetSummary
              ? `${datasetSummary.num_rows} rows, ${datasetSummary.matched_feature_count} matched model features`
              : 'Manual feature editing active'}
          </span>
        </footer>
      </section>
    </main>
  )
}

export default App
