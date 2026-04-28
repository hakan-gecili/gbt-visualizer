import { useEffect, useRef, useState } from 'react'

import { ContributionChartPanel } from './components/ContributionChartPanel'
import { CounterfactualPanel } from './components/CounterfactualPanel'
import { DatasetTable } from './components/DatasetTable'
import { ExamplesPanel } from './components/ExamplesPanel'
import { FeatureImportancePanel } from './components/FeatureImportancePanel'
import { FeatureControlPanel } from './components/FeatureControlPanel'
import { FloatingProbabilityGauge } from './components/FloatingProbabilityGauge'
import { PathExplanationPanel } from './components/PathExplanationPanel'
import { PredictionSummary } from './components/PredictionSummary'
import { RadialTreeView } from './components/RadialTreeView'
import { SelectedTreePanel } from './components/SelectedTreePanel'
import { UploadPanel } from './components/UploadPanel'
import { useDebouncedValue } from './hooks/useDebouncedValue'
import { fetchExamples, fetchLayout, generateCounterfactual, loadExample, predict, selectDatasetRow, uploadDataset, uploadFeatureSchema, uploadModel } from './services/model'
import type {
  CounterfactualResponse,
  DatasetSummary,
  ExampleDatasetGroup,
  FeatureImportanceEntry,
  FeatureMetadata,
  FeatureValue,
  PredictionSummary as PredictionSummaryType,
  PreviewPayload,
  TreeLayout,
  TreePredictionResult,
} from './types/api'

function buildDefaultFeatureVector(featureMetadata: FeatureMetadata[]) {
  return Object.fromEntries(
    featureMetadata.map((feature) => [feature.name, feature.default_value ?? 0]),
  ) as Record<string, FeatureValue>
}

function serializeFeatureVector(featureVector: Record<string, FeatureValue>) {
  return JSON.stringify(
    Object.entries(featureVector).sort(([left], [right]) => left.localeCompare(right)),
  )
}

function isValidFeatureValue(feature: FeatureMetadata, value: FeatureValue) {
  if (value === null) {
    return feature.missing_allowed
  }
  if (feature.type === 'numeric') {
    return typeof value === 'number' && Number.isFinite(value)
  }
  return feature.options.some((option) => option.value === value)
}

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [modelFamily, setModelFamily] = useState<string | null>(null)
  const [availableExamples, setAvailableExamples] = useState<ExampleDatasetGroup[]>([])
  const [selectedExample, setSelectedExample] = useState('')
  const [featureMetadata, setFeatureMetadata] = useState<FeatureMetadata[]>([])
  const [featureVector, setFeatureVector] = useState<Record<string, FeatureValue>>({})
  const [layoutTrees, setLayoutTrees] = useState<TreeLayout[]>([])
  const [datasetPreview, setDatasetPreview] = useState<PreviewPayload | null>(null)
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null)
  const [prediction, setPrediction] = useState<PredictionSummaryType | null>(null)
  const [treeResults, setTreeResults] = useState<TreePredictionResult[]>([])
  const [globalFeatureImportance, setGlobalFeatureImportance] = useState<FeatureImportanceEntry[]>([])
  const [isFeatureImportanceOpen, setIsFeatureImportanceOpen] = useState(false)
  const [panelScale, setPanelScale] = useState(1)
  const [hoveredTreeIndex, setHoveredTreeIndex] = useState<number | null>(null)
  const [selectedTreeIndex, setSelectedTreeIndex] = useState<number | null>(null)
  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null)
  const [counterfactualResult, setCounterfactualResult] = useState<CounterfactualResponse | null>(null)
  const [isGeneratingCounterfactual, setIsGeneratingCounterfactual] = useState(false)
  const [counterfactualError, setCounterfactualError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const predictionRequestIdRef = useRef(0)
  const appliedFeatureVectorKeyRef = useRef('')
  const appShellRef = useRef<HTMLElement | null>(null)
  const sidebarRef = useRef<HTMLElement | null>(null)
  const [isRadialDarkMode, setIsRadialDarkMode] = useState(false)

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
      setModelFamily(modelResponse.model_summary.model_family)
      setFeatureMetadata(modelResponse.feature_metadata)
      setFeatureVector(nextFeatureVector)
      setGlobalFeatureImportance(modelResponse.model_summary.global_feature_importance)
      setLayoutTrees(layoutResponse.layout.trees)
      setSelectedTreeIndex(layoutResponse.layout.trees[0]?.tree_index ?? null)
      setDatasetPreview(null)
      setDatasetSummary(null)
      setPrediction(null)
      setTreeResults([])
      setCounterfactualResult(null)
      setCounterfactualError(null)
      setIsFeatureImportanceOpen(false)
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
      setCounterfactualResult(null)
      setCounterfactualError(null)
      appliedFeatureVectorKeyRef.current = ''
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Dataset upload failed.')
    } finally {
      setBusy(false)
    }
  }

  async function handleSchemaUpload(file: File) {
    if (!sessionId) {
      return
    }

    setBusy(true)
    setErrorMessage(null)

    try {
      const response = await uploadFeatureSchema(sessionId, file)
      setFeatureMetadata(response.feature_metadata)
      setDatasetSummary(response.dataset_summary)
      setDatasetPreview(response.preview)
      const nextFeatureVector = buildDefaultFeatureVector(response.feature_metadata)
      setFeatureVector(nextFeatureVector)
      setSelectedRowIndex(null)
      setHoveredTreeIndex(null)
      setCounterfactualResult(null)
      setCounterfactualError(null)
      appliedFeatureVectorKeyRef.current = ''
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Feature schema upload failed.')
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
      setModelFamily(response.model_summary.model_family)
      setFeatureMetadata(response.feature_metadata)
      setFeatureVector(nextFeatureVector)
      setGlobalFeatureImportance(response.model_summary.global_feature_importance)
      setLayoutTrees(layoutResponse.layout.trees)
      setSelectedTreeIndex(layoutResponse.layout.trees[0]?.tree_index ?? null)
      setDatasetPreview(response.preview)
      setDatasetSummary(response.dataset_summary)
      setPrediction(null)
      setTreeResults([])
      setCounterfactualResult(null)
      setCounterfactualError(null)
      setIsFeatureImportanceOpen(false)
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
      setCounterfactualResult(null)
      setCounterfactualError(null)
      setHoveredTreeIndex(null)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Row selection failed.')
    }
  }

  function handleFeatureChange(featureName: string, value: FeatureValue) {
    setSelectedRowIndex(null)
    setHoveredTreeIndex(null)
    setCounterfactualResult(null)
    setCounterfactualError(null)
    setFeatureVector((current) => ({
      ...current,
      [featureName]: value,
    }))
  }

  async function handleGenerateCounterfactual() {
    if (!sessionId || selectedRowIndex === null || !prediction) {
      return
    }

    setIsGeneratingCounterfactual(true)
    setCounterfactualError(null)

    try {
      const targetClass = prediction.predicted_label === 1 ? 0 : 1
      const response = await generateCounterfactual(
        sessionId,
        selectedRowIndex,
        prediction.decision_threshold,
        targetClass,
        3,
      )
      setCounterfactualResult(response)
    } catch (error) {
      setCounterfactualResult(null)
      setCounterfactualError(error instanceof Error ? error.message : 'Counterfactual generation failed.')
    } finally {
      setIsGeneratingCounterfactual(false)
    }
  }

  function handleApplyCounterfactualChanges() {
    const changes = counterfactualResult?.counterfactuals[0]?.changes ?? []
    if (!changes.length) {
      return
    }

    for (const change of changes) {
      const feature = featureMetadata.find((item) => item.name === change.feature)
      if (!feature || !isValidFeatureValue(feature, change.new_value)) {
        const expectedValues = feature?.options.map((option) => option.value).join(', ') ?? 'a known feature'
        setCounterfactualError(
          `Cannot apply counterfactual change for ${change.feature}: ${String(change.new_value)} is not valid. Expected ${expectedValues}.`,
        )
        return
      }
    }

    setSelectedRowIndex(null)
    setHoveredTreeIndex(null)
    setCounterfactualResult(null)
    setCounterfactualError(null)
    appliedFeatureVectorKeyRef.current = ''
    setFeatureVector((current) => {
      const nextFeatureVector = { ...current }
      for (const change of changes) {
        nextFeatureVector[change.feature] = change.new_value
      }
      return nextFeatureVector
    })
  }

  return (
    <main ref={appShellRef} className="app-shell">
      <aside ref={sidebarRef} className="sidebar">
        <UploadPanel
          sessionId={sessionId}
          modelFamily={modelFamily}
          busy={busy}
          onModelUpload={handleModelUpload}
          onDatasetUpload={handleDatasetUpload}
          onSchemaUpload={handleSchemaUpload}
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
        <CounterfactualPanel
          hasSession={sessionId !== null}
          selectedRowIndex={selectedRowIndex}
          prediction={prediction}
          busy={busy}
          isGenerating={isGeneratingCounterfactual}
          errorMessage={counterfactualError}
          result={counterfactualResult}
          onGenerate={handleGenerateCounterfactual}
          onApplyChanges={handleApplyCounterfactualChanges}
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
          panelScale={panelScale}
          onPanelScaleChange={setPanelScale}
          hoveredTreeIndex={hoveredTreeIndex}
          onHoverTree={setHoveredTreeIndex}
          selectedTreeIndex={selectedTreeIndex}
          onSelectTree={setSelectedTreeIndex}
          isDarkMode={isRadialDarkMode}
          onToggleDarkMode={() => setIsRadialDarkMode((current) => !current)}
        />
        <SelectedTreePanel
          trees={layoutTrees}
          treeResults={treeResults}
          selectedTreeIndex={selectedTreeIndex}
        />
        <PathExplanationPanel
          trees={layoutTrees}
          treeResults={treeResults}
          selectedTreeIndex={selectedTreeIndex}
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
      {sessionId ? (
        <FloatingProbabilityGauge
          probability={prediction?.probability ?? null}
          isDarkMode={isRadialDarkMode}
          appShellRef={appShellRef}
          sidebarRef={sidebarRef}
          isFeatureImportanceOpen={isFeatureImportanceOpen}
          onToggleFeatureImportance={() => setIsFeatureImportanceOpen((current) => !current)}
        />
      ) : null}
      <FeatureImportancePanel
        isOpen={isFeatureImportanceOpen}
        onClose={() => setIsFeatureImportanceOpen(false)}
        globalImportance={globalFeatureImportance}
        localImportance={prediction?.local_feature_importance ?? []}
      />
    </main>
  )
}

export default App
