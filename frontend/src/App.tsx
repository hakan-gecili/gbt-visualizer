import { useEffect, useMemo, useRef, useState } from 'react'

import { ContributionChartPanel } from './components/ContributionChartPanel'
import { CounterfactualPanel } from './components/CounterfactualPanel'
import { DatasetTable } from './components/DatasetTable'
import { EnsembleStructureCube } from './components/EnsembleStructureCube'
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
import { buildFeatureVectorPathEdgeSet } from './components/treeRenderUtils'
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

function randomOptionValue(feature: FeatureMetadata) {
  return feature.options[Math.floor(Math.random() * feature.options.length)]?.value ?? feature.default_value ?? null
}

function isIntegerLikeNumericFeature(feature: FeatureMetadata) {
  if (feature.min_value === null || feature.max_value === null) {
    return false
  }

  const hasIntegerBounds = Number.isInteger(feature.min_value) && Number.isInteger(feature.max_value)
  return hasIntegerBounds && feature.max_value - feature.min_value >= 1
}

function randomFeatureValue(feature: FeatureMetadata): FeatureValue {
  if (feature.type === 'binary' || feature.type === 'categorical') {
    return randomOptionValue(feature)
  }

  if (feature.min_value === null || feature.max_value === null) {
    return typeof feature.default_value === 'number' ? feature.default_value : 0
  }

  const min = Math.min(feature.min_value, feature.max_value)
  const max = Math.max(feature.min_value, feature.max_value)
  if (isIntegerLikeNumericFeature(feature)) {
    const integerMin = Math.ceil(min)
    const integerMax = Math.floor(max)
    return integerMin + Math.floor(Math.random() * (integerMax - integerMin + 1))
  }

  return min + Math.random() * (max - min)
}

function buildRandomFeatureVector(featureMetadata: FeatureMetadata[]) {
  return Object.fromEntries(
    featureMetadata.map((feature) => [feature.name, randomFeatureValue(feature)]),
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
  const [decisionThreshold, setDecisionThreshold] = useState(0.5)
  const [treeResults, setTreeResults] = useState<TreePredictionResult[]>([])
  const [globalFeatureImportance, setGlobalFeatureImportance] = useState<FeatureImportanceEntry[]>([])
  const [isFeatureImportanceOpen, setIsFeatureImportanceOpen] = useState(false)
  const [panelScale, setPanelScale] = useState(1)
  const [hoveredTreeIndex, setHoveredTreeIndex] = useState<number | null>(null)
  const [selectedTreeIndex, setSelectedTreeIndex] = useState<number | null>(null)
  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null)
  const [selectedRowFeatureVector, setSelectedRowFeatureVector] = useState<Record<string, FeatureValue> | null>(null)
  const [isRandomSample, setIsRandomSample] = useState(false)
  const [counterfactualResult, setCounterfactualResult] = useState<CounterfactualResponse | null>(null)
  const [isGeneratingCounterfactual, setIsGeneratingCounterfactual] = useState(false)
  const [counterfactualError, setCounterfactualError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [appliedFeatureVectorKey, setAppliedFeatureVectorKey] = useState('')
  const predictionRequestIdRef = useRef(0)
  const appliedFeatureVectorKeyRef = useRef('')
  const appShellRef = useRef<HTMLElement | null>(null)
  const sidebarRef = useRef<HTMLElement | null>(null)
  const [isRadialDarkMode, setIsRadialDarkMode] = useState(false)

  const debouncedFeatureVector = useDebouncedValue(featureVector, 150)
  const isFeatureVectorEdited =
    selectedRowFeatureVector !== null &&
    serializeFeatureVector(featureVector) !== serializeFeatureVector(selectedRowFeatureVector)
  const isPredictionCurrent =
    prediction !== null && serializeFeatureVector(featureVector) === appliedFeatureVectorKey
  const displayedPrediction = prediction
    ? {
        ...prediction,
        decision_threshold: decisionThreshold,
        predicted_label: Number(prediction.probability >= decisionThreshold),
      }
    : null
  const counterfactualPathEdgeMap = useMemo(() => {
    if (!counterfactualResult?.counterfactuals.length) {
      return new Map<number, Set<string>>()
    }

    const counterfactualVector = { ...featureVector }
    for (const change of counterfactualResult.counterfactuals[0].changes) {
      counterfactualVector[change.feature] = change.new_value
    }

    return new Map(
      layoutTrees.map((tree) => [
        tree.tree_index,
        buildFeatureVectorPathEdgeSet(tree, featureMetadata, counterfactualVector),
      ]),
    )
  }, [counterfactualResult, featureMetadata, featureVector, layoutTrees])

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
        setAppliedFeatureVectorKeyValue(featureVectorKey)
        setErrorMessage(null)
        setPrediction(response.prediction)
        setTreeResults(response.tree_results)
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : 'Prediction request failed.')
      }
    }

    void runPrediction()
  }, [debouncedFeatureVector, sessionId])

  function setAppliedFeatureVectorKeyValue(featureVectorKey: string) {
    appliedFeatureVectorKeyRef.current = featureVectorKey
    setAppliedFeatureVectorKey(featureVectorKey)
  }

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
      setSelectedTreeIndex(null)
      setDatasetPreview(null)
      setDatasetSummary(null)
      setPrediction(null)
      setDecisionThreshold(modelResponse.model_summary.decision_threshold)
      setTreeResults([])
      setCounterfactualResult(null)
      setCounterfactualError(null)
      setIsFeatureImportanceOpen(false)
      setHoveredTreeIndex(null)
      setSelectedRowIndex(null)
      setSelectedRowFeatureVector(null)
      setIsRandomSample(false)
      predictionRequestIdRef.current = 0
      setAppliedFeatureVectorKeyValue('')
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
      setSelectedRowFeatureVector(null)
      setIsRandomSample(false)
      setHoveredTreeIndex(null)
      setCounterfactualResult(null)
      setCounterfactualError(null)
      setAppliedFeatureVectorKeyValue('')
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
      setSelectedRowFeatureVector(null)
      setIsRandomSample(false)
      setHoveredTreeIndex(null)
      setCounterfactualResult(null)
      setCounterfactualError(null)
      setAppliedFeatureVectorKeyValue('')
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
      setSelectedTreeIndex(null)
      setDatasetPreview(response.preview)
      setDatasetSummary(response.dataset_summary)
      setPrediction(null)
      setDecisionThreshold(response.model_summary.decision_threshold)
      setTreeResults([])
      setCounterfactualResult(null)
      setCounterfactualError(null)
      setIsFeatureImportanceOpen(false)
      setHoveredTreeIndex(null)
      setSelectedRowIndex(null)
      setSelectedRowFeatureVector(null)
      setIsRandomSample(false)
      predictionRequestIdRef.current = 0
      setAppliedFeatureVectorKeyValue('')
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
      setSelectedRowFeatureVector(response.sample.feature_vector)
      setIsRandomSample(false)
      setFeatureVector(response.sample.feature_vector)
      setAppliedFeatureVectorKeyValue(serializeFeatureVector(response.sample.feature_vector))
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
    setHoveredTreeIndex(null)
    setCounterfactualResult(null)
    setCounterfactualError(null)
    setFeatureVector((current) => ({
      ...current,
      [featureName]: value,
    }))
  }

  function handleGenerateRandomSample() {
    if (!featureMetadata.length) {
      return
    }

    setHoveredTreeIndex(null)
    setCounterfactualResult(null)
    setCounterfactualError(null)
    setIsRandomSample(true)
    setAppliedFeatureVectorKeyValue('')
    setFeatureVector(buildRandomFeatureVector(featureMetadata))
  }

  async function handleGenerateCounterfactual() {
    const counterfactualRowIndex = selectedRowIndex ?? (isRandomSample ? 0 : null)
    if (!sessionId || counterfactualRowIndex === null || !prediction || !isPredictionCurrent) {
      return
    }

    setIsGeneratingCounterfactual(true)
    setCounterfactualError(null)

    try {
      const currentLabel = Number(prediction.probability >= decisionThreshold)
      const targetClass = currentLabel === 1 ? 0 : 1
      const response = await generateCounterfactual(
        sessionId,
        counterfactualRowIndex,
        decisionThreshold,
        targetClass,
        featureVector,
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

    setHoveredTreeIndex(null)
    setCounterfactualResult(null)
    setCounterfactualError(null)
    setIsRandomSample(false)
    setAppliedFeatureVectorKeyValue('')
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
          canGenerateRandomSample={featureMetadata.length > 0}
          onFeatureChange={handleFeatureChange}
          onGenerateRandomSample={handleGenerateRandomSample}
        />
      </aside>

      <section className="main-stage">
        {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}
        <PredictionSummary
          prediction={displayedPrediction}
          treeResults={treeResults}
          threshold={decisionThreshold}
          onThresholdChange={setDecisionThreshold}
        />
        <CounterfactualPanel
          hasSession={sessionId !== null}
          modelFamily={modelFamily}
          selectedRowIndex={selectedRowIndex}
          isFeatureVectorEdited={isFeatureVectorEdited}
          isRandomSample={isRandomSample}
          isPredictionCurrent={isPredictionCurrent}
          prediction={displayedPrediction}
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
          counterfactualPathEdgeMap={counterfactualPathEdgeMap}
          panelScale={panelScale}
          onPanelScaleChange={setPanelScale}
          hoveredTreeIndex={hoveredTreeIndex}
          onHoverTree={setHoveredTreeIndex}
          selectedTreeIndex={selectedTreeIndex}
          onSelectTree={setSelectedTreeIndex}
          isDarkMode={isRadialDarkMode}
          onToggleDarkMode={() => setIsRadialDarkMode((current) => !current)}
        />
        <EnsembleStructureCube
          trees={layoutTrees}
          featureMetadata={featureMetadata}
          treeResults={treeResults}
          selectedTreeIndex={selectedTreeIndex}
          onSelectTree={setSelectedTreeIndex}
        />
        <SelectedTreePanel
          trees={layoutTrees}
          treeResults={treeResults}
          selectedTreeIndex={selectedTreeIndex}
          onSelectTree={setSelectedTreeIndex}
          counterfactualPathEdgeMap={counterfactualPathEdgeMap}
        />
        <PathExplanationPanel
          trees={layoutTrees}
          treeResults={treeResults}
          selectedTreeIndex={selectedTreeIndex}
          featureMetadata={featureMetadata}
          featureVector={featureVector}
          counterfactualResult={counterfactualResult}
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
