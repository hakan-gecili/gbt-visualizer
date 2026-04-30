import type { CounterfactualResponse, PredictionSummary as PredictionSummaryType } from '../types/api'

type CounterfactualPanelProps = {
  hasSession: boolean
  modelFamily: string | null
  selectedRowIndex: number | null
  isFeatureVectorEdited: boolean
  prediction: PredictionSummaryType | null
  busy: boolean
  isGenerating: boolean
  errorMessage: string | null
  result: CounterfactualResponse | null
  onGenerate: () => void
  onApplyChanges: () => void
}

function formatValue(value: string | number | null) {
  if (value === null) {
    return 'null'
  }
  if (typeof value === 'number') {
    return String(value)
  }
  return value
}

export function CounterfactualPanel({
  hasSession,
  modelFamily,
  selectedRowIndex,
  isFeatureVectorEdited,
  prediction,
  busy,
  isGenerating,
  errorMessage,
  result,
  onGenerate,
  onApplyChanges,
}: CounterfactualPanelProps) {
  const currentThreshold = prediction?.decision_threshold ?? null
  const currentLabel = prediction?.predicted_label ?? null
  const targetClass = currentLabel === null ? null : currentLabel === 1 ? 0 : 1
  const supportsCounterfactuals = ['lightgbm', 'xgboost'].includes((modelFamily ?? '').toLowerCase())
  const startingPointText =
    selectedRowIndex === null
      ? 'No row selected'
      : isFeatureVectorEdited
        ? `Starting point: Edited feature values (from row ${selectedRowIndex})`
        : `Starting point: Dataset row ${selectedRowIndex}`
  const canGenerate =
    hasSession &&
    supportsCounterfactuals &&
    selectedRowIndex !== null &&
    currentThreshold !== null &&
    currentLabel !== null &&
    !busy &&
    !isGenerating
  const canApplyChanges = canGenerate && Boolean(result?.counterfactuals[0]?.changes.length)

  function renderBody() {
    if (!hasSession) {
      return <div className="empty-state">Load a model and start a session before generating counterfactuals.</div>
    }

    if (!supportsCounterfactuals) {
      return <div className="empty-state">Counterfactual generation is not available for this model family.</div>
    }

    if (selectedRowIndex === null) {
      return <div className="empty-state">Select a dataset row to enable counterfactual generation.</div>
    }

    if (!prediction) {
      return <div className="empty-state">Wait for a prediction on the selected row before generating a counterfactual.</div>
    }

    if (isGenerating) {
      return <div className="counterfactual-status">Generating counterfactual from the current feature values…</div>
    }

    if (errorMessage) {
      return <div className="error-banner counterfactual-error">{errorMessage}</div>
    }

    if (!result) {
      return (
        <div className="empty-state">
          Generate a counterfactual using the current feature values, current threshold, and the opposite predicted class.
        </div>
      )
    }

    if (result.counterfactuals.length === 0) {
      return <div className="empty-state">No counterfactual was found for this row within the configured search limit.</div>
    }

    return (
      <div className="counterfactual-card">
        <div className="counterfactual-stats">
          <div className="counterfactual-metric">
            <span className="counterfactual-metric-label">Current</span>
            <div className="counterfactual-metric-value">
              <strong>{result.current_prediction}</strong>
              <small>{`${(result.current_probability * 100).toFixed(1)}%`}</small>
            </div>
          </div>
          <div className="counterfactual-metric">
            <span className="counterfactual-metric-label">Counterfactual</span>
            <div className="counterfactual-metric-value">
              <strong>{result.counterfactuals[0].new_prediction}</strong>
              <small>{`${(result.counterfactuals[0].new_probability * 100).toFixed(1)}%`}</small>
            </div>
          </div>
          <div className="counterfactual-metric">
            <span className="counterfactual-metric-label">Cost</span>
            <div className="counterfactual-metric-value">
              <strong>{result.counterfactuals[0].cost.toFixed(3)}</strong>
              <small>{`${result.counterfactuals[0].steps.length} steps`}</small>
            </div>
          </div>
          <div className="counterfactual-metric">
            <span className="counterfactual-metric-label">Runtime</span>
            <div className="counterfactual-metric-value">
              <strong>{result.runtime_ms.toFixed(1)}</strong>
              <small>ms</small>
            </div>
          </div>
          <div className="counterfactual-metric">
            <span className="counterfactual-metric-label">Threshold</span>
            <div className="counterfactual-metric-value">
              <strong>{result.threshold.toFixed(3)}</strong>
            </div>
          </div>
        </div>

        <div className="counterfactual-changes">
          <span className="panel-caption">Changes</span>
          {result.counterfactuals[0].changes.length ? (
            <ul className="counterfactual-change-list">
              {result.counterfactuals[0].changes.map((change) => (
                <li key={`${change.feature}-${String(change.old_value)}-${String(change.new_value)}`}>
                  <strong>{change.feature}</strong>
                  <span>{`${formatValue(change.old_value)} → ${formatValue(change.new_value)}`}</span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="empty-inline">No explicit feature changes were returned.</div>
          )}
        </div>
      </div>
    )
  }

  return (
    <section className="panel counterfactual-panel">
      <div className="panel-header">
        <div>
          <h2>Counterfactual</h2>
          <span className="panel-caption">Generate a minimal flip from the current feature values</span>
        </div>
      </div>

      <div className="counterfactual-actions">
        <div className="counterfactual-meta">
          <div className="counterfactual-meta-row">
            <span>{hasSession ? 'Session active' : 'No active session'}</span>
            <span>{startingPointText}</span>
          </div>
          <div className="counterfactual-meta-row">
            <span>{currentThreshold === null ? 'Threshold unavailable' : `Threshold ${currentThreshold.toFixed(3)}`}</span>
            <span>{targetClass === null ? 'Target unavailable' : `Target class ${targetClass}`}</span>
            <span>{supportsCounterfactuals ? `${modelFamily} counterfactuals` : 'Counterfactuals unavailable'}</span>
          </div>
        </div>
        <div className="counterfactual-buttons">
          <button type="button" className="action-button" disabled={!canGenerate} onClick={onGenerate}>
            {isGenerating ? 'Generating…' : 'Generate Counterfactual'}
          </button>
          <button type="button" className="action-button" disabled={!canApplyChanges} onClick={onApplyChanges}>
            Apply Changes
          </button>
        </div>
      </div>
      {renderBody()}
    </section>
  )
}
