import type { CounterfactualResponse, PredictionSummary as PredictionSummaryType } from '../types/api'

type CounterfactualPanelProps = {
  hasSession: boolean
  selectedRowIndex: number | null
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
  selectedRowIndex,
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
  const canGenerate =
    hasSession && selectedRowIndex !== null && currentThreshold !== null && currentLabel !== null && !busy && !isGenerating
  const canApplyChanges = canGenerate && Boolean(result?.counterfactuals[0]?.changes.length)

  function renderBody() {
    if (!hasSession) {
      return <div className="empty-state">Load a model and start a session before generating counterfactuals.</div>
    }

    if (selectedRowIndex === null) {
      return <div className="empty-state">Select a dataset row to enable counterfactual generation.</div>
    }

    if (!prediction) {
      return <div className="empty-state">Wait for a prediction on the selected row before generating a counterfactual.</div>
    }

    if (isGenerating) {
      return <div className="counterfactual-status">Generating counterfactual for the selected row…</div>
    }

    if (errorMessage) {
      return <div className="error-banner counterfactual-error">{errorMessage}</div>
    }

    if (!result) {
      return (
        <div className="empty-state">
          Generate a counterfactual using the current row, current threshold, and the opposite predicted class.
        </div>
      )
    }

    if (result.counterfactuals.length === 0) {
      return <div className="empty-state">No counterfactual was found for this row within `max_steps = 3`.</div>
    }

    return (
      <div className="counterfactual-card">
        <div className="counterfactual-stats">
          <div className="info-card">
            <span>Current</span>
            <strong>{result.current_prediction}</strong>
            <small>{result.current_probability.toFixed(3)}</small>
          </div>
          <div className="info-card">
            <span>Counterfactual</span>
            <strong>{result.counterfactuals[0].new_prediction}</strong>
            <small>{result.counterfactuals[0].new_probability.toFixed(3)}</small>
          </div>
          <div className="info-card">
            <span>Cost</span>
            <strong>{result.counterfactuals[0].cost.toFixed(3)}</strong>
            <small>{`${result.counterfactuals[0].steps.length} steps`}</small>
          </div>
          <div className="info-card">
            <span>Runtime</span>
            <strong>{`${result.runtime_ms.toFixed(1)} ms`}</strong>
            <small>{`Threshold ${result.threshold.toFixed(3)}`}</small>
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
          <span className="panel-caption">Generate a minimal flip for the selected dataset row</span>
        </div>
      </div>

      <div className="counterfactual-actions">
        <div className="counterfactual-meta">
          <span>{hasSession ? 'Session active' : 'No active session'}</span>
          <span>{selectedRowIndex === null ? 'No row selected' : `Row ${selectedRowIndex} selected`}</span>
          <span>{currentThreshold === null ? 'Threshold unavailable' : `Threshold ${currentThreshold.toFixed(2)}`}</span>
          <span>{targetClass === null ? 'Target unavailable' : `Target class ${targetClass}`}</span>
        </div>
        <button type="button" className="action-button" disabled={!canGenerate} onClick={onGenerate}>
          {isGenerating ? 'Generating…' : 'Generate Counterfactual'}
        </button>
        <button type="button" className="action-button" disabled={!canApplyChanges} onClick={onApplyChanges}>
          Apply Changes
        </button>
      </div>
      {renderBody()}
    </section>
  )
}
