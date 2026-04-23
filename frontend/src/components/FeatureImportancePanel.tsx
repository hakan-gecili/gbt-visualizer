import { useEffect, useMemo, useState } from 'react'

import type { FeatureImportanceEntry } from '../types/api'

type FeatureImportancePanelProps = {
  isOpen: boolean
  onClose: () => void
  globalImportance: FeatureImportanceEntry[]
  localImportance: FeatureImportanceEntry[]
}

type ImportanceMode = 'global' | 'local'

const MAX_FEATURES = 12

function formatImportanceValue(value: number, mode: ImportanceMode) {
  if (mode === 'local') {
    return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`
  }
  return value.toFixed(2)
}

export function FeatureImportancePanel({
  isOpen,
  onClose,
  globalImportance,
  localImportance,
}: FeatureImportancePanelProps) {
  const [mode, setMode] = useState<ImportanceMode>('global')
  const subtitle = mode === 'global' ? 'Model-wide importance' : 'Importance for current prediction'

  useEffect(() => {
    if (!isOpen) {
      return
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  const entries = useMemo(() => {
    const source = mode === 'global' ? globalImportance : localImportance
    const sorted = [...source].sort((left, right) =>
      mode === 'global' ? right.value - left.value : Math.abs(right.value) - Math.abs(left.value),
    )
    return sorted.slice(0, MAX_FEATURES)
  }, [globalImportance, localImportance, mode])

  const maxAbsValue = Math.max(...entries.map((entry) => Math.abs(entry.value)), 0)

  return (
    <div
      className={isOpen ? 'feature-importance-shell open' : 'feature-importance-shell'}
      aria-hidden={!isOpen}
      onClick={onClose}
    >
      <aside
        className={isOpen ? 'feature-importance-panel open' : 'feature-importance-panel'}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="feature-importance-header">
          <div className="feature-importance-title-group">
            <h2>Feature Importance</h2>
            <span className="feature-importance-subtitle">{subtitle}</span>
          </div>
          <button type="button" className="feature-importance-close" onClick={onClose} aria-label="Close feature importance panel">
            ×
          </button>
        </div>

        <div className="feature-importance-mode-toggle" role="tablist" aria-label="Feature importance mode">
          <button
            type="button"
            className={mode === 'global' ? 'active' : ''}
            onClick={() => setMode('global')}
          >
            Global
          </button>
          <button
            type="button"
            className={mode === 'local' ? 'active' : ''}
            onClick={() => setMode('local')}
          >
            Local
          </button>
        </div>

        <div className="feature-importance-chart">
          {entries.length ? (
            entries.map((entry) => {
              const widthPercent = maxAbsValue === 0 ? 0 : (Math.abs(entry.value) / maxAbsValue) * 100
              return (
                <div key={`${mode}-${entry.feature_name}`} className="feature-importance-row">
                  <div className="feature-importance-row-header">
                    <span className="feature-importance-name" title={entry.feature_name}>
                      {entry.feature_name}
                    </span>
                    <strong className="feature-importance-value">{formatImportanceValue(entry.value, mode)}</strong>
                  </div>
                  {mode === 'global' ? (
                    <div className="feature-importance-bar-track">
                      <div className="feature-importance-bar global" style={{ width: `${widthPercent}%` }} />
                    </div>
                  ) : (
                    <div className="feature-importance-bar-track local">
                      <div className="feature-importance-zero-line" />
                      <div
                        className={entry.value >= 0 ? 'feature-importance-bar local positive' : 'feature-importance-bar local negative'}
                        style={{
                          width: `${widthPercent / 2}%`,
                          [entry.value >= 0 ? 'left' : 'right']: '50%',
                        }}
                      />
                    </div>
                  )}
                </div>
              )
            })
          ) : (
            <div className="empty-state">
              {mode === 'global'
                ? 'Feature importance will appear after a model is loaded.'
                : 'Local importance will appear after a prediction is available.'}
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}
