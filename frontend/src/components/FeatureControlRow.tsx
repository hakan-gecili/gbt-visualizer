import type { FeatureMetadata, FeatureValue } from '../types/api'

type FeatureControlRowProps = {
  feature: FeatureMetadata
  value: FeatureValue
  onChange: (nextValue: FeatureValue) => void
}

function resolveNumericBounds(feature: FeatureMetadata, value: number | null) {
  if (feature.min_value !== null && feature.max_value !== null && feature.min_value !== feature.max_value) {
    return {
      min: feature.min_value,
      max: feature.max_value,
      step: Math.max((feature.max_value - feature.min_value) / 200, 0.01),
    }
  }

  const fallbackValue = typeof feature.default_value === 'number' ? feature.default_value : 0
  const amplitude = Math.max(Math.abs(value ?? 0), Math.abs(fallbackValue), 1)
  return {
    min: -amplitude,
    max: amplitude,
    step: Math.max((amplitude * 2) / 200, 0.01),
  }
}

function nextPresentValue(feature: FeatureMetadata, value: FeatureValue) {
  if (value !== null) {
    return value
  }
  if (feature.default_value !== null) {
    return feature.default_value
  }
  if (feature.type === 'numeric') {
    return 0
  }
  return feature.options[0]?.value ?? null
}

export function FeatureControlRow({ feature, value, onChange }: FeatureControlRowProps) {
  const isMissing = value === null
  const effectiveValue = nextPresentValue(feature, value)
  const featureLabel = feature.name || feature.short_name

  return (
    <div className={feature.type === 'numeric' ? 'feature-row numeric-feature-row' : 'feature-row'}>
      <div className="feature-label" title={feature.name}>
        <span>{featureLabel}</span>
      </div>

      <div className="feature-control">
        {feature.type === 'numeric' ? (
          (() => {
            const numericValue = typeof effectiveValue === 'number' ? effectiveValue : 0
            const bounds = resolveNumericBounds(feature, typeof value === 'number' ? value : null)
            return (
              <>
                <input
                  className="feature-slider"
                  type="range"
                  min={bounds.min}
                  max={bounds.max}
                  step={bounds.step}
                  value={numericValue}
                  disabled={isMissing}
                  onChange={(event) => onChange(Number(event.target.value))}
                />
                <input
                  className="feature-number"
                  type="number"
                  value={isMissing ? '' : numericValue}
                  step={bounds.step}
                  disabled={isMissing}
                  onChange={(event) => onChange(event.target.value === '' ? null : Number(event.target.value))}
                />
              </>
            )
          })()
        ) : null}

        {feature.type === 'binary' ? (
          <div className="feature-binary-group" role="radiogroup" aria-label={`${feature.name} options`}>
            {feature.options.map((option) => {
              const active = value === option.value
              return (
                <button
                  key={option.value}
                  type="button"
                  className={active ? 'feature-chip active' : 'feature-chip'}
                  disabled={isMissing}
                  onClick={() => onChange(option.value)}
                >
                  {option.label}
                </button>
              )
            })}
          </div>
        ) : null}

        {feature.type === 'categorical' ? (
          <select
            className="feature-select"
            value={typeof effectiveValue === 'string' ? effectiveValue : feature.options[0]?.value ?? ''}
            disabled={isMissing}
            onChange={(event) => onChange(event.target.value)}
          >
            {feature.options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        ) : null}
      </div>

      <button
        type="button"
        aria-pressed={isMissing}
        className={isMissing ? 'feature-missing-toggle active' : 'feature-missing-toggle'}
        disabled={!feature.missing_allowed}
        onClick={() => onChange(isMissing ? nextPresentValue(feature, null) : null)}
      >
        <span className="feature-missing-dot" aria-hidden="true" />
        <span>Missing</span>
      </button>
    </div>
  )
}
