import type { FeatureMetadata } from '../types/api'

type FeatureSliderRowProps = {
  feature: FeatureMetadata
  value: number
  onChange: (nextValue: number) => void
}

function resolveBounds(feature: FeatureMetadata, value: number) {
  if (feature.min_value !== null && feature.max_value !== null && feature.min_value !== feature.max_value) {
    return {
      min: feature.min_value,
      max: feature.max_value,
      step: Math.max((feature.max_value - feature.min_value) / 200, 0.01),
    }
  }

  const amplitude = Math.max(Math.abs(value), Math.abs(feature.default_value), 1)
  return {
    min: -amplitude,
    max: amplitude,
    step: Math.max((amplitude * 2) / 200, 0.01),
  }
}

export function FeatureSliderRow({ feature, value, onChange }: FeatureSliderRowProps) {
  const bounds = resolveBounds(feature, value)

  return (
    <div className="feature-row">
      <div className="feature-label">
        <span>{feature.short_name}</span>
        <small>{feature.name}</small>
      </div>
      <input
        className="feature-slider"
        type="range"
        min={bounds.min}
        max={bounds.max}
        step={bounds.step}
        value={Number.isFinite(value) ? value : 0}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <input
        className="feature-number"
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={bounds.step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  )
}
