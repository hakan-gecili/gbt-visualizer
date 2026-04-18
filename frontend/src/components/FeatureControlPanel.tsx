import type { FeatureMetadata } from '../types/api'
import { FeatureSliderRow } from './FeatureSliderRow'

type FeatureControlPanelProps = {
  featureMetadata: FeatureMetadata[]
  featureVector: Record<string, number>
  onFeatureChange: (featureName: string, value: number) => void
}

export function FeatureControlPanel({
  featureMetadata,
  featureVector,
  onFeatureChange,
}: FeatureControlPanelProps) {
  return (
    <section className="panel control-panel">
      <div className="panel-header">
        <h2>Feature Controls</h2>
        <span className="panel-caption">{featureMetadata.length} numeric inputs</span>
      </div>
      <div className="feature-list">
        {featureMetadata.map((feature) => (
          <FeatureSliderRow
            key={feature.name}
            feature={feature}
            value={featureVector[feature.name] ?? feature.default_value}
            onChange={(value) => onFeatureChange(feature.name, value)}
          />
        ))}
      </div>
    </section>
  )
}
