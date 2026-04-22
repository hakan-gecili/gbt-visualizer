import type { FeatureMetadata, FeatureValue } from '../types/api'
import { FeatureControlRow } from './FeatureControlRow'

type FeatureControlPanelProps = {
  featureMetadata: FeatureMetadata[]
  featureVector: Record<string, FeatureValue>
  onFeatureChange: (featureName: string, value: FeatureValue) => void
}

export function FeatureControlPanel({
  featureMetadata,
  featureVector,
  onFeatureChange,
}: FeatureControlPanelProps) {
  const typeSummary = featureMetadata.reduce(
    (counts, feature) => {
      counts[feature.type] += 1
      return counts
    },
    { numeric: 0, binary: 0, categorical: 0 },
  )

  return (
    <section className="panel control-panel">
      <div className="panel-header">
        <h2>Feature Controls</h2>
        <span className="panel-caption">
          {`${featureMetadata.length} inputs · ${typeSummary.numeric} numeric · ${typeSummary.binary} binary · ${typeSummary.categorical} categorical`}
        </span>
      </div>
      <div className="feature-list">
        {featureMetadata.map((feature) => (
          <FeatureControlRow
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
