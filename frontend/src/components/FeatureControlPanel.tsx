import type { FeatureMetadata, FeatureValue } from '../types/api'
import { FeatureControlRow } from './FeatureControlRow'

type FeatureControlPanelProps = {
  featureMetadata: FeatureMetadata[]
  featureVector: Record<string, FeatureValue>
  canGenerateRandomSample: boolean
  onFeatureChange: (featureName: string, value: FeatureValue) => void
  onGenerateRandomSample: () => void
}

export function FeatureControlPanel({
  featureMetadata,
  featureVector,
  canGenerateRandomSample,
  onFeatureChange,
  onGenerateRandomSample,
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
        <div>
          <h2>Feature Controls</h2>
          <span className="panel-caption">
            {`${featureMetadata.length} inputs · ${typeSummary.numeric} numeric · ${typeSummary.binary} binary · ${typeSummary.categorical} categorical`}
          </span>
        </div>
        <button
          type="button"
          className="ghost-button random-sample-button"
          disabled={!canGenerateRandomSample}
          onClick={onGenerateRandomSample}
        >
          Generate Random Sample
        </button>
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
