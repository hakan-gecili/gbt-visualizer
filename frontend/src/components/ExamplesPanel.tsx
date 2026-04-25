import type { ExampleDatasetGroup } from '../types/api'

type ExamplesPanelProps = {
  examples: ExampleDatasetGroup[]
  selectedExample: string
  busy: boolean
  onSelectExample: (exampleId: string) => void
}

function formatDatasetName(datasetName: string) {
  return datasetName
    .split(/[_-]+/)
    .filter(Boolean)
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(' ')
}

function formatModelFamily(modelFamily: string) {
  if (modelFamily.toLowerCase() === 'lightgbm') {
    return 'LightGBM'
  }
  if (modelFamily.toLowerCase() === 'xgboost') {
    return 'XGBoost'
  }
  return modelFamily
}

export function ExamplesPanel({
  examples,
  selectedExample,
  busy,
  onSelectExample,
}: ExamplesPanelProps) {
  return (
    <section className="panel examples-panel">
      <div className="panel-header">
        <h2>Examples</h2>
        <span className="panel-caption">{examples.length} available</span>
      </div>
      <p className="lede examples-copy">
        Select a bundled dataset, then choose the model family to load its model, dataset, and schema.
      </p>
      <div className="examples-group-list">
        {examples.map((exampleGroup) => (
          <div key={exampleGroup.dataset_name} className="example-group">
            <div className="example-group-title">{formatDatasetName(exampleGroup.dataset_name)}</div>
            <div className="example-variant-row">
              {exampleGroup.variants.map((variant) => (
                <button
                  key={variant.id}
                  type="button"
                  className={selectedExample === variant.id ? 'example-variant-button active' : 'example-variant-button'}
                  disabled={busy || !variant.has_dataset}
                  onClick={() => onSelectExample(variant.id)}
                  title={variant.path}
                >
                  {formatModelFamily(variant.model_family)}
                </button>
              ))}
            </div>
          </div>
        ))}
        {examples.length === 0 ? <div className="empty-state">No bundled examples found.</div> : null}
      </div>
    </section>
  )
}
