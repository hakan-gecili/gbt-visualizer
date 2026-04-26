import { useEffect, useMemo, useState } from 'react'

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

function findVariant(exampleGroup: ExampleDatasetGroup, modelFamily: string) {
  return exampleGroup.variants.find((variant) => variant.model_family.toLowerCase() === modelFamily)
}

export function ExamplesPanel({
  examples,
  selectedExample,
  busy,
  onSelectExample,
}: ExamplesPanelProps) {
  const [selectedDatasetName, setSelectedDatasetName] = useState('')
  const [selectedModelFamily, setSelectedModelFamily] = useState('')

  const selectedDataset = useMemo(
    () => examples.find((exampleGroup) => exampleGroup.dataset_name === selectedDatasetName) ?? examples[0] ?? null,
    [examples, selectedDatasetName],
  )
  const selectedVariant = selectedDataset ? findVariant(selectedDataset, selectedModelFamily) ?? selectedDataset.variants[0] : null
  const loadedVariant = examples.flatMap((exampleGroup) => exampleGroup.variants).find((variant) => variant.id === selectedExample)
  const loadedDataset = loadedVariant
    ? examples.find((exampleGroup) => exampleGroup.variants.some((variant) => variant.id === loadedVariant.id))
    : null

  useEffect(() => {
    if (!examples.length) {
      setSelectedDatasetName('')
      setSelectedModelFamily('')
      return
    }

    const nextDataset = examples.find((exampleGroup) => exampleGroup.dataset_name === selectedDatasetName) ?? examples[0]
    const nextVariant =
      nextDataset.variants.find((variant) => variant.model_family.toLowerCase() === selectedModelFamily) ??
      nextDataset.variants[0]

    setSelectedDatasetName(nextDataset.dataset_name)
    setSelectedModelFamily(nextVariant?.model_family.toLowerCase() ?? '')
  }, [examples, selectedDatasetName, selectedModelFamily])

  function handleDatasetChange(datasetName: string) {
    const nextDataset = examples.find((exampleGroup) => exampleGroup.dataset_name === datasetName)
    setSelectedDatasetName(datasetName)
    setSelectedModelFamily(nextDataset?.variants[0]?.model_family.toLowerCase() ?? '')
  }

  function handleLoadExample() {
    if (selectedVariant) {
      onSelectExample(selectedVariant.id)
    }
  }

  return (
    <section className="panel examples-panel">
      <div className="panel-header">
        <h2>Examples</h2>
        <span className="panel-caption">{examples.length} available</span>
      </div>
      <p className="lede examples-copy">
        Select a bundled dataset, then choose the model family to load its model, dataset, and schema.
      </p>
      <div className="examples-loader">
        {examples.length && selectedDataset ? (
          <>
            <label className="examples-loader-field">
              <span>Dataset</span>
              <select
                value={selectedDataset.dataset_name}
                disabled={busy}
                onChange={(event) => handleDatasetChange(event.target.value)}
              >
                {examples.map((exampleGroup) => (
                  <option key={exampleGroup.dataset_name} value={exampleGroup.dataset_name}>
                    {formatDatasetName(exampleGroup.dataset_name)}
                  </option>
                ))}
              </select>
            </label>
            <label className="examples-loader-field">
              <span>Model</span>
              <select
                value={selectedVariant?.model_family.toLowerCase() ?? ''}
                disabled={busy || selectedDataset.variants.length === 0}
                onChange={(event) => setSelectedModelFamily(event.target.value)}
              >
                {selectedDataset.variants.map((variant) => (
                  <option key={variant.id} value={variant.model_family.toLowerCase()}>
                    {formatModelFamily(variant.model_family)}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className={selectedVariant?.id === selectedExample ? 'example-load-button active' : 'example-load-button'}
              disabled={busy || !selectedVariant?.has_dataset}
              onClick={handleLoadExample}
              title={selectedVariant?.path}
            >
              Load Example
            </button>
          </>
        ) : (
          <div className="empty-state">No bundled examples found.</div>
        )}
      </div>
      {loadedVariant && loadedDataset ? (
        <div className="examples-loaded-status">
          {`Loaded: ${formatDatasetName(loadedDataset.dataset_name)} · ${formatModelFamily(loadedVariant.model_family)}`}
        </div>
      ) : null}
    </section>
  )
}
