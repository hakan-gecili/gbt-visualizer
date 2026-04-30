import { useMemo } from 'react'

import type { CounterfactualResponse, FeatureMetadata, FeatureValue, TreeLayout, TreePredictionResult } from '../types/api'
import { buildFeatureVectorPathConditions, buildSelectedLeafPathConditions } from './treeRenderUtils'

type PathExplanationPanelProps = {
  trees: TreeLayout[]
  treeResults: TreePredictionResult[]
  selectedTreeIndex: number | null
  featureMetadata: FeatureMetadata[]
  featureVector: Record<string, FeatureValue>
  counterfactualResult: CounterfactualResponse | null
}

const EMPTY_PATH_MESSAGE = 'Select a leaf to view conditions'
const EMPTY_COUNTERFACTUAL_PATH_MESSAGE = 'Generate a counterfactual to view its path.'
const NO_COUNTERFACTUAL_PATH_MESSAGE = 'No counterfactual was found within the configured search limit.'

export function PathExplanationPanel({
  trees,
  treeResults,
  selectedTreeIndex,
  featureMetadata,
  featureVector,
  counterfactualResult,
}: PathExplanationPanelProps) {
  const selectedTree =
    selectedTreeIndex === null ? null : trees.find((tree) => tree.tree_index === selectedTreeIndex) ?? null
  const treeResult = treeResults.find((item) => item.tree_index === selectedTreeIndex)

  const currentConditions = useMemo(() => {
    if (!selectedTree || !treeResult) {
      return []
    }

    return buildSelectedLeafPathConditions(selectedTree, treeResult)
  }, [selectedTree, treeResult])

  const counterfactualConditions = useMemo(() => {
    if (!selectedTree || !counterfactualResult?.counterfactuals.length) {
      return []
    }

    const counterfactualVector = { ...featureVector }
    for (const change of counterfactualResult.counterfactuals[0].changes) {
      counterfactualVector[change.feature] = change.new_value
    }
    return buildFeatureVectorPathConditions(selectedTree, featureMetadata, counterfactualVector)
  }, [counterfactualResult, featureMetadata, featureVector, selectedTree])

  const currentTitle = currentConditions.length ? currentConditions.join(' ∧ ') : EMPTY_PATH_MESSAGE
  const counterfactualTitle = counterfactualConditions.length
    ? counterfactualConditions.join(' ∧ ')
    : counterfactualResult && !counterfactualResult.counterfactuals.length
      ? NO_COUNTERFACTUAL_PATH_MESSAGE
      : EMPTY_COUNTERFACTUAL_PATH_MESSAGE

  return (
    <section className="panel path-panel">
      <div className="panel-header compact">
        <h2>Path</h2>
      </div>
      <div className="path-section">
        <span className="path-section-label">Current Path</span>
        <div className="path-panel-content" title={currentTitle}>
          {currentConditions.length ? renderPathConditions(currentConditions) : EMPTY_PATH_MESSAGE}
        </div>
      </div>
      <div className="path-section">
        <span className="path-section-label">Counterfactual Path</span>
        <div className="path-panel-content" title={counterfactualTitle}>
          {counterfactualConditions.length
            ? renderPathConditions(counterfactualConditions, currentConditions)
            : counterfactualTitle}
        </div>
      </div>
    </section>
  )
}

function renderPathConditions(conditions: string[], comparisonConditions?: string[]) {
  return conditions.map((condition, index) => {
    const isChanged = comparisonConditions !== undefined && condition !== comparisonConditions[index]
    const content = isChanged ? <strong>{condition}</strong> : condition
    return (
      <span key={`${condition}-${index}`} className="path-condition">
        {index > 0 ? <span className="path-separator"> ∧ </span> : null}
        {content}
      </span>
    )
  })
}
