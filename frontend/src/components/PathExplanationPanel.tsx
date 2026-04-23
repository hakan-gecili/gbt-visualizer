import { useMemo } from 'react'

import type { TreeLayout, TreePredictionResult } from '../types/api'
import { buildSelectedLeafPathConditions } from './treeRenderUtils'

type PathExplanationPanelProps = {
  trees: TreeLayout[]
  treeResults: TreePredictionResult[]
  selectedTreeIndex: number | null
}

const EMPTY_PATH_MESSAGE = 'Path: select a leaf to view conditions'

export function PathExplanationPanel({
  trees,
  treeResults,
  selectedTreeIndex,
}: PathExplanationPanelProps) {
  const selectedTree =
    selectedTreeIndex === null ? null : trees.find((tree) => tree.tree_index === selectedTreeIndex) ?? null
  const treeResult = treeResults.find((item) => item.tree_index === selectedTreeIndex)

  const pathText = useMemo(() => {
    if (!selectedTree || !treeResult) {
      return EMPTY_PATH_MESSAGE
    }

    const conditions = buildSelectedLeafPathConditions(selectedTree, treeResult)
    return conditions.length ? `Path: ${conditions.join(' ∧ ')}` : EMPTY_PATH_MESSAGE
  }, [selectedTree, treeResult])

  return (
    <section className="panel path-panel">
      <div className="panel-header compact">
        <h2>Path</h2>
      </div>
      <div className="path-panel-content" title={pathText}>
        {pathText}
      </div>
    </section>
  )
}
