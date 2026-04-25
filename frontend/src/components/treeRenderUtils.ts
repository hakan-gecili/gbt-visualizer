import type { TreeLayout, TreeLayoutNode, TreePredictionResult } from '../types/api'

export type Point = {
  x: number
  y: number
}

export function buildHighlightedEdgeSet(treeResult: TreePredictionResult | undefined) {
  const highlightedEdges = new Set<string>()
  if (!treeResult) {
    return highlightedEdges
  }

  const sequence = [...treeResult.active_path_node_ids, treeResult.selected_leaf_id]
  sequence.forEach((currentId, index) => {
    if (index === 0) {
      return
    }
    highlightedEdges.add(`${sequence[index - 1]}-${currentId}`)
  })
  return highlightedEdges
}

export function buildTreePointMap(tree: TreeLayout) {
  const pointMap = new Map<number, Point>()
  tree.nodes.forEach((node) => {
    pointMap.set(node.node_id, { x: node.x, y: node.y })
  })
  tree.leaves.forEach((leaf) => {
    pointMap.set(leaf.leaf_id, { x: leaf.x, y: leaf.y })
  })
  return pointMap
}

export function findRootNodeId(tree: TreeLayout) {
  return tree.nodes.find((node) => node.depth === 0)?.node_id ?? null
}

export function formatThreshold(threshold: number | string) {
  if (typeof threshold === 'number') {
    if (Number.isInteger(threshold)) {
      return String(threshold)
    }
    return threshold.toFixed(3).replace(/\.?0+$/, '')
  }

  const normalized = threshold.trim()
  const numeric = Number(normalized)
  if (!normalized || Number.isNaN(numeric)) {
    return threshold
  }
  if (Number.isInteger(numeric)) {
    return String(numeric)
  }
  return numeric.toFixed(3).replace(/\.?0+$/, '')
}

export function formatSplitCondition(node: TreeLayoutNode, branchDirection: 'left' | 'right') {
  const operator = node.decision_type === '<' ? (branchDirection === 'left' ? '<' : '≥') : branchDirection === 'left' ? '≤' : '>'
  return `${node.split_feature} ${operator} ${formatThreshold(node.threshold)}`
}

export function buildSelectedLeafPathConditions(tree: TreeLayout, treeResult: TreePredictionResult | undefined) {
  if (!treeResult) {
    return []
  }

  const nodeMap = new Map(tree.nodes.map((node) => [node.node_id, node]))
  const traversalSequence = [...treeResult.active_path_node_ids, treeResult.selected_leaf_id]

  return treeResult.active_path_node_ids.flatMap((nodeId, index) => {
    const node = nodeMap.get(nodeId)
    const nextId = traversalSequence[index + 1]
    if (!node || nextId === undefined) {
      return []
    }

    const branchDirection =
      nextId === node.left_child_id ? 'left' : nextId === node.right_child_id ? 'right' : null
    if (!branchDirection) {
      return []
    }

    return [formatSplitCondition(node, branchDirection)]
  })
}
