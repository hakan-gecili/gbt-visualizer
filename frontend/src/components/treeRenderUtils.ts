import type { FeatureMetadata, FeatureValue, TreeLayout, TreeLayoutNode, TreePredictionResult } from '../types/api'

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

function formatCategoryValue(value: number | string) {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (!Number.isNaN(numeric)) {
    return Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(3).replace(/\.?0+$/, '')
  }
  return String(value)
}

function parseCategoryValues(node: TreeLayoutNode) {
  if (node.category_values.length) {
    return node.category_values.map(formatCategoryValue)
  }
  if (typeof node.threshold === 'string') {
    return node.threshold.split('||').filter(Boolean).map(formatCategoryValue)
  }
  return [formatCategoryValue(node.threshold)]
}

export function formatNodeCondition(node: TreeLayoutNode) {
  if (node.decision_type === '==') {
    return `in {${parseCategoryValues(node).join(', ')}}`
  }
  return `${node.decision_type} ${formatThreshold(node.threshold)}`
}

export function formatSplitCondition(node: TreeLayoutNode, branchDirection: 'left' | 'right') {
  if (node.decision_type === '==') {
    const operator = branchDirection === 'left' ? 'in' : 'not in'
    return `${node.split_feature} ${operator} {${parseCategoryValues(node).join(', ')}}`
  }

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

export function buildFeatureVectorPathConditions(
  tree: TreeLayout,
  featureMetadata: FeatureMetadata[],
  featureVector: Record<string, FeatureValue>,
) {
  return buildFeatureVectorPathTraversal(tree, featureMetadata, featureVector).conditions
}

export function buildFeatureVectorPathEdgeSet(
  tree: TreeLayout,
  featureMetadata: FeatureMetadata[],
  featureVector: Record<string, FeatureValue>,
) {
  return new Set(buildFeatureVectorPathTraversal(tree, featureMetadata, featureVector).edgeIds)
}

function buildFeatureVectorPathTraversal(
  tree: TreeLayout,
  featureMetadata: FeatureMetadata[],
  featureVector: Record<string, FeatureValue>,
) {
  const rootNodeId = findRootNodeId(tree)
  if (rootNodeId === null) {
    return { conditions: [], edgeIds: [] }
  }

  const nodeMap = new Map(tree.nodes.map((node) => [node.node_id, node]))
  const leafIds = new Set(tree.leaves.map((leaf) => leaf.leaf_id))
  const featureByName = new Map(featureMetadata.map((feature) => [feature.name, feature]))
  const conditions: string[] = []
  const edgeIds: string[] = []
  let objectId: number | null = rootNodeId

  while (objectId !== null && !leafIds.has(objectId)) {
    const node = nodeMap.get(objectId)
    if (!node) {
      break
    }

    const encodedValue = encodeFeatureValue(featureByName.get(node.split_feature), featureVector[node.split_feature])
    const branchDirection = resolveBranchDirection(node, encodedValue)
    const nextObjectId = branchDirection === 'left' ? node.left_child_id : node.right_child_id
    conditions.push(formatSplitCondition(node, branchDirection))
    edgeIds.push(`${node.node_id}-${nextObjectId}`)
    objectId = nextObjectId
  }

  return { conditions, edgeIds }
}

function encodeFeatureValue(feature: FeatureMetadata | undefined, value: FeatureValue) {
  if (value === null || value === undefined) {
    return null
  }

  if (!feature || feature.type === 'numeric') {
    const numeric = Number(value)
    return Number.isFinite(numeric) ? numeric : null
  }

  const option = feature.options.find((item) => item.value === value || item.label === value)
  if (option) {
    return option.encoded_value
  }

  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

function resolveBranchDirection(node: TreeLayoutNode, encodedValue: number | null): 'left' | 'right' {
  if (encodedValue === null || Number.isNaN(encodedValue)) {
    return 'left'
  }

  if (node.decision_type === '==') {
    const categoryValues = parseCategoryValues(node).map(Number).filter((value) => !Number.isNaN(value))
    return categoryValues.includes(encodedValue) ? 'left' : 'right'
  }

  const threshold = Number(node.threshold)
  if (!Number.isFinite(threshold)) {
    return 'left'
  }

  if (node.decision_type === '<') {
    return encodedValue < threshold ? 'left' : 'right'
  }

  return encodedValue <= threshold ? 'left' : 'right'
}
