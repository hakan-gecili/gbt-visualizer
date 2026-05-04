import type { FeatureMetadata, TreeLayout, TreePredictionResult } from '../types/api'

export type EnsembleCubeLayer = 'positive' | 'transition' | 'negative'

export type EnsembleCubeCell = {
  key: string
  featureName: string
  featureIndex: number
  depth: number
  layer: EnsembleCubeLayer
  layerIndex: number
  count: number
  treeIndices: number[]
  treeCounts: Record<number, number>
  nodeIds: number[]
  positiveLeafCount: number
  negativeLeafCount: number
  examplePaths: string[]
}

export type EnsembleCubeTreeRoot = {
  treeIndex: number
  featureName: string
  featureIndex: number
  nodeId: number
  depth: 0
}

export type EnsembleCubeData = {
  cells: EnsembleCubeCell[]
  featureNames: string[]
  maxDepth: number
  maxCountsByLayer: Record<EnsembleCubeLayer, number>
  selectedTreeCellKeys: Set<string>
  observationCellKeys: Set<string>
  treeRoots: EnsembleCubeTreeRoot[]
}

type MutableCubeCell = Omit<EnsembleCubeCell, 'treeIndices' | 'treeCounts' | 'nodeIds'> & {
  treeIndices: Set<number>
  treeCounts: Map<number, number>
  nodeIds: Set<number>
}

const LAYER_INDEX: Record<EnsembleCubeLayer, number> = {
  positive: 0,
  transition: 1,
  negative: 2,
}

function cellKey(featureName: string, depth: number, layer: EnsembleCubeLayer) {
  return `${featureName}::${depth}::${layer}`
}

function createMutableCell(
  featureName: string,
  featureIndex: number,
  depth: number,
  layer: EnsembleCubeLayer,
): MutableCubeCell {
  return {
    key: cellKey(featureName, depth, layer),
    featureName,
    featureIndex,
    depth,
    layer,
    layerIndex: LAYER_INDEX[layer],
    count: 0,
    treeIndices: new Set<number>(),
    treeCounts: new Map<number, number>(),
    nodeIds: new Set<number>(),
    positiveLeafCount: 0,
    negativeLeafCount: 0,
    examplePaths: [],
  }
}

function normalizeFeatureNames(featureMetadata: FeatureMetadata[], trees: TreeLayout[]) {
  const usedFeatureNames = new Set(trees.flatMap((tree) => tree.nodes.map((node) => node.split_feature)))
  const metadataNames = featureMetadata.map((feature) => feature.name)
  if (metadataNames.length) {
    const unseenUsedNames = [...usedFeatureNames]
      .filter((featureName) => !metadataNames.includes(featureName))
      .sort((left, right) => left.localeCompare(right))
    return [...metadataNames, ...unseenUsedNames]
  }

  return [...usedFeatureNames].sort((left, right) => left.localeCompare(right))
}

function leafLayer(leafValue: number): EnsembleCubeLayer {
  return leafValue >= 0 ? 'positive' : 'negative'
}

function addCellContribution(
  cellMap: Map<string, MutableCubeCell>,
  featureIndexMap: Map<string, number>,
  featureName: string,
  depth: number,
  layer: EnsembleCubeLayer,
  treeIndex: number,
  nodeId: number,
  examplePath?: string,
) {
  const featureIndex = featureIndexMap.get(featureName)
  if (featureIndex === undefined) {
    return
  }

  const key = cellKey(featureName, depth, layer)
  const cell = cellMap.get(key) ?? createMutableCell(featureName, featureIndex, depth, layer)
  cell.count += 1
  cell.treeIndices.add(treeIndex)
  cell.treeCounts.set(treeIndex, (cell.treeCounts.get(treeIndex) ?? 0) + 1)
  cell.nodeIds.add(nodeId)
  if (layer === 'positive') {
    cell.positiveLeafCount += 1
  }
  if (layer === 'negative') {
    cell.negativeLeafCount += 1
  }
  if (examplePath && cell.examplePaths.length < 8) {
    cell.examplePaths.push(examplePath)
  }
  cellMap.set(key, cell)
}

function getParentNodeForLeaf(tree: TreeLayout, leafId: number) {
  return tree.nodes.find((node) => node.left_child_id === leafId || node.right_child_id === leafId)
}

function getRootNode(tree: TreeLayout) {
  const childIds = new Set(tree.edges.map((edge) => edge.target_id))
  return tree.nodes.find((node) => !childIds.has(node.node_id)) ?? tree.nodes.find((node) => node.depth === 0) ?? null
}

export function buildEnsembleCubeData(
  trees: TreeLayout[],
  featureMetadata: FeatureMetadata[],
  selectedTreeIndex: number | null,
  treeResults: TreePredictionResult[],
): EnsembleCubeData {
  const featureNames = normalizeFeatureNames(featureMetadata, trees)
  const featureIndexMap = new Map(featureNames.map((featureName, index) => [featureName, index]))
  const cellMap = new Map<string, MutableCubeCell>()
  const selectedTreeCellKeys = new Set<string>()
  const observationCellKeys = new Set<string>()
  const treeRoots: EnsembleCubeTreeRoot[] = []

  for (const tree of trees) {
    const leafMap = new Map(tree.leaves.map((leaf) => [leaf.leaf_id, leaf]))
    const rootNode = getRootNode(tree)
    const rootFeatureIndex = rootNode ? featureIndexMap.get(rootNode.split_feature) : undefined
    if (rootNode && rootFeatureIndex !== undefined) {
      treeRoots.push({
        treeIndex: tree.tree_index,
        featureName: rootNode.split_feature,
        featureIndex: rootFeatureIndex,
        nodeId: rootNode.node_id,
        depth: 0,
      })
    }

    for (const node of tree.nodes) {
      addCellContribution(
        cellMap,
        featureIndexMap,
        node.split_feature,
        node.depth,
        'transition',
        tree.tree_index,
        node.node_id,
        `Tree ${tree.tree_index}: node ${node.node_id} uses ${node.split_feature} at depth ${node.depth}`,
      )

      for (const childId of [node.left_child_id, node.right_child_id]) {
        const leaf = leafMap.get(childId)
        if (!leaf) {
          continue
        }
        const layer = leafLayer(leaf.leaf_value)
        addCellContribution(
          cellMap,
          featureIndexMap,
          node.split_feature,
          node.depth,
          layer,
          tree.tree_index,
          node.node_id,
          `Tree ${tree.tree_index}: node ${node.node_id} connects to ${layer} leaf ${leaf.leaf_id} (${leaf.leaf_value.toFixed(4)})`,
        )
      }
    }
  }

  for (const cell of cellMap.values()) {
    if (selectedTreeIndex !== null && cell.treeIndices.has(selectedTreeIndex)) {
      selectedTreeCellKeys.add(cell.key)
    }
  }

  const treeMap = new Map(trees.map((tree) => [tree.tree_index, tree]))
  const observedTreeResults =
    selectedTreeIndex === null ? treeResults : treeResults.filter((result) => result.tree_index === selectedTreeIndex)
  for (const result of observedTreeResults) {
    const tree = treeMap.get(result.tree_index)
    if (!tree) {
      continue
    }
    const nodeMap = new Map(tree.nodes.map((node) => [node.node_id, node]))
    for (const nodeId of result.active_path_node_ids) {
      const node = nodeMap.get(nodeId)
      if (node) {
        observationCellKeys.add(cellKey(node.split_feature, node.depth, 'transition'))
      }
    }

    const parentNode = getParentNodeForLeaf(tree, result.selected_leaf_id)
    if (parentNode) {
      observationCellKeys.add(cellKey(parentNode.split_feature, parentNode.depth, leafLayer(result.leaf_value)))
    }
  }

  const cells = [...cellMap.values()]
    .map((cell) => ({
      ...cell,
      treeIndices: [...cell.treeIndices].sort((left, right) => left - right),
      treeCounts: Object.fromEntries([...cell.treeCounts].sort(([left], [right]) => left - right)),
      nodeIds: [...cell.nodeIds].sort((left, right) => left - right),
    }))
    .sort((left, right) => {
      if (left.layerIndex !== right.layerIndex) {
        return left.layerIndex - right.layerIndex
      }
      if (left.depth !== right.depth) {
        return left.depth - right.depth
      }
      return left.featureIndex - right.featureIndex
    })

  const maxCountsByLayer = cells.reduce<Record<EnsembleCubeLayer, number>>(
    (accumulator, cell) => ({
      ...accumulator,
      [cell.layer]: Math.max(accumulator[cell.layer], cell.count),
    }),
    { positive: 0, transition: 0, negative: 0 },
  )

  return {
    cells,
    featureNames,
    maxDepth: Math.max(...trees.map((tree) => tree.max_depth), 0),
    maxCountsByLayer,
    selectedTreeCellKeys,
    observationCellKeys,
    treeRoots,
  }
}
