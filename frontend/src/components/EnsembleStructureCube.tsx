import { Canvas, type ThreeEvent, useThree } from '@react-three/fiber'
import { Edges, Line, OrbitControls, Text } from '@react-three/drei'
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Color, Object3D, type InstancedMesh } from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

import type { FeatureMetadata, TreeLayout, TreePredictionResult } from '../types/api'
import {
  buildEnsembleCubeData,
  type EnsembleCubeCell,
  type EnsembleCubeLayer,
  type EnsembleCubeTreeRoot,
} from './ensembleCubeUtils'

type EnsembleStructureCubeProps = {
  trees: TreeLayout[]
  featureMetadata: FeatureMetadata[]
  treeResults: TreePredictionResult[]
  selectedTreeIndex: number | null
  onSelectTree: (treeIndex: number | null) => void
}

type TooltipState = {
  cell: CubeRenderCell
  x: number
  y: number
}

type CellInstancesProps = {
  cells: CubeRenderCell[]
  featureCount: number
  maxDepth: number
  opacity: number
  color: Color
  onHover: (cell: CubeRenderCell, x: number, y: number) => void
  onLeave: () => void
  onSelect: (cell: CubeRenderCell) => void
}

type CubeRenderLayer = 'transition' | 'positiveBars' | 'negativeBars'
type VisibleRenderLayers = Record<CubeRenderLayer, boolean>

type CubeRenderCell = EnsembleCubeCell & {
  transitionCount: number
  positiveConnectionCount: number
  negativeConnectionCount: number
  transitionCell: EnsembleCubeCell
  positiveCell: EnsembleCubeCell | null
  negativeCell: EnsembleCubeCell | null
}

type BarInstance = {
  cell: CubeRenderCell
  direction: 1 | -1
  count: number
  maxCount: number
  highlighted: boolean
  observed: boolean
  color: string
}

const CELL_GAP = 0.82
const CELL_SIZE = 0.54
const BAR_GAP = 0.12
const BAR_THICKNESS = 0.16
const MAX_BAR_LENGTH = 1.45
const dummyObject = new Object3D()

const LAYER_LABELS: Record<EnsembleCubeLayer, string> = {
  positive: 'Positive leaf connections',
  transition: 'Transition / internal usage',
  negative: 'Negative leaf connections',
}

const RENDER_LAYER_LABELS: Record<CubeRenderLayer, string> = {
  transition: 'Show transition cubes',
  positiveBars: 'Show positive bars',
  negativeBars: 'Show negative bars',
}

const RENDER_LAYERS: CubeRenderLayer[] = ['transition', 'positiveBars', 'negativeBars']

function cellPosition(cell: Pick<EnsembleCubeCell, 'featureIndex' | 'depth'>, featureCount: number, maxDepth: number) {
  return [
    (cell.featureIndex - (featureCount - 1) / 2) * CELL_GAP,
    ((maxDepth || 1) / 2 - cell.depth) * CELL_GAP,
    0,
  ] as const
}

function rootCellPosition(root: EnsembleCubeTreeRoot, featureCount: number, maxDepth: number) {
  return [
    (root.featureIndex - (featureCount - 1) / 2) * CELL_GAP,
    ((maxDepth || 1) / 2) * CELL_GAP,
    0,
  ] as const
}

function colorForCell(cell: CubeRenderCell, maxCountsByLayer: Record<EnsembleCubeLayer, number>) {
  const maxCount = Math.max(maxCountsByLayer.transition, 1)
  const intensity = Math.sqrt(cell.transitionCount / maxCount)
  return new Color('#dce8ee').lerp(new Color('#236a92'), 0.34 + intensity * 0.66)
}

function opacityBucket(cell: CubeRenderCell, maxCountsByLayer: Record<EnsembleCubeLayer, number>) {
  const maxCount = Math.max(maxCountsByLayer.transition, 1)
  const intensity = Math.sqrt(cell.transitionCount / maxCount)
  if (intensity > 0.78) {
    return 1
  }
  if (intensity > 0.48) {
    return 0.92
  }
  return 0.78
}

function combineUniqueNumbers(...groups: Array<number[] | undefined>) {
  return [...new Set(groups.flatMap((group) => group ?? []))].sort((left, right) => left - right)
}

function combineExamplePaths(...groups: Array<string[] | undefined>) {
  return [...new Set(groups.flatMap((group) => group ?? []))].slice(0, 8)
}

function selectedTreeCount(cell: EnsembleCubeCell | null, selectedTreeIndex: number | null) {
  if (!cell) {
    return 0
  }
  return selectedTreeIndex === null ? cell.count : cell.treeCounts[selectedTreeIndex] ?? 0
}

function renderCellKey(featureIndex: number, depth: number) {
  return `${featureIndex}::${depth}`
}

function buildRenderCells(cells: EnsembleCubeCell[]) {
  const grouped = new Map<string, Partial<Record<EnsembleCubeLayer, EnsembleCubeCell>>>()
  for (const cell of cells) {
    const key = renderCellKey(cell.featureIndex, cell.depth)
    grouped.set(key, {
      ...grouped.get(key),
      [cell.layer]: cell,
    })
  }

  return [...grouped.values()]
    .filter((group): group is Partial<Record<EnsembleCubeLayer, EnsembleCubeCell>> & { transition: EnsembleCubeCell } => Boolean(group.transition))
    .map((group) => {
      const transitionCell = group.transition
      const positiveCell = group.positive ?? null
      const negativeCell = group.negative ?? null
      return {
        ...transitionCell,
        layer: 'transition' as const,
        layerIndex: 1,
        count: transitionCell.count,
        transitionCount: transitionCell.count,
        positiveConnectionCount: positiveCell?.count ?? 0,
        negativeConnectionCount: negativeCell?.count ?? 0,
        treeIndices: combineUniqueNumbers(transitionCell.treeIndices, positiveCell?.treeIndices, negativeCell?.treeIndices),
        nodeIds: combineUniqueNumbers(transitionCell.nodeIds, positiveCell?.nodeIds, negativeCell?.nodeIds),
        positiveLeafCount: positiveCell?.count ?? 0,
        negativeLeafCount: negativeCell?.count ?? 0,
        examplePaths: combineExamplePaths(transitionCell.examplePaths, positiveCell?.examplePaths, negativeCell?.examplePaths),
        transitionCell,
        positiveCell,
        negativeCell,
      }
    })
}

function CellInstances({
  cells,
  featureCount,
  maxDepth,
  opacity,
  color,
  onHover,
  onLeave,
  onSelect,
}: CellInstancesProps) {
  const meshRef = useRef<InstancedMesh>(null)

  useLayoutEffect(() => {
    const mesh = meshRef.current
    if (!mesh) {
      return
    }

    mesh.count = cells.length
    cells.forEach((cell, index) => {
      const [x, y, z] = cellPosition(cell, featureCount, maxDepth)
      dummyObject.position.set(x, y, z)
      dummyObject.scale.setScalar(CELL_SIZE)
      dummyObject.updateMatrix()
      mesh.setMatrixAt(index, dummyObject.matrix)
    })

    mesh.instanceMatrix.needsUpdate = true
  }, [cells, featureCount, maxDepth])

  const handlePointerMove = (event: ThreeEvent<PointerEvent>) => {
    if (event.instanceId === undefined) {
      return
    }
    event.stopPropagation()
    onHover(cells[event.instanceId], event.nativeEvent.clientX, event.nativeEvent.clientY)
  }

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    if (event.instanceId === undefined) {
      return
    }
    event.stopPropagation()
    onSelect(cells[event.instanceId])
  }

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, cells.length]}
      onPointerMove={handlePointerMove}
      onPointerOut={onLeave}
      onClick={handleClick}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} />
    </instancedMesh>
  )
}

function ActiveCellOutlines({
  cells,
  featureCount,
  maxDepth,
}: {
  cells: CubeRenderCell[]
  featureCount: number
  maxDepth: number
}) {
  if (!cells.length) {
    return null
  }

  return (
    <group>
      {cells.map((cell) => {
        const [x, y, z] = cellPosition(cell, featureCount, maxDepth)
        return (
          <mesh
            key={`active-outline-${cell.transitionCell.key}`}
            position={[x, y, z]}
            scale={[CELL_SIZE * 1.04, CELL_SIZE * 1.04, CELL_SIZE * 1.04]}
            raycast={() => null}
          >
            <boxGeometry args={[1, 1, 1]} />
            <meshBasicMaterial transparent opacity={0} depthWrite={false} />
            <Edges color="#050505" lineWidth={1.6} />
          </mesh>
        )
      })}
    </group>
  )
}

function BarInstances({ bars, featureCount, maxDepth }: { bars: BarInstance[]; featureCount: number; maxDepth: number }) {
  const meshRef = useRef<InstancedMesh>(null)

  useLayoutEffect(() => {
    const mesh = meshRef.current
    if (!mesh) {
      return
    }

    mesh.count = bars.length
    bars.forEach((bar, index) => {
      const [x, y, z] = cellPosition(bar.cell, featureCount, maxDepth)
      const normalized = Math.log1p(bar.count) / Math.log1p(Math.max(bar.maxCount, 1))
      const length = 0.26 + normalized * MAX_BAR_LENGTH
      const start = z + bar.direction * (CELL_SIZE / 2 + BAR_GAP)
      const centerZ = start + bar.direction * (length / 2)
      const thickness = BAR_THICKNESS * (bar.observed ? 1.45 : bar.highlighted ? 1.22 : 1)
      dummyObject.position.set(x, y, centerZ)
      dummyObject.scale.set(thickness, thickness, length)
      dummyObject.updateMatrix()
      mesh.setMatrixAt(index, dummyObject.matrix)
    })

    mesh.instanceMatrix.needsUpdate = true
  }, [bars, featureCount, maxDepth])

  if (!bars.length) {
    return null
  }

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, bars.length]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial color={bars[0]?.color ?? '#2f9e62'} transparent opacity={0.86} />
    </instancedMesh>
  )
}

function FeatureDepthGrid({ featureCount, maxDepth }: { featureCount: number; maxDepth: number }) {
  const width = Math.max((featureCount - 1) * CELL_GAP, CELL_GAP)
  const topY = ((maxDepth || 1) / 2) * CELL_GAP
  const bottomY = topY - (maxDepth || 1) * CELL_GAP
  const leftX = -width / 2
  const rightX = width / 2

  return (
    <group>
      {Array.from({ length: featureCount }, (_, index) => {
        const x = (index - (featureCount - 1) / 2) * CELL_GAP
        return (
          <Line
            key={`feature-grid-${index}`}
            points={[[x, topY + CELL_SIZE * 0.46, 0], [x, bottomY - CELL_SIZE * 0.46, 0]]}
            color="#6d777d"
            transparent
            opacity={0.14}
            lineWidth={0.45}
          />
        )
      })}
      {Array.from({ length: (maxDepth || 0) + 1 }, (_, depth) => {
        const y = ((maxDepth || 1) / 2 - depth) * CELL_GAP
        return (
          <Line
            key={`depth-grid-${depth}`}
            points={[[leftX - CELL_SIZE * 0.46, y, 0], [rightX + CELL_SIZE * 0.46, y, 0]]}
            color="#6d777d"
            transparent
            opacity={0.16}
            lineWidth={0.45}
          />
        )
      })}
    </group>
  )
}

function TreeRootContext({
  treeRoots,
  selectedTreeIndex,
  featureCount,
  maxDepth,
}: {
  treeRoots: EnsembleCubeTreeRoot[]
  selectedTreeIndex: number | null
  featureCount: number
  maxDepth: number
}) {
  const meshRef = useRef<InstancedMesh>(null)
  const sortedTreeRoots = useMemo(
    () => treeRoots.slice().sort((left, right) => left.treeIndex - right.treeIndex),
    [treeRoots],
  )
  const visibleTreeRoots = useMemo(
    () =>
      selectedTreeIndex === null
        ? sortedTreeRoots
        : sortedTreeRoots.filter((root) => root.treeIndex === selectedTreeIndex),
    [selectedTreeIndex, sortedTreeRoots],
  )
  const treeRootPositionMap = useMemo(
    () => new Map(sortedTreeRoots.map((root, index) => [root.treeIndex, index])),
    [sortedTreeRoots],
  )
  const topY = ((maxDepth || 1) / 2) * CELL_GAP
  const treeY = topY + CELL_SIZE * 3
  const treeZ = 0
  const width = Math.max((featureCount - 1) * CELL_GAP, CELL_GAP)
  const treeCellX = useCallback((root: EnsembleCubeTreeRoot) => {
    const index = treeRootPositionMap.get(root.treeIndex) ?? 0
    const denominator = Math.max(sortedTreeRoots.length - 1, 1)
    return -width / 2 + (index / denominator) * width
  }, [sortedTreeRoots.length, treeRootPositionMap, width])

  useLayoutEffect(() => {
    const mesh = meshRef.current
    if (!mesh) {
      return
    }

    mesh.count = visibleTreeRoots.length
    visibleTreeRoots.forEach((root, index) => {
      const x = treeCellX(root)
      dummyObject.position.set(x, treeY, treeZ)
      dummyObject.scale.set(CELL_SIZE * 0.42, CELL_SIZE * 0.42, CELL_SIZE * 0.42)
      dummyObject.updateMatrix()
      mesh.setMatrixAt(index, dummyObject.matrix)
    })

    mesh.instanceMatrix.needsUpdate = true
  }, [treeY, treeZ, visibleTreeRoots, treeCellX])

  if (!visibleTreeRoots.length) {
    return null
  }

  return (
    <group>
      <instancedMesh
        key={`tree-cells-${visibleTreeRoots.map((root) => root.treeIndex).join('-')}`}
        ref={meshRef}
        args={[undefined, undefined, visibleTreeRoots.length]}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial color="#39423d" transparent opacity={0.42} />
      </instancedMesh>
      {visibleTreeRoots.map((root) => {
        const treeX = treeCellX(root)
        const [rootX, rootY, rootZ] = rootCellPosition(root, featureCount, maxDepth)
        return (
          <group key={`tree-root-context-${root.treeIndex}`}>
            <Line
              points={[
                [treeX, treeY - CELL_SIZE * 0.28, treeZ],
                [rootX, rootY + CELL_SIZE * 0.36, rootZ],
              ]}
              color="#66746d"
              lineWidth={0.7}
              transparent
              opacity={selectedTreeIndex === null ? 0.22 : 0.52}
            />
            {selectedTreeIndex !== null ? (
              <Text
                position={[treeX, treeY + CELL_SIZE * 0.42, treeZ]}
                fontSize={0.12}
                color="#1b1e1d"
                anchorX="center"
                anchorY="middle"
              >
                {`T${root.treeIndex}`}
              </Text>
            ) : null}
          </group>
        )
      })}
      <Text
        position={[-width / 2 - 0.2, treeY + CELL_SIZE * 0.46, treeZ]}
        fontSize={0.12}
        color="#5f655e"
        anchorX="left"
        anchorY="middle"
      >
        Trees
      </Text>
    </group>
  )
}

function CubeScene({
  cells,
  treeRoots,
  featureNames,
  maxDepth,
  maxCountsByLayer,
  selectedTreeCellKeys,
  observationCellKeys,
  visibleRenderLayers,
  selectedTreeIndex,
  controlsRef,
  onHover,
  onLeave,
  onSelect,
}: {
  cells: EnsembleCubeCell[]
  treeRoots: EnsembleCubeTreeRoot[]
  featureNames: string[]
  maxDepth: number
  maxCountsByLayer: Record<EnsembleCubeLayer, number>
  selectedTreeCellKeys: Set<string>
  observationCellKeys: Set<string>
  visibleRenderLayers: VisibleRenderLayers
  selectedTreeIndex: number | null
  controlsRef: React.RefObject<OrbitControlsImpl | null>
  onHover: (cell: CubeRenderCell, x: number, y: number) => void
  onLeave: () => void
  onSelect: (cell: CubeRenderCell) => void
}) {
  const featureCount = Math.max(featureNames.length, 1)
  const width = Math.max((featureCount - 1) * CELL_GAP + CELL_SIZE, 1)
  const height = Math.max((maxDepth || 1) * CELL_GAP + CELL_SIZE, 1)
  const { camera } = useThree()
  const renderCells = useMemo(() => buildRenderCells(cells), [cells])
  const visibleRenderCells = useMemo(
    () =>
      selectedTreeIndex === null
        ? renderCells
        : renderCells.filter((cell) => cell.transitionCell.treeIndices.includes(selectedTreeIndex)),
    [renderCells, selectedTreeIndex],
  )
  const focusX = useMemo(() => {
    if (!visibleRenderCells.length) {
      return 0
    }
    const occupiedX = visibleRenderCells.map((cell) => cellPosition(cell, featureCount, maxDepth)[0])
    return (Math.min(...occupiedX) + Math.max(...occupiedX)) / 2
  }, [featureCount, maxDepth, visibleRenderCells])
  const groupedCells = useMemo(
    () => {
      if (!visibleRenderLayers.transition) {
        return []
      }
      const groups = new Map<string, { cells: CubeRenderCell[]; color: Color; opacity: number }>()
      for (const cell of visibleRenderCells) {
        const maxCount = Math.max(maxCountsByLayer.transition, 1)
        const intensityBucket = Math.round(Math.sqrt(cell.transitionCount / maxCount) * 5)
        const opacity = opacityBucket(cell, maxCountsByLayer)
        const key = `transition-${intensityBucket}-${opacity}`
        const group = groups.get(key)
        if (group) {
          group.cells.push(cell)
        } else {
          groups.set(key, {
            cells: [cell],
            color: colorForCell(cell, maxCountsByLayer),
            opacity,
          })
        }
      }
      return [...groups.values()]
    },
    [maxCountsByLayer, visibleRenderCells, visibleRenderLayers.transition],
  )
  const activePathCells = useMemo(
    () =>
      visibleRenderLayers.transition
        ? visibleRenderCells.filter((cell) => observationCellKeys.has(cell.transitionCell.key))
        : [],
    [observationCellKeys, visibleRenderCells, visibleRenderLayers.transition],
  )
  const maxVisiblePositiveBarCount = useMemo(
    () =>
      Math.max(
        ...visibleRenderCells.map((cell) => selectedTreeCount(cell.positiveCell, selectedTreeIndex)),
        0,
      ),
    [selectedTreeIndex, visibleRenderCells],
  )
  const maxVisibleNegativeBarCount = useMemo(
    () =>
      Math.max(
        ...visibleRenderCells.map((cell) => selectedTreeCount(cell.negativeCell, selectedTreeIndex)),
        0,
      ),
    [selectedTreeIndex, visibleRenderCells],
  )
  const positiveBars = useMemo(
    () =>
      visibleRenderCells
        .map((cell) => ({
          cell,
          count: selectedTreeCount(cell.positiveCell, selectedTreeIndex),
        }))
        .filter(({ count }) => visibleRenderLayers.positiveBars && count > 0)
        .map(({ cell, count }) => ({
          cell,
          direction: 1 as const,
          count,
          maxCount: selectedTreeIndex === null ? maxCountsByLayer.positive : maxVisiblePositiveBarCount,
          highlighted: Boolean(cell.positiveCell && selectedTreeCellKeys.has(cell.positiveCell.key)),
          observed: Boolean(cell.positiveCell && observationCellKeys.has(cell.positiveCell.key)),
          color: '#2e9f62',
        })),
    [
      maxCountsByLayer.positive,
      maxVisiblePositiveBarCount,
      observationCellKeys,
      selectedTreeCellKeys,
      selectedTreeIndex,
      visibleRenderCells,
      visibleRenderLayers.positiveBars,
    ],
  )
  const negativeBars = useMemo(
    () =>
      visibleRenderCells
        .map((cell) => ({
          cell,
          count: selectedTreeCount(cell.negativeCell, selectedTreeIndex),
        }))
        .filter(({ count }) => visibleRenderLayers.negativeBars && count > 0)
        .map(({ cell, count }) => ({
          cell,
          direction: -1 as const,
          count,
          maxCount: selectedTreeIndex === null ? maxCountsByLayer.negative : maxVisibleNegativeBarCount,
          highlighted: Boolean(cell.negativeCell && selectedTreeCellKeys.has(cell.negativeCell.key)),
          observed: Boolean(cell.negativeCell && observationCellKeys.has(cell.negativeCell.key)),
          color: '#c94b47',
        })),
    [
      maxCountsByLayer.negative,
      maxVisibleNegativeBarCount,
      observationCellKeys,
      selectedTreeCellKeys,
      selectedTreeIndex,
      visibleRenderCells,
      visibleRenderLayers.negativeBars,
    ],
  )

  useEffect(() => {
    camera.lookAt(focusX, 0, 0)
    controlsRef.current?.target.set(focusX, 0, 0)
    controlsRef.current?.update()
  }, [camera, controlsRef, focusX])

  return (
    <>
      <ambientLight intensity={0.82} />
      <directionalLight position={[4, 7, 5]} intensity={1.25} />
      <directionalLight position={[-5, -3, -4]} intensity={0.35} />
      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableRotate
        enableZoom
        enablePan
        enableDamping
        dampingFactor={0.08}
        screenSpacePanning
      />

      <TreeRootContext
        treeRoots={treeRoots}
        selectedTreeIndex={selectedTreeIndex}
        featureCount={featureCount}
        maxDepth={maxDepth}
      />

      <FeatureDepthGrid featureCount={featureCount} maxDepth={maxDepth} />
      <BarInstances bars={positiveBars} featureCount={featureCount} maxDepth={maxDepth} />
      <BarInstances bars={negativeBars} featureCount={featureCount} maxDepth={maxDepth} />

      {groupedCells.map((group) =>
        group.cells.length ? (
          <CellInstances
            key={`cells-${group.color.getHexString()}-${group.opacity}`}
            cells={group.cells}
            featureCount={featureCount}
            maxDepth={maxDepth}
            opacity={group.opacity}
            color={group.color}
            onHover={onHover}
            onLeave={onLeave}
            onSelect={onSelect}
          />
        ) : null,
      )}
      <ActiveCellOutlines cells={activePathCells} featureCount={featureCount} maxDepth={maxDepth} />

      <mesh position={[0, 0, -0.012]}>
        <boxGeometry args={[width + 0.34, height + 0.34, 0.018]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.12} />
      </mesh>

      <Text position={[0, -height / 2 - 0.48, -0.2]} fontSize={0.15} color="#1b1e1d">
        X = Features
      </Text>
      <Text
        position={[-width / 2 - 0.8, 0, -0.2]}
        rotation={[0, 0, Math.PI / 2]}
        fontSize={0.15}
        color="#1b1e1d"
      >
        Y = Depth
      </Text>
      <Text
        position={[-width / 2 - 0.78, height / 2 + 0.36, 0]}
        fontSize={0.13}
        color="#4a4f4a"
        anchorX="left"
        anchorY="middle"
      >
        Transition / internal usage
      </Text>
      <Text
        position={[width / 2 + 0.92, height / 2 + 0.28, 0.72]}
        rotation={[0, -Math.PI / 2, 0]}
        fontSize={0.13}
        color="#268554"
      >
        +Z = Positive leaf direction
      </Text>
      <Text
        position={[width / 2 + 0.92, height / 2 + 0.28, -0.72]}
        rotation={[0, -Math.PI / 2, 0]}
        fontSize={0.13}
        color="#a8403d"
      >
        -Z = Negative leaf direction
      </Text>
    </>
  )
}

function summarizeTrees(treeIndices: number[]) {
  if (!treeIndices.length) {
    return 'None'
  }
  if (treeIndices.length <= 12) {
    return treeIndices.join(', ')
  }
  return `${treeIndices.slice(0, 12).join(', ')} +${treeIndices.length - 12} more`
}

export function EnsembleStructureCube({
  trees,
  featureMetadata,
  treeResults,
  selectedTreeIndex,
  onSelectTree,
}: EnsembleStructureCubeProps) {
  const [visibleRenderLayers, setVisibleRenderLayers] = useState<VisibleRenderLayers>({
    transition: true,
    positiveBars: true,
    negativeBars: true,
  })
  const [tooltip, setTooltip] = useState<TooltipState | null>(null)
  const [selectedCell, setSelectedCell] = useState<CubeRenderCell | null>(null)
  const controlsRef = useRef<OrbitControlsImpl | null>(null)
  const cubeData = useMemo(
    () => buildEnsembleCubeData(trees, featureMetadata, selectedTreeIndex, treeResults),
    [featureMetadata, selectedTreeIndex, treeResults, trees],
  )

  const handleResetView = () => {
    controlsRef.current?.reset()
  }

  const toggleLayer = (layer: CubeRenderLayer) => {
    if (selectedCell && layer === 'transition' && visibleRenderLayers.transition) {
      setSelectedCell(null)
    }
    if (tooltip && layer === 'transition' && visibleRenderLayers.transition) {
      setTooltip(null)
    }
    setVisibleRenderLayers((current) => ({
      ...current,
      [layer]: !current[layer],
    }))
  }

  return (
    <section className="panel ensemble-cube-panel">
      <div className="panel-header ensemble-cube-header">
        <div>
          <h2>Ensemble Structure Cube</h2>
          <span className="panel-caption">Feature/depth transition usage with directional leaf connections</span>
        </div>
        <div className="ensemble-cube-actions">
          <div className="ensemble-cube-layer-toggles" aria-label="Cube visual visibility">
            {RENDER_LAYERS.map((layer) => (
              <button
                key={layer}
                type="button"
                className={visibleRenderLayers[layer] ? 'ensemble-cube-layer-toggle active' : 'ensemble-cube-layer-toggle'}
                onClick={() => toggleLayer(layer)}
                aria-pressed={visibleRenderLayers[layer]}
              >
                {RENDER_LAYER_LABELS[layer]}
              </button>
            ))}
          </div>
          <label className="ensemble-cube-select-label">
            Tree
            <select
              value={selectedTreeIndex ?? ''}
              onChange={(event) => onSelectTree(event.target.value === '' ? null : Number(event.target.value))}
            >
              <option value="">All</option>
              {trees.map((tree) => (
                <option key={tree.tree_index} value={tree.tree_index}>
                  {tree.tree_index}
                </option>
              ))}
            </select>
          </label>
          <button type="button" className="ghost-button ensemble-cube-reset" onClick={handleResetView}>
            Reset view
          </button>
        </div>
      </div>

      {!trees.length ? (
        <div className="empty-state">Load a model to render the structure cube.</div>
      ) : (
        <div className="ensemble-cube-layout">
          <div className="ensemble-cube-stage">
            <Canvas camera={{ position: [0, 5, 13], fov: 58 }} dpr={[1, 1.6]}>
              <CubeScene
                cells={cubeData.cells}
                treeRoots={cubeData.treeRoots}
                featureNames={cubeData.featureNames}
                maxDepth={cubeData.maxDepth}
                maxCountsByLayer={cubeData.maxCountsByLayer}
                selectedTreeCellKeys={cubeData.selectedTreeCellKeys}
                observationCellKeys={cubeData.observationCellKeys}
                visibleRenderLayers={visibleRenderLayers}
                selectedTreeIndex={selectedTreeIndex}
                controlsRef={controlsRef}
                onHover={(cell, x, y) => setTooltip({ cell, x, y })}
                onLeave={() => setTooltip(null)}
                onSelect={setSelectedCell}
              />
            </Canvas>
            {tooltip ? (
              <div className="ensemble-cube-tooltip" style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}>
                <strong>{tooltip.cell.featureName}</strong>
                <span>{`Depth ${tooltip.cell.depth} · ${LAYER_LABELS.transition}`}</span>
                <span>{`Transition count ${tooltip.cell.transitionCount}`}</span>
                <span>{`Positive leaf connections ${tooltip.cell.positiveConnectionCount}`}</span>
                <span>{`Negative leaf connections ${tooltip.cell.negativeConnectionCount}`}</span>
                <span>{`Trees ${summarizeTrees(tooltip.cell.treeIndices)}`}</span>
              </div>
            ) : null}
          </div>

          <aside className="ensemble-cube-details">
            <div className="ensemble-cube-annotations">
              <span>X = Features</span>
              <span>Y = Depth</span>
              <span>+Z = Positive leaf direction</span>
              <span>-Z = Negative leaf direction</span>
              <span>
                Observation highlight =
                {selectedTreeIndex === null ? ' union of active paths across all trees' : ` active path for tree ${selectedTreeIndex}`}
              </span>
              <div className="ensemble-cube-color-legend" aria-label="Transition count color scale">
                <span>Transition count</span>
                <div className="ensemble-cube-color-scale" />
                <div className="ensemble-cube-color-labels">
                  <span>Low</span>
                  <span>High</span>
                </div>
              </div>
              <span className="ensemble-cube-legend-row">
                <i className="ensemble-cube-legend-swatch cube" />
                Cube fill color = transition count
              </span>
              <span className="ensemble-cube-legend-row">
                <i className="ensemble-cube-outline-swatch" />
                Black cube outline = active path cell
              </span>
              <span className="ensemble-cube-legend-row">
                <i className="ensemble-cube-legend-swatch positive" />
                Green bar = positive leaf connection count
              </span>
              <span className="ensemble-cube-legend-row">
                <i className="ensemble-cube-legend-swatch negative" />
                Red bar = negative leaf connection count
              </span>
            </div>

            {selectedCell ? (
              <div className="ensemble-cube-cell-card">
                <div>
                  <span className="panel-caption">Selected cell</span>
                  <h3>{selectedCell.featureName}</h3>
                </div>
                <dl>
                  <div>
                    <dt>Depth</dt>
                    <dd>{selectedCell.depth}</dd>
                  </div>
                  <div>
                    <dt>Encoding</dt>
                    <dd>Transition cube with leaf-direction bars</dd>
                  </div>
                  <div>
                    <dt>Transition</dt>
                    <dd>{selectedCell.transitionCount}</dd>
                  </div>
                  <div>
                    <dt>Positive</dt>
                    <dd>{selectedCell.positiveConnectionCount}</dd>
                  </div>
                  <div>
                    <dt>Negative</dt>
                    <dd>{selectedCell.negativeConnectionCount}</dd>
                  </div>
                  <div>
                    <dt>Trees</dt>
                    <dd>{summarizeTrees(selectedCell.treeIndices)}</dd>
                  </div>
                  <div>
                    <dt>Node ids</dt>
                    <dd>{selectedCell.nodeIds.slice(0, 18).join(', ') || 'Unavailable'}</dd>
                  </div>
                  <div>
                    <dt>Leaf outcomes</dt>
                    <dd>{`+${selectedCell.positiveLeafCount} / -${selectedCell.negativeLeafCount}`}</dd>
                  </div>
                </dl>
              </div>
            ) : (
              <div className="empty-state ensemble-cube-empty-detail">Click a cell to inspect trees, nodes, and outcomes.</div>
            )}
          </aside>
        </div>
      )}
    </section>
  )
}
