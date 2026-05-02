import { Canvas, type ThreeEvent, useThree } from '@react-three/fiber'
import { Line, OrbitControls, Text } from '@react-three/drei'
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
}

type TooltipState = {
  cell: EnsembleCubeCell
  x: number
  y: number
}

type CellInstancesProps = {
  cells: EnsembleCubeCell[]
  featureCount: number
  maxDepth: number
  selectedTreeCellKeys: Set<string>
  observationCellKeys: Set<string>
  opacity: number
  color: Color
  onHover: (cell: EnsembleCubeCell, x: number, y: number) => void
  onLeave: () => void
  onSelect: (cell: EnsembleCubeCell) => void
}

type VisibleLayers = Record<EnsembleCubeLayer, boolean>

const CELL_GAP = 0.82
const LAYER_GAP = 1.7
const CELL_SIZE = 0.54
const dummyObject = new Object3D()

const LAYER_LABELS: Record<EnsembleCubeLayer, string> = {
  positive: 'Positive leaf connections',
  transition: 'Transition / internal usage',
  negative: 'Negative leaf connections',
}

const LAYERS: EnsembleCubeLayer[] = ['positive', 'transition', 'negative']

function cellPosition(cell: EnsembleCubeCell, featureCount: number, maxDepth: number) {
  return [
    (cell.featureIndex - (featureCount - 1) / 2) * CELL_GAP,
    ((maxDepth || 1) / 2 - cell.depth) * CELL_GAP,
    cell.layerIndex * LAYER_GAP,
  ] as const
}

function rootCellPosition(root: EnsembleCubeTreeRoot, featureCount: number, maxDepth: number) {
  return [
    (root.featureIndex - (featureCount - 1) / 2) * CELL_GAP,
    ((maxDepth || 1) / 2) * CELL_GAP,
    LAYER_GAP,
  ] as const
}

function layerBaseColor(layer: EnsembleCubeLayer) {
  if (layer === 'positive') {
    return new Color('#1f9d68')
  }
  if (layer === 'negative') {
    return new Color('#2f6fb4')
  }
  return new Color('#d97831')
}

function colorForCell(
  cell: EnsembleCubeCell,
  maxCountsByLayer: Record<EnsembleCubeLayer, number>,
  highlight: 'none' | 'tree' | 'observation',
) {
  const maxCount = Math.max(maxCountsByLayer[cell.layer], 1)
  const intensity = Math.sqrt(cell.count / maxCount)
  const color = new Color('#f4efe5').lerp(layerBaseColor(cell.layer), 0.38 + intensity * 0.62)
  if (highlight === 'observation') {
    return color.lerp(new Color('#f8e04e'), 0.48)
  }
  if (highlight === 'tree') {
    return color.lerp(new Color('#111111'), 0.18)
  }
  return color
}

function opacityBucket(cell: EnsembleCubeCell, maxCountsByLayer: Record<EnsembleCubeLayer, number>) {
  const maxCount = Math.max(maxCountsByLayer[cell.layer], 1)
  const intensity = Math.sqrt(cell.count / maxCount)
  if (intensity > 0.78) {
    return 0.9
  }
  if (intensity > 0.48) {
    return 0.66
  }
  return 0.38
}

function CellInstances({
  cells,
  featureCount,
  maxDepth,
  selectedTreeCellKeys,
  observationCellKeys,
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

    cells.forEach((cell, index) => {
      const [x, y, z] = cellPosition(cell, featureCount, maxDepth)
      const isTreeHighlighted = selectedTreeCellKeys.has(cell.key)
      const isObservationHighlighted = observationCellKeys.has(cell.key)
      const scale = isObservationHighlighted ? 1.18 : isTreeHighlighted ? 1.05 : 1
      dummyObject.position.set(x, y, z)
      dummyObject.scale.setScalar(CELL_SIZE * scale)
      dummyObject.updateMatrix()
      mesh.setMatrixAt(index, dummyObject.matrix)
    })

    mesh.instanceMatrix.needsUpdate = true
  }, [cells, featureCount, maxDepth, observationCellKeys, selectedTreeCellKeys])

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
  const treeZ = LAYER_GAP
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
  visibleLayers,
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
  visibleLayers: VisibleLayers
  selectedTreeIndex: number | null
  controlsRef: React.RefObject<OrbitControlsImpl | null>
  onHover: (cell: EnsembleCubeCell, x: number, y: number) => void
  onLeave: () => void
  onSelect: (cell: EnsembleCubeCell) => void
}) {
  const featureCount = Math.max(featureNames.length, 1)
  const width = Math.max((featureCount - 1) * CELL_GAP + CELL_SIZE, 1)
  const height = Math.max((maxDepth || 1) * CELL_GAP + CELL_SIZE, 1)
  const { camera } = useThree()
  const visibleCells = useMemo(
    () => cells.filter((cell) => visibleLayers[cell.layer]),
    [cells, visibleLayers],
  )
  const focusX = useMemo(() => {
    if (!visibleCells.length) {
      return 0
    }
    const occupiedX = visibleCells.map((cell) => cellPosition(cell, featureCount, maxDepth)[0])
    return (Math.min(...occupiedX) + Math.max(...occupiedX)) / 2
  }, [featureCount, maxDepth, visibleCells])
  const groupedCells = useMemo(
    () => {
      const groups = new Map<string, { cells: EnsembleCubeCell[]; color: Color; opacity: number }>()
      for (const cell of visibleCells) {
        const maxCount = Math.max(maxCountsByLayer[cell.layer], 1)
        const intensityBucket = Math.round(Math.sqrt(cell.count / maxCount) * 5)
        const opacity = opacityBucket(cell, maxCountsByLayer)
        const highlight = observationCellKeys.has(cell.key)
          ? 'observation'
          : selectedTreeCellKeys.has(cell.key)
            ? 'tree'
            : 'none'
        const key = `${cell.layer}-${intensityBucket}-${opacity}-${highlight}`
        const group = groups.get(key)
        if (group) {
          group.cells.push(cell)
        } else {
          groups.set(key, {
            cells: [cell],
            color: colorForCell(cell, maxCountsByLayer, highlight),
            opacity,
          })
        }
      }
      return [...groups.values()]
    },
    [maxCountsByLayer, observationCellKeys, selectedTreeCellKeys, visibleCells],
  )

  useEffect(() => {
    camera.lookAt(focusX, 0, LAYER_GAP)
    controlsRef.current?.target.set(focusX, 0, LAYER_GAP)
    controlsRef.current?.update()
  }, [camera, controlsRef, focusX])

  return (
    <>
      <ambientLight intensity={0.82} />
      <directionalLight position={[4, 7, 5]} intensity={1.25} />
      <directionalLight position={[-5, -3, -4]} intensity={0.35} />
      <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.08} screenSpacePanning />

      <TreeRootContext
        treeRoots={treeRoots}
        selectedTreeIndex={selectedTreeIndex}
        featureCount={featureCount}
        maxDepth={maxDepth}
      />

      {groupedCells.map((group) =>
        group.cells.length ? (
          <CellInstances
            key={`cells-${group.color.getHexString()}-${group.opacity}`}
            cells={group.cells}
            featureCount={featureCount}
            maxDepth={maxDepth}
            selectedTreeCellKeys={selectedTreeCellKeys}
            observationCellKeys={observationCellKeys}
            opacity={group.opacity}
            color={group.color}
            onHover={onHover}
            onLeave={onLeave}
            onSelect={onSelect}
          />
        ) : null,
      )}

      {[0, 1, 2].map((layerIndex) => {
        const layer = Object.entries({ positive: 0, transition: 1, negative: 2 }).find(
          ([, index]) => index === layerIndex,
        )?.[0] as EnsembleCubeLayer
        if (!visibleLayers[layer]) {
          return null
        }
        const z = layerIndex * LAYER_GAP
        return (
          <group key={`plate-${layer}`} position={[0, 0, z]}>
            <mesh>
              <boxGeometry args={[width + 0.34, height + 0.34, 0.018]} />
              <meshBasicMaterial color="#ffffff" transparent opacity={layer === 'transition' ? 0.105 : 0.065} />
            </mesh>
            <Text
              position={[-width / 2 - 0.72, height / 2 + 0.36, 0.04]}
              fontSize={0.13}
              color="#4a4f4a"
              anchorX="left"
              anchorY="middle"
            >
              {LAYER_LABELS[layer]}
            </Text>
          </group>
        )
      })}

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
      <Text position={[width / 2 + 0.95, height / 2 + 0.34, LAYER_GAP]} rotation={[0, -Math.PI / 2, 0]} fontSize={0.14} color="#1b1e1d">
        Z = Outcome / transition layer
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

export function EnsembleStructureCube({ trees, featureMetadata, treeResults }: EnsembleStructureCubeProps) {
  const [selectedTreeIndex, setSelectedTreeIndex] = useState<number | null>(null)
  const [visibleLayers, setVisibleLayers] = useState<VisibleLayers>({
    positive: true,
    transition: true,
    negative: true,
  })
  const [tooltip, setTooltip] = useState<TooltipState | null>(null)
  const [selectedCell, setSelectedCell] = useState<EnsembleCubeCell | null>(null)
  const controlsRef = useRef<OrbitControlsImpl | null>(null)
  const cubeData = useMemo(
    () => buildEnsembleCubeData(trees, featureMetadata, selectedTreeIndex, treeResults),
    [featureMetadata, selectedTreeIndex, treeResults, trees],
  )

  const handleResetView = () => {
    controlsRef.current?.reset()
  }

  const toggleLayer = (layer: EnsembleCubeLayer) => {
    if (selectedCell?.layer === layer && visibleLayers[layer]) {
      setSelectedCell(null)
    }
    if (tooltip?.cell.layer === layer && visibleLayers[layer]) {
      setTooltip(null)
    }
    setVisibleLayers((current) => ({
      ...current,
      [layer]: !current[layer],
    }))
  }

  return (
    <section className="panel ensemble-cube-panel">
      <div className="panel-header ensemble-cube-header">
        <div>
          <h2>Ensemble Structure Cube</h2>
          <span className="panel-caption">Feature/depth/layer density across the forest</span>
        </div>
        <div className="ensemble-cube-actions">
          <div className="ensemble-cube-layer-toggles" aria-label="Cube plate visibility">
            {LAYERS.map((layer) => (
              <button
                key={layer}
                type="button"
                className={visibleLayers[layer] ? 'ensemble-cube-layer-toggle active' : 'ensemble-cube-layer-toggle'}
                onClick={() => toggleLayer(layer)}
                aria-pressed={visibleLayers[layer]}
              >
                {LAYER_LABELS[layer]}
              </button>
            ))}
          </div>
          <label className="ensemble-cube-select-label">
            Tree
            <select
              value={selectedTreeIndex ?? ''}
              onChange={(event) => setSelectedTreeIndex(event.target.value === '' ? null : Number(event.target.value))}
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
                visibleLayers={visibleLayers}
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
                <span>{`Depth ${tooltip.cell.depth} · ${LAYER_LABELS[tooltip.cell.layer]}`}</span>
                <span>{`Count ${tooltip.cell.count}`}</span>
                <span>{`Trees ${summarizeTrees(tooltip.cell.treeIndices)}`}</span>
              </div>
            ) : null}
          </div>

          <aside className="ensemble-cube-details">
            <div className="ensemble-cube-annotations">
              <span>X = Features</span>
              <span>Y = Depth</span>
              <span>Z = Outcome / transition layer</span>
              <span>Middle plate = feature transition/internal node usage</span>
              <span>Positive plate = feature nodes connected to positive leaves</span>
              <span>Negative plate = feature nodes connected to negative leaves</span>
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
                    <dt>Layer</dt>
                    <dd>{LAYER_LABELS[selectedCell.layer]}</dd>
                  </div>
                  <div>
                    <dt>Count</dt>
                    <dd>{selectedCell.count}</dd>
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
                <div className="ensemble-cube-paths">
                  <span className="panel-caption">Example paths</span>
                  {selectedCell.examplePaths.length ? (
                    selectedCell.examplePaths.map((path) => <p key={path}>{path}</p>)
                  ) : (
                    <p>No path examples available for this cell.</p>
                  )}
                </div>
              </div>
            ) : (
              <div className="empty-state ensemble-cube-empty-detail">Click a cell to inspect trees, nodes, outcomes, and example paths.</div>
            )}
          </aside>
        </div>
      )}
    </section>
  )
}
