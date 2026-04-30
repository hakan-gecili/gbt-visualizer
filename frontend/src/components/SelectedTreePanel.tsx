import { useEffect, useMemo, useRef, useState } from 'react'

import type { TreeLayout, TreePredictionResult } from '../types/api'
import { buildHighlightedEdgeSet, findRootNodeId, formatNodeCondition } from './treeRenderUtils'

type SelectedTreePanelProps = {
  trees: TreeLayout[]
  treeResults: TreePredictionResult[]
  selectedTreeIndex: number | null
  counterfactualPathEdgeMap: Map<number, Set<string>>
}

type Point = {
  x: number
  y: number
}

const VIEWBOX_WIDTH = 920
const VIEWBOX_HEIGHT = 520
const ROOT_Y = 54
const BOTTOM_PADDING = 64
const SIDE_PADDING = 64
const DEFAULT_ZOOM = 0.9
const DEFAULT_PAN_Y = 0
const DEFAULT_PAN_OFFSET = { x: 0, y: 0 }

function contributionLabel(value: number) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(4)}`
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function buildTopDownLayout(tree: TreeLayout) {
  const rootNodeId = findRootNodeId(tree)
  if (rootNodeId === null) {
    return null
  }

  const nodeMap = new Map(tree.nodes.map((node) => [node.node_id, node]))
  const orderedLeaves = tree.leaves.slice().sort((left, right) => left.leaf_order - right.leaf_order)
  const leafCount = Math.max(orderedLeaves.length, 1)
  const usableWidth = VIEWBOX_WIDTH - SIDE_PADDING * 2
  const depthLevels = Math.max(tree.max_depth, 1)
  const verticalStep = (VIEWBOX_HEIGHT - ROOT_Y - BOTTOM_PADDING) / depthLevels
  const leafSpacing = leafCount === 1 ? 0 : usableWidth / (leafCount - 1)

  const pointMap = new Map<number, Point>()

  orderedLeaves.forEach((leaf, index) => {
    pointMap.set(leaf.leaf_id, {
      x: SIDE_PADDING + index * leafSpacing,
      y: ROOT_Y + leaf.depth * verticalStep,
    })
  })

  const positionNode = (nodeId: number): Point => {
    const existingPoint = pointMap.get(nodeId)
    if (existingPoint) {
      return existingPoint
    }

    const node = nodeMap.get(nodeId)
    if (!node) {
      return { x: VIEWBOX_WIDTH / 2, y: ROOT_Y }
    }

    const leftPoint = nodeMap.has(node.left_child_id)
      ? positionNode(node.left_child_id)
      : pointMap.get(node.left_child_id)
    const rightPoint = nodeMap.has(node.right_child_id)
      ? positionNode(node.right_child_id)
      : pointMap.get(node.right_child_id)

    const point = {
      x: ((leftPoint?.x ?? VIEWBOX_WIDTH / 2) + (rightPoint?.x ?? VIEWBOX_WIDTH / 2)) / 2,
      y: ROOT_Y + node.depth * verticalStep,
    }
    pointMap.set(nodeId, point)
    return point
  }

  positionNode(rootNodeId)

  return {
    rootNodeId,
    pointMap,
  }
}

export function SelectedTreePanel({
  trees,
  treeResults,
  selectedTreeIndex,
  counterfactualPathEdgeMap,
}: SelectedTreePanelProps) {
  const [zoom, setZoom] = useState(DEFAULT_ZOOM)
  const [panY, setPanY] = useState(DEFAULT_PAN_Y)
  const [panOffset, setPanOffset] = useState(DEFAULT_PAN_OFFSET)
  const [isDraggingBackground, setIsDraggingBackground] = useState(false)
  const dragStartMouseRef = useRef<Point | null>(null)
  const dragStartOffsetRef = useRef<Point>(DEFAULT_PAN_OFFSET)
  const selectedTree =
    selectedTreeIndex === null ? null : trees.find((tree) => tree.tree_index === selectedTreeIndex) ?? null
  const treeResult = treeResults.find((item) => item.tree_index === selectedTreeIndex)

  useEffect(() => {
    setZoom(DEFAULT_ZOOM)
    setPanY(DEFAULT_PAN_Y)
    setPanOffset(DEFAULT_PAN_OFFSET)
    setIsDraggingBackground(false)
    dragStartMouseRef.current = null
    dragStartOffsetRef.current = DEFAULT_PAN_OFFSET
  }, [selectedTreeIndex])

  useEffect(() => {
    if (!isDraggingBackground) {
      return
    }

    const handleMouseMove = (event: MouseEvent) => {
      const dragStartMouse = dragStartMouseRef.current
      if (!dragStartMouse) {
        return
      }

      setPanOffset({
        x: dragStartOffsetRef.current.x + (event.clientX - dragStartMouse.x),
        y: dragStartOffsetRef.current.y + (event.clientY - dragStartMouse.y),
      })
    }

    const stopDragging = () => {
      setIsDraggingBackground(false)
      dragStartMouseRef.current = null
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', stopDragging)
    window.addEventListener('blur', stopDragging)

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', stopDragging)
      window.removeEventListener('blur', stopDragging)
    }
  }, [isDraggingBackground])

  const detailPayload = useMemo(() => {
    if (!selectedTree) {
      return null
    }

    const layout = buildTopDownLayout(selectedTree)
    if (!layout) {
      return null
    }

    return {
      ...layout,
      activePathNodeIds: new Set(treeResult?.active_path_node_ids ?? []),
      highlightedEdges: buildHighlightedEdgeSet(treeResult),
      counterfactualEdges: counterfactualPathEdgeMap.get(selectedTree.tree_index) ?? new Set<string>(),
    }
  }, [counterfactualPathEdgeMap, selectedTree, treeResult])

  const handleBackgroundMouseDown = (event: React.MouseEvent<SVGRectElement>) => {
    if (event.button !== 0) {
      return
    }

    event.preventDefault()
    dragStartMouseRef.current = { x: event.clientX, y: event.clientY }
    dragStartOffsetRef.current = panOffset
    setIsDraggingBackground(true)
  }

  const transformX = (VIEWBOX_WIDTH * (1 - zoom)) / 2 + panOffset.x
  const transformY = clamp(panY, -200, 200) + panOffset.y

  return (
    <section className="panel selected-tree-panel">
      <div className="panel-header">
        <div>
          <h2>Selected Tree</h2>
          <span className="panel-caption">Expanded structure synchronized with the active observation</span>
        </div>
        {selectedTree ? (
          <div className="selected-tree-summary">
            <strong>{`Tree ${selectedTree.tree_index}`}</strong>
            {treeResult ? (
              <span>{`Contribution ${contributionLabel(treeResult.contribution)} · Leaf ${treeResult.leaf_value.toFixed(4)}`}</span>
            ) : (
              <span>No active prediction yet</span>
            )}
          </div>
        ) : null}
      </div>

      {!selectedTree || !detailPayload ? (
        <div className="empty-state">Select a tree from the radial layout to inspect it in detail.</div>
      ) : (
        <>
          <div className="selected-tree-workspace">
            <div className="selected-tree-shell">
              <svg
                className={isDraggingBackground ? 'selected-tree-svg dragging' : 'selected-tree-svg'}
                viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
                role="img"
                aria-label={`Detailed view for tree ${selectedTree.tree_index}`}
              >
                <rect
                  x={0}
                  y={0}
                  width={VIEWBOX_WIDTH}
                  height={VIEWBOX_HEIGHT}
                  className="selected-tree-drag-surface"
                  onMouseDown={handleBackgroundMouseDown}
                />
                <g transform={`translate(${transformX} ${transformY}) scale(${zoom})`}>
                  {selectedTree.edges.map((edge) => {
                    const source = detailPayload.pointMap.get(edge.source_id)
                    const target = detailPayload.pointMap.get(edge.target_id)
                    if (!source || !target) {
                      return null
                    }
                    return (
                      <line
                        key={`detail-edge-${edge.source_id}-${edge.target_id}`}
                        x1={source.x}
                        y1={source.y}
                        x2={target.x}
                        y2={target.y}
                        className="detail-edge"
                      />
                    )
                  })}

                  {selectedTree.edges
                    .filter((edge) => detailPayload.counterfactualEdges.has(`${edge.source_id}-${edge.target_id}`))
                    .map((edge) => {
                      const source = detailPayload.pointMap.get(edge.source_id)
                      const target = detailPayload.pointMap.get(edge.target_id)
                      if (!source || !target) {
                        return null
                      }
                      return (
                        <line
                          key={`detail-counterfactual-edge-${edge.source_id}-${edge.target_id}`}
                          x1={source.x}
                          y1={source.y}
                          x2={target.x}
                          y2={target.y}
                          className="detail-edge counterfactual"
                        />
                      )
                    })}

                  {selectedTree.edges
                    .filter((edge) => detailPayload.highlightedEdges.has(`${edge.source_id}-${edge.target_id}`))
                    .map((edge) => {
                      const source = detailPayload.pointMap.get(edge.source_id)
                      const target = detailPayload.pointMap.get(edge.target_id)
                      if (!source || !target) {
                        return null
                      }
                      return (
                        <line
                          key={`detail-active-edge-${edge.source_id}-${edge.target_id}`}
                          x1={source.x}
                          y1={source.y}
                          x2={target.x}
                          y2={target.y}
                          className="detail-edge active"
                        />
                      )
                    })}

                  {selectedTree.nodes.map((node) => {
                    const point = detailPayload.pointMap.get(node.node_id)
                    if (!point) {
                      return null
                    }
                    const isRoot = node.node_id === detailPayload.rootNodeId
                    const isActive = detailPayload.activePathNodeIds.has(node.node_id)
                    return (
                      <g key={`detail-node-${node.node_id}`}>
                        <circle
                          cx={point.x}
                          cy={point.y}
                          r={isRoot ? 13 : 9}
                          className={isActive || isRoot ? 'detail-node active' : 'detail-node'}
                        />
                        <text x={point.x} y={point.y - 18} textAnchor="middle" className="detail-node-label">
                          {node.split_feature}
                        </text>
                        <text x={point.x} y={point.y + 24} textAnchor="middle" className="detail-threshold-label">
                          {formatNodeCondition(node)}
                        </text>
                      </g>
                    )
                  })}

                  {selectedTree.leaves.map((leaf) => {
                    const point = detailPayload.pointMap.get(leaf.leaf_id)
                    if (!point) {
                      return null
                    }
                    const isActive = leaf.leaf_id === treeResult?.selected_leaf_id
                    return (
                      <g key={`detail-leaf-${leaf.leaf_id}`}>
                        <circle
                          cx={point.x}
                          cy={point.y}
                          r={isActive ? 12 : 8}
                          className={isActive ? 'detail-leaf active' : 'detail-leaf'}
                        />
                        <text x={point.x} y={point.y + 28} textAnchor="middle" className="detail-leaf-label">
                          {leaf.leaf_value.toFixed(3)}
                        </text>
                      </g>
                    )
                  })}
                </g>
              </svg>
            </div>

            <label className="selected-tree-side-control" htmlFor="selected-tree-pan-y" aria-label="Vertical shift">
              <input
                id="selected-tree-pan-y"
                type="range"
                min={-200}
                max={200}
                step={1}
                value={panY}
                onChange={(event) => setPanY(Number(event.target.value))}
              />
            </label>
          </div>

          <div className="selected-tree-controls">
            <label className="selected-tree-control compact" htmlFor="selected-tree-zoom">
              <span>Zoom</span>
              <input
                id="selected-tree-zoom"
                type="range"
                min={0.75}
                max={2.25}
                step={0.01}
                value={zoom}
                onChange={(event) => setZoom(Number(event.target.value))}
              />
              <strong>{`${Math.round(zoom * 100)}%`}</strong>
            </label>
          </div>
        </>
      )}
    </section>
  )
}
