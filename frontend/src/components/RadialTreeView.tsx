import { useMemo, useState } from 'react'

import type { TreeLayout, TreePredictionResult } from '../types/api'

type RadialTreeViewProps = {
  trees: TreeLayout[]
  treeResults: TreePredictionResult[]
  hoveredTreeIndex: number | null
  onHoverTree: (treeIndex: number | null) => void
}

const VIEWBOX_SIZE = 920
const CENTER = VIEWBOX_SIZE / 2
const SCALE = VIEWBOX_SIZE * 0.38
const MAX_TREE_LABELS = 12

function contributionColor(value: number, alpha = 1) {
  return value >= 0 ? `rgba(218, 88, 56, ${alpha})` : `rgba(34, 129, 126, ${alpha})`
}

function contributionIntensity(value: number, maxAbsValue: number) {
  if (maxAbsValue === 0) {
    return 0.25
  }
  return 0.2 + (Math.abs(value) / maxAbsValue) * 0.8
}

function sectorPath(start: number, end: number, innerRadius: number, outerRadius: number) {
  const x1 = CENTER + Math.cos(start) * innerRadius
  const y1 = CENTER + Math.sin(start) * innerRadius
  const x2 = CENTER + Math.cos(end) * innerRadius
  const y2 = CENTER + Math.sin(end) * innerRadius
  const x3 = CENTER + Math.cos(end) * outerRadius
  const y3 = CENTER + Math.sin(end) * outerRadius
  const x4 = CENTER + Math.cos(start) * outerRadius
  const y4 = CENTER + Math.sin(start) * outerRadius
  const largeArc = end - start > Math.PI ? 1 : 0

  return `M ${x1} ${y1} A ${innerRadius} ${innerRadius} 0 ${largeArc} 1 ${x2} ${y2} L ${x3} ${y3} A ${outerRadius} ${outerRadius} 0 ${largeArc} 0 ${x4} ${y4} Z`
}

function contributionArc(start: number, end: number, radius: number) {
  const x1 = CENTER + Math.cos(start) * radius
  const y1 = CENTER + Math.sin(start) * radius
  const x2 = CENTER + Math.cos(end) * radius
  const y2 = CENTER + Math.sin(end) * radius
  const largeArc = end - start > Math.PI ? 1 : 0
  return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`
}

function treeLabelIndices(numTrees: number) {
  if (numTrees <= MAX_TREE_LABELS) {
    return Array.from({ length: numTrees }, (_, index) => index)
  }

  const indices = new Set<number>()
  for (let labelIndex = 0; labelIndex < MAX_TREE_LABELS; labelIndex += 1) {
    const treeIndex = Math.round((labelIndex * (numTrees - 1)) / (MAX_TREE_LABELS - 1))
    indices.add(treeIndex)
  }
  return [...indices].sort((left, right) => left - right)
}

export function RadialTreeView({
  trees,
  treeResults,
  hoveredTreeIndex,
  onHoverTree,
}: RadialTreeViewProps) {
  const [tooltip, setTooltip] = useState<{ treeIndex: number; x: number; y: number } | null>(null)
  const treeResultMap = new Map(treeResults.map((result) => [result.tree_index, result]))
  const maxAbsContribution = Math.max(...treeResults.map((result) => Math.abs(result.contribution)), 0)
  const visibleTreeLabels = useMemo(
    () => treeLabelIndices(trees.length).map((treeIndex) => trees[treeIndex]).filter(Boolean),
    [trees],
  )

  return (
    <section className="panel radial-panel">
      <div className="panel-header">
        <h2>Radial Tree Layout</h2>
        <span className="panel-caption">
          Static skeleton geometry with dynamic path and contribution overlays
        </span>
      </div>

      {trees.length ? (
        <div className="radial-shell">
          <svg className="radial-svg" viewBox={`0 0 ${VIEWBOX_SIZE} ${VIEWBOX_SIZE}`} role="img">
            <defs>
              <radialGradient id="centerGlow" cx="50%" cy="50%" r="70%">
                <stop offset="0%" stopColor="rgba(255,255,255,0.8)" />
                <stop offset="100%" stopColor="rgba(255,255,255,0)" />
              </radialGradient>
            </defs>

            <circle cx={CENTER} cy={CENTER} r={CENTER - 24} fill="url(#centerGlow)" />

            {trees.map((tree) => (
              <path
                key={`sector-${tree.tree_index}`}
                d={sectorPath(tree.sector_start_angle, tree.sector_end_angle, SCALE * 0.06, SCALE * 1.02)}
                className={hoveredTreeIndex === tree.tree_index ? 'sector-fill active' : 'sector-fill'}
              />
            ))}

            {trees.map((tree) => {
              const treeResult = treeResultMap.get(tree.tree_index)
              const contribution = treeResult?.contribution ?? 0
              const overlayAlpha = contributionIntensity(contribution, maxAbsContribution) * 0.85
              const isHovered = hoveredTreeIndex === tree.tree_index
              return (
                <path
                  key={`arc-${tree.tree_index}`}
                  d={contributionArc(tree.sector_start_angle + 0.01, tree.sector_end_angle - 0.01, SCALE * 1.08)}
                  stroke={contributionColor(contribution, isHovered ? 1 : overlayAlpha)}
                  strokeWidth={isHovered ? 18 : 14}
                  strokeLinecap="round"
                  fill="none"
                  onMouseMove={(event) => {
                    setTooltip({
                      treeIndex: tree.tree_index,
                      x: event.clientX,
                      y: event.clientY,
                    })
                    onHoverTree(tree.tree_index)
                  }}
                  onMouseLeave={() => {
                    setTooltip(null)
                    onHoverTree(null)
                  }}
                />
              )
            })}

            {trees.map((tree) => {
              const rootNodeId = tree.nodes.find((node) => node.depth === 0)?.node_id
              const pointMap = new Map<number, { x: number; y: number }>()
              tree.nodes.forEach((node) => {
                pointMap.set(node.node_id, {
                  x: CENTER + node.x * SCALE,
                  y: CENTER + node.y * SCALE,
                })
              })
              tree.leaves.forEach((leaf) => {
                pointMap.set(leaf.leaf_id, {
                  x: CENTER + leaf.x * SCALE,
                  y: CENTER + leaf.y * SCALE,
                })
              })

              const treeResult = treeResultMap.get(tree.tree_index)
              const activePathNodeIds = new Set(treeResult?.active_path_node_ids ?? [])
              const highlightedEdges = new Set<string>()
              if (treeResult) {
                const sequence = [...treeResult.active_path_node_ids, treeResult.selected_leaf_id]
                sequence.forEach((currentId, index) => {
                  if (index === 0) {
                    return
                  }
                  highlightedEdges.add(`${sequence[index - 1]}-${currentId}`)
                })
              }

              return (
                <g key={`tree-${tree.tree_index}`}>
                  {tree.edges.map((edge) => {
                    const source =
                      edge.source_id === rootNodeId
                        ? { x: CENTER, y: CENTER }
                        : pointMap.get(edge.source_id)
                    const target = pointMap.get(edge.target_id)
                    if (!source || !target) {
                      return null
                    }

                    const highlighted =
                      hoveredTreeIndex === tree.tree_index || highlightedEdges.has(`${edge.source_id}-${edge.target_id}`)
                    return (
                      <line
                        key={`${edge.source_id}-${edge.target_id}`}
                        x1={source.x}
                        y1={source.y}
                        x2={target.x}
                        y2={target.y}
                        className={highlighted ? 'edge-line active' : 'edge-line'}
                      />
                    )
                  })}

                  {tree.nodes
                    .filter((node) => node.node_id !== rootNodeId)
                    .map((node) => {
                      const highlighted = activePathNodeIds.has(node.node_id)
                      return (
                        <circle
                          key={`node-${tree.tree_index}-${node.node_id}`}
                          cx={CENTER + node.x * SCALE}
                          cy={CENTER + node.y * SCALE}
                          r={3.25}
                          className={highlighted ? 'tree-node active' : 'tree-node'}
                        />
                      )
                    })}

                  {tree.leaves.map((leaf) => {
                    const highlighted =
                      hoveredTreeIndex === tree.tree_index || leaf.leaf_id === treeResult?.selected_leaf_id
                    return (
                      <circle
                        key={`leaf-${tree.tree_index}-${leaf.leaf_id}`}
                        cx={CENTER + leaf.x * SCALE}
                        cy={CENTER + leaf.y * SCALE}
                        r={highlighted ? 5.5 : 3.5}
                        className={highlighted ? 'tree-leaf active' : 'tree-leaf'}
                        fill={highlighted ? contributionColor(treeResult?.contribution ?? 0, 1) : undefined}
                      />
                    )
                  })}
                </g>
              )
            })}

            <circle cx={CENTER} cy={CENTER} r={6.5} className="shared-root-node" />

            {visibleTreeLabels.map((tree) => {
              const midAngle = (tree.sector_start_angle + tree.sector_end_angle) * 0.5
              const labelRadius = SCALE * 1.18
              const labelX = CENTER + Math.cos(midAngle) * labelRadius
              const labelY = CENTER + Math.sin(midAngle) * labelRadius
              return (
                <text
                  key={`label-${tree.tree_index}`}
                  x={labelX}
                  y={labelY}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="tree-index-label"
                >
                  {tree.tree_index}
                </text>
              )
            })}
          </svg>
          {tooltip ? (
            <div
              className="radial-tooltip"
              style={{
                left: tooltip.x + 6,
                top: tooltip.y + 6,
              }}
            >
              {`Tree ${tooltip.treeIndex}`}
            </div>
          ) : null}
        </div>
      ) : (
        <div className="empty-state">Upload a model to compute static radial geometry.</div>
      )}
    </section>
  )
}
