import { useMemo, useState } from 'react'

import type { TreeLayout, TreePredictionResult } from '../types/api'
import { radialTheme } from './radialTheme'
import { buildHighlightedEdgeSet, buildTreePointMap, findRootNodeId } from './treeRenderUtils'

type RadialTreeViewProps = {
  trees: TreeLayout[]
  treeResults: TreePredictionResult[]
  counterfactualPathEdgeMap: Map<number, Set<string>>
  panelScale: number
  onPanelScaleChange: (value: number) => void
  hoveredTreeIndex: number | null
  onHoverTree: (treeIndex: number | null) => void
  selectedTreeIndex: number | null
  onSelectTree: (treeIndex: number) => void
  isDarkMode: boolean
  onToggleDarkMode: () => void
}

const VIEWBOX_SIZE = 920
const CENTER = VIEWBOX_SIZE / 2
const SCALE = VIEWBOX_SIZE * 0.38
const MAX_TREE_LABELS = 12
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
  counterfactualPathEdgeMap,
  panelScale,
  onPanelScaleChange,
  hoveredTreeIndex,
  onHoverTree,
  selectedTreeIndex,
  onSelectTree,
  isDarkMode,
  onToggleDarkMode,
}: RadialTreeViewProps) {
  const [tooltip, setTooltip] = useState<{ treeIndex: number; x: number; y: number } | null>(null)
  const treeResultMap = new Map(treeResults.map((result) => [result.tree_index, result]))
  const maxAbsContribution = Math.max(...treeResults.map((result) => Math.abs(result.contribution)), 0)
  const visibleTreeLabels = useMemo(
    () => treeLabelIndices(trees.length).map((treeIndex) => trees[treeIndex]).filter(Boolean),
    [trees],
  )
  const theme = radialTheme(isDarkMode)

  return (
    <div className="radial-panel-shell">
      <section
        className="panel radial-panel"
        style={{
          width: `${panelScale * 100}%`,
        }}
      >
        <div className="panel-header">
          <h2>Radial Tree Layout</h2>
          <span className="panel-caption">
            Static skeleton geometry with dynamic path and contribution overlays
          </span>
        </div>

        {trees.length ? (
          <div className={isDarkMode ? 'radial-shell dark' : 'radial-shell'}>
            <button
              type="button"
              className="radial-mode-toggle"
              onClick={onToggleDarkMode}
            >
              {isDarkMode ? 'Light' : 'Dark'}
            </button>

            <div className="radial-svg-stage">
              <svg className="radial-svg" viewBox={`0 0 ${VIEWBOX_SIZE} ${VIEWBOX_SIZE}`} role="img">
              <defs>
                <radialGradient id="centerGlow" cx="50%" cy="50%" r="70%">
                  <stop offset="0%" stopColor={theme.gradientInner} />
                  <stop offset="100%" stopColor={theme.gradientOuter} />
                </radialGradient>
              </defs>

              <circle cx={CENTER} cy={CENTER} r={CENTER - 24} fill="url(#centerGlow)" />

              {trees.map((tree) => (
                <path
                  key={`sector-${tree.tree_index}`}
                  d={sectorPath(tree.sector_start_angle, tree.sector_end_angle, SCALE * 0.06, SCALE * 1.02)}
                  className={
                    hoveredTreeIndex === tree.tree_index
                      ? 'sector-fill active'
                      : 'sector-fill'
                  }
                  style={{
                    fill:
                      hoveredTreeIndex === tree.tree_index
                        ? theme.sectorActiveFill
                        : theme.sectorFill,
                    stroke:
                      hoveredTreeIndex === tree.tree_index
                        ? theme.sectorActiveStroke
                        : theme.sectorStroke,
                    strokeWidth: hoveredTreeIndex === tree.tree_index ? 2 : 1.2,
                  }}
                  onClick={() => onSelectTree(tree.tree_index)}
                />
              ))}

              {selectedTreeIndex !== null
                ? trees
                    .filter((tree) => tree.tree_index === selectedTreeIndex)
                    .map((tree) => {
                      const midAngle = (tree.sector_start_angle + tree.sector_end_angle) * 0.5
                      const starRadius = SCALE * 1.14
                      const starX = CENTER + Math.cos(midAngle) * starRadius
                      const starY = CENTER + Math.sin(midAngle) * starRadius
                      return (
                        <text
                          key={`selected-star-${tree.tree_index}`}
                          x={starX}
                          y={starY}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          className="selected-tree-star"
                          fill={theme.label}
                          onClick={() => onSelectTree(tree.tree_index)}
                        >
                          ★
                        </text>
                      )
                    })
                : null}

              {trees.map((tree) => {
              const treeResult = treeResultMap.get(tree.tree_index)
              const contribution = treeResult?.contribution ?? 0
              const overlayAlpha = contributionIntensity(contribution, maxAbsContribution) * 0.85
              const isHovered = hoveredTreeIndex === tree.tree_index
              return (
                <path
                  key={`arc-${tree.tree_index}`}
                  d={contributionArc(tree.sector_start_angle + 0.01, tree.sector_end_angle - 0.01, SCALE * 1.08)}
                  stroke={
                    contribution >= 0
                      ? `rgba(${isDarkMode ? '255, 77, 77' : '218, 88, 56'}, ${isHovered ? 1 : overlayAlpha})`
                      : `rgba(${isDarkMode ? '94, 160, 255' : '34, 129, 126'}, ${isHovered ? 1 : overlayAlpha})`
                  }
                  strokeWidth={isHovered ? 18 : 14}
                  strokeLinecap="round"
                  fill="none"
                  style={{
                    cursor: 'pointer',
                  }}
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
                  onClick={() => onSelectTree(tree.tree_index)}
                />
              )
              })}

              {trees.map((tree) => {
              const rootNodeId = findRootNodeId(tree)
              const pointMap = buildTreePointMap(tree)

              const treeResult = treeResultMap.get(tree.tree_index)
              const activePathNodeIds = new Set(treeResult?.active_path_node_ids ?? [])
              const highlightedEdges = buildHighlightedEdgeSet(treeResult)
              const counterfactualEdges = counterfactualPathEdgeMap.get(tree.tree_index) ?? new Set<string>()

              const renderEdgeLine = (
                edge: TreeLayout['edges'][number],
                className: string,
                stroke: string,
                strokeWidth: number,
                keyPrefix: string,
              ) => {
                const source =
                  edge.source_id === rootNodeId
                    ? { x: CENTER, y: CENTER }
                    : pointMap.get(edge.source_id)
                const target = pointMap.get(edge.target_id)
                if (!source || !target) {
                  return null
                }
                const scaledSource = edge.source_id === rootNodeId
                  ? source
                  : { x: CENTER + source.x * SCALE, y: CENTER + source.y * SCALE }
                const scaledTarget = { x: CENTER + target.x * SCALE, y: CENTER + target.y * SCALE }

                return (
                  <line
                    key={`${keyPrefix}-${edge.source_id}-${edge.target_id}`}
                    x1={scaledSource.x}
                    y1={scaledSource.y}
                    x2={scaledTarget.x}
                    y2={scaledTarget.y}
                    className={className}
                    style={{ stroke, strokeWidth }}
                  />
                )
              }

              return (
                <g key={`tree-${tree.tree_index}`}>
                  {tree.edges.map((edge) => renderEdgeLine(edge, 'edge-line', theme.edge, 1.4, 'edge'))}
                  {tree.edges
                    .filter((edge) => counterfactualEdges.has(`${edge.source_id}-${edge.target_id}`))
                    .map((edge) => renderEdgeLine(edge, 'edge-line counterfactual', '#FF00FF', 1.9, 'cf-edge'))}
                  {tree.edges
                    .filter((edge) => highlightedEdges.has(`${edge.source_id}-${edge.target_id}`))
                    .map((edge) => renderEdgeLine(edge, 'edge-line active', theme.edgeActive, !isDarkMode ? 2.1 : 1.9, 'active-edge'))}

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
                          style={{ fill: highlighted ? theme.nodeActive : theme.node }}
                        />
                      )
                    })}

                  {tree.leaves.map((leaf) => {
                    const highlighted = leaf.leaf_id === treeResult?.selected_leaf_id
                    return (
                      <circle
                        key={`leaf-${tree.tree_index}-${leaf.leaf_id}`}
                        cx={CENTER + leaf.x * SCALE}
                        cy={CENTER + leaf.y * SCALE}
                        r={highlighted ? 5.5 : 3.5}
                        className={highlighted ? 'tree-leaf active' : 'tree-leaf'}
                        fill={
                          highlighted
                            ? isDarkMode
                              ? theme.nodeActive
                              : treeResult && treeResult.contribution < 0
                                ? theme.contributionNegative
                                : theme.contributionPositive
                            : theme.leaf
                        }
                      />
                    )
                  })}
                </g>
              )
              })}

              <circle
                cx={CENTER}
                cy={CENTER}
                r={6.5}
                className="shared-root-node"
                style={{ fill: theme.root, stroke: theme.rootStroke }}
              />

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
                  fill={theme.label}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onSelectTree(tree.tree_index)}
                >
                  {tree.tree_index}
                </text>
              )
              })}
              </svg>
            </div>
            {tooltip ? (
            <div
              className="radial-tooltip"
              style={{
                left: tooltip.x,
                top: tooltip.y,
                background: theme.tooltip,
                color: theme.tooltipText,
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

      <div
        className="radial-scale-control"
        style={{
          width: `${panelScale * 100}%`,
        }}
      >
        <label className="radial-scale-label" htmlFor="radial-scale-slider">
          Panel Size
        </label>
        <input
          id="radial-scale-slider"
          className="radial-scale-slider"
          type="range"
          min={1 / 3}
          max={1}
          step={0.01}
          value={panelScale}
          onChange={(event) => onPanelScaleChange(Number(event.target.value))}
        />
        <span className="radial-scale-value">{`${Math.round(panelScale * 100)}%`}</span>
      </div>
    </div>
  )
}
