import { useMemo, useState } from 'react'

import type { TreePredictionResult } from '../types/api'

type TreeContributionChartProps = {
  treeResults: TreePredictionResult[]
  hoveredTreeIndex: number | null
  onHoverTree: (treeIndex: number | null) => void
}

type TooltipState = {
  treeIndex: number
  contribution: number
  x: number
  y: number
} | null

const CHART_WIDTH = 920
const CHART_HEIGHT = 280
const MARGIN = {
  top: 18,
  right: 18,
  bottom: 48,
  left: 56,
}
const MAX_X_TICKS = 18

function contributionColor(value: number) {
  return value >= 0 ? '#d95736' : '#22817e'
}

function formatContribution(value: number) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(4)}`
}

function buildTickIndices(numTrees: number) {
  if (numTrees <= MAX_X_TICKS) {
    return Array.from({ length: numTrees }, (_, index) => index)
  }

  const step = Math.ceil(numTrees / MAX_X_TICKS)
  const ticks: number[] = []
  for (let index = 0; index < numTrees; index += step) {
    ticks.push(index)
  }
  if (ticks[ticks.length - 1] !== numTrees - 1) {
    ticks.push(numTrees - 1)
  }
  return ticks
}

export function TreeContributionChart({
  treeResults,
  hoveredTreeIndex,
  onHoverTree,
}: TreeContributionChartProps) {
  const [tooltip, setTooltip] = useState<TooltipState>(null)

  const chartData = useMemo(
    () =>
      treeResults
        .slice()
        .sort((left, right) => left.tree_index - right.tree_index)
        .map((result) => ({
          treeIndex: result.tree_index,
          contribution: result.contribution,
        })),
    [treeResults],
  )

  const maxAbsContribution = Math.max(...chartData.map((item) => Math.abs(item.contribution)), 0)
  const yExtent = maxAbsContribution === 0 ? 1 : maxAbsContribution
  const ticks = buildTickIndices(chartData.length)
  const innerWidth = CHART_WIDTH - MARGIN.left - MARGIN.right
  const innerHeight = CHART_HEIGHT - MARGIN.top - MARGIN.bottom
  const baselineY = MARGIN.top + innerHeight / 2
  const bandWidth = chartData.length > 0 ? innerWidth / chartData.length : innerWidth
  const barWidth = Math.max(2, bandWidth * 0.72)

  function yScale(value: number) {
    const normalized = value / yExtent
    return MARGIN.top + ((1 - normalized) / 2) * innerHeight
  }

  const yTicks = [yExtent, yExtent / 2, 0, -yExtent / 2, -yExtent]

  if (!chartData.length) {
    return <div className="empty-state">Per-tree contributions appear once a prediction is available.</div>
  }

  return (
    <div className="contribution-chart-shell">
      <p className="contribution-chart-copy">
        Raw contribution of trees to the ensemble margin for the current observation. Positive
        contribution increases the final score, negative contribution decreases it.
      </p>
      <div className="contribution-chart-frame">
        <svg
          className="contribution-chart"
          viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
          role="img"
          aria-label="Per-tree raw contribution chart centered on zero"
          onMouseLeave={() => {
            setTooltip(null)
            onHoverTree(null)
          }}
        >
          {yTicks.map((tickValue) => {
            const y = yScale(tickValue)
            const isBaseline = tickValue === 0
            return (
              <g key={tickValue}>
                <line
                  x1={MARGIN.left}
                  y1={y}
                  x2={CHART_WIDTH - MARGIN.right}
                  y2={y}
                  className={isBaseline ? 'chart-baseline' : 'chart-gridline'}
                />
                <text
                  x={MARGIN.left - 10}
                  y={y}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="chart-axis-label"
                >
                  {tickValue.toFixed(3)}
                </text>
              </g>
            )
          })}

          {ticks.map((tickIndex) => {
            const x = MARGIN.left + bandWidth * tickIndex + bandWidth / 2
            return (
              <g key={tickIndex}>
                <line
                  x1={x}
                  y1={CHART_HEIGHT - MARGIN.bottom}
                  x2={x}
                  y2={CHART_HEIGHT - MARGIN.bottom + 6}
                  className="chart-tick-mark"
                />
                <text
                  x={x}
                  y={CHART_HEIGHT - MARGIN.bottom + 18}
                  textAnchor="middle"
                  className="chart-axis-label"
                >
                  {chartData[tickIndex]?.treeIndex ?? tickIndex}
                </text>
              </g>
            )
          })}

          {chartData.map((item, index) => {
            const x = MARGIN.left + bandWidth * index + (bandWidth - barWidth) / 2
            const barTop = item.contribution >= 0 ? yScale(item.contribution) : baselineY
            const barBottom = item.contribution >= 0 ? baselineY : yScale(item.contribution)
            const barHeight = Math.max(1.5, Math.abs(barBottom - barTop))
            const isHovered = hoveredTreeIndex === item.treeIndex

            return (
              <rect
                key={item.treeIndex}
                x={x}
                y={Math.min(barTop, barBottom)}
                width={barWidth}
                height={barHeight}
                rx={Math.min(4, barWidth / 3)}
                fill={contributionColor(item.contribution)}
                opacity={isHovered ? 1 : 0.88}
                className={isHovered ? 'chart-bar active' : 'chart-bar'}
                onMouseMove={(event) => {
                  setTooltip({
                    treeIndex: item.treeIndex,
                    contribution: item.contribution,
                    x: event.clientX,
                    y: event.clientY,
                  })
                  onHoverTree(item.treeIndex)
                }}
                onFocus={() => {
                  onHoverTree(item.treeIndex)
                  setTooltip(null)
                }}
                onBlur={() => onHoverTree(null)}
                tabIndex={0}
              />
            )
          })}
        </svg>

        {tooltip ? (
          <div
            className="chart-tooltip"
            style={{
              left: tooltip.x + 10,
              top: tooltip.y + 10,
            }}
          >
            <div>{`Tree: ${tooltip.treeIndex}`}</div>
            <div>{`Contribution: ${formatContribution(tooltip.contribution)}`}</div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
