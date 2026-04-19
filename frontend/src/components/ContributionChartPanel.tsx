import type { TreePredictionResult } from '../types/api'
import { TreeContributionChart } from './TreeContributionChart'

type ContributionChartPanelProps = {
  treeResults: TreePredictionResult[]
  panelScale: number
  hoveredTreeIndex: number | null
  onHoverTree: (treeIndex: number | null) => void
}

export function ContributionChartPanel({
  treeResults,
  panelScale,
  hoveredTreeIndex,
  onHoverTree,
}: ContributionChartPanelProps) {
  return (
    <div className="contribution-panel-shell">
      <section
        className="panel contribution-panel"
        style={{
          width: `${panelScale * 100}%`,
        }}
      >
        <div className="panel-header">
          <h2>Tree Contributions</h2>
          <span className="panel-caption">Per-tree margin contributions for the active observation</span>
        </div>
        <TreeContributionChart
          treeResults={treeResults}
          hoveredTreeIndex={hoveredTreeIndex}
          onHoverTree={onHoverTree}
        />
      </section>
    </div>
  )
}
