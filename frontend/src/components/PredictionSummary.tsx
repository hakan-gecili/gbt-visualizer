import type { PredictionSummary as PredictionSummaryType, TreePredictionResult } from '../types/api'

type PredictionSummaryProps = {
  prediction: PredictionSummaryType | null
  treeResults: TreePredictionResult[]
}

const MARGIN_EXPLANATION =
  'Margin is the raw ensemble score before applying sigmoid. Each tree contributes a signed leaf value to this total. Positive values push the prediction toward class 1, negative values push it toward class 0.'
const LABEL_EXPLANATION =
  'Label is the predicted class after converting the final margin into probability and applying the classification threshold used in the app.'
const TREES_EXPLANATION =
  'Trees is the total number of boosting trees in the loaded LightGBM model. The final prediction is formed by summing contributions from all of them.'
export function PredictionSummary({
  prediction,
  treeResults,
}: PredictionSummaryProps) {
  return (
    <section className="panel prediction-panel">
      <div className="panel-header">
        <h2>Prediction Summary</h2>
        <span className="panel-caption">Raw margin first, sigmoid once at the end</span>
      </div>
      {prediction ? (
        <>
          <div className="probability-meter">
            <div className="probability-fill" style={{ width: `${prediction.probability * 100}%` }} />
            <div className="probability-copy">
              <span>Probability</span>
              <strong>{prediction.probability.toFixed(4)}</strong>
            </div>
          </div>
          <div className="stat-grid">
            <div className="info-card" data-tooltip={MARGIN_EXPLANATION} tabIndex={0}>
              <span>Margin</span>
              <strong>{prediction.margin.toFixed(4)}</strong>
            </div>
            <div className="info-card" data-tooltip={LABEL_EXPLANATION} tabIndex={0}>
              <span>Label</span>
              <strong>{prediction.predicted_label}</strong>
            </div>
            <div className="info-card" data-tooltip={TREES_EXPLANATION} tabIndex={0}>
              <span>Trees</span>
              <strong>{treeResults.length}</strong>
            </div>
          </div>
        </>
      ) : (
        <div className="empty-state">Load a model to compute margin and probability.</div>
      )}
    </section>
  )
}
