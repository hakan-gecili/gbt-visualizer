import { useRef, useState } from 'react'

import type { KeyboardEvent, PointerEvent } from 'react'
import type { PredictionSummary as PredictionSummaryType, TreePredictionResult } from '../types/api'

type PredictionSummaryProps = {
  prediction: PredictionSummaryType | null
  treeResults: TreePredictionResult[]
  threshold: number
  onThresholdChange: (threshold: number) => void
}

const MARGIN_EXPLANATION =
  'Margin is the raw ensemble score before applying sigmoid. Each tree contributes a signed leaf value to this total. Positive values push the prediction toward class 1, negative values push it toward class 0.'
const LABEL_EXPLANATION =
  'Label is the predicted class after converting the final margin into probability and applying the classification threshold used in the app.'
const TREES_EXPLANATION =
  'Trees is the total number of boosting trees in the loaded LightGBM model. The final prediction is formed by summing contributions from all of them.'
const DISTANCE_TO_FLIP_EXPLANATION =
  'Distance to Flip is the absolute probability change needed to cross the decision threshold. Positive means the probability must increase to flip; negative means it must decrease.'

function formatDecisionThreshold(threshold: number) {
  return threshold.toFixed(3)
}

function formatDistanceToFlip(probability: number, decisionThreshold: number) {
  const difference = probability - decisionThreshold
  const absoluteDistance = Math.abs(difference)

  if (absoluteDistance < 0.005) {
    return {
      value: '0.00',
      detail: 'At threshold',
    }
  }

  const needsIncrease = difference < 0
  return {
    value: `${needsIncrease ? '+' : '-'}${absoluteDistance.toFixed(2)}`,
    detail: needsIncrease ? 'to Class 1' : 'to Class 0',
  }
}

export function PredictionSummary({
  prediction,
  treeResults,
  threshold,
  onThresholdChange,
}: PredictionSummaryProps) {
  const meterRef = useRef<HTMLDivElement | null>(null)
  const [isDraggingThreshold, setIsDraggingThreshold] = useState(false)
  const decisionThreshold = threshold
  const thresholdPercent = Math.max(0, Math.min(1, decisionThreshold)) * 100
  const thresholdLabelClass =
    thresholdPercent >= 78 ? 'probability-threshold-label align-left' : 'probability-threshold-label align-right'
  const distanceToFlip = prediction
    ? formatDistanceToFlip(prediction.probability, decisionThreshold)
    : null
  const displayedLabel = prediction ? intLabel(prediction.probability, decisionThreshold) : null

  function thresholdFromPointer(clientX: number) {
    const bounds = meterRef.current?.getBoundingClientRect()
    if (!bounds || bounds.width <= 0) {
      return decisionThreshold
    }

    const rawThreshold = (clientX - bounds.left) / bounds.width
    const clampedThreshold = Math.max(0, Math.min(1, rawThreshold))
    return Math.abs(clampedThreshold - 0.5) <= 0.01 ? 0.5 : clampedThreshold
  }

  function updateThresholdFromPointer(event: PointerEvent<HTMLDivElement>) {
    onThresholdChange(thresholdFromPointer(event.clientX))
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    event.preventDefault()
    setIsDraggingThreshold(true)
    event.currentTarget.setPointerCapture(event.pointerId)
    updateThresholdFromPointer(event)
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (!isDraggingThreshold) {
      return
    }
    updateThresholdFromPointer(event)
  }

  function handlePointerEnd(event: PointerEvent<HTMLDivElement>) {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    setIsDraggingThreshold(false)
  }

  function handleThresholdKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const step = event.shiftKey ? 0.05 : 0.01
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
      event.preventDefault()
      onThresholdChange(Math.max(0, decisionThreshold - step))
    }
    if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
      event.preventDefault()
      onThresholdChange(Math.min(1, decisionThreshold + step))
    }
    if (event.key === 'Home') {
      event.preventDefault()
      onThresholdChange(0)
    }
    if (event.key === 'End') {
      event.preventDefault()
      onThresholdChange(1)
    }
  }

  return (
    <section className="panel prediction-panel">
      <div className="panel-header">
        <div>
          <h2>Prediction Summary</h2>
          <span className="panel-caption">Raw margin first, sigmoid once at the end</span>
        </div>
      </div>
      {prediction ? (
        <>
          <div
            ref={meterRef}
            className={`probability-meter${isDraggingThreshold ? ' is-dragging-threshold' : ''}`}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerEnd}
            onPointerCancel={handlePointerEnd}
          >
            <div className="probability-fill" style={{ width: `${prediction.probability * 100}%` }} />
            <div
              className="probability-threshold-marker"
              style={{ left: `${thresholdPercent}%` }}
              role="slider"
              aria-label="Decision threshold"
              aria-valuemin={0}
              aria-valuemax={1}
              aria-valuenow={decisionThreshold}
              tabIndex={0}
              onKeyDown={handleThresholdKeyDown}
            >
              <span className={thresholdLabelClass}>{`Threshold = ${formatDecisionThreshold(decisionThreshold)}`}</span>
            </div>
            <div className="probability-copy">
              <span>Probability</span>
              <strong>{prediction.probability.toFixed(4)}</strong>
            </div>
          </div>
          <div className="stat-grid">
            <div className="info-card" data-tooltip={DISTANCE_TO_FLIP_EXPLANATION} tabIndex={0}>
              <span>Distance to Flip</span>
              <strong>{distanceToFlip?.value ?? '0.00'}</strong>
              <small>{distanceToFlip?.detail ?? 'At threshold'}</small>
            </div>
            <div className="info-card" data-tooltip={MARGIN_EXPLANATION} tabIndex={0}>
              <span>Margin</span>
              <strong>{prediction.margin.toFixed(4)}</strong>
            </div>
            <div className="info-card" data-tooltip={LABEL_EXPLANATION} tabIndex={0}>
              <span>Label</span>
              <strong>{displayedLabel}</strong>
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

function intLabel(probability: number, threshold: number) {
  return Number(probability >= threshold)
}
