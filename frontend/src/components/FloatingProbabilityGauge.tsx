import { useEffect, useMemo, useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent, RefObject } from 'react'

import { radialTheme } from './radialTheme'

type FloatingProbabilityGaugeProps = {
  probability: number | null
  isDarkMode: boolean
  appShellRef: RefObject<HTMLElement | null>
  sidebarRef: RefObject<HTMLElement | null>
  isFeatureImportanceOpen: boolean
  onToggleFeatureImportance: () => void
}

type Point = {
  x: number
  y: number
}

const GAUGE_SIZE = 112
const GAUGE_CENTER = GAUGE_SIZE / 2
const GAUGE_RADIUS = 38
const GAUGE_START_ANGLE = -200
const GAUGE_SWEEP = 225
const PANEL_MARGIN = 16
const INITIAL_TOP = 214

function gaugeArcPath(radius: number) {
  const startRadians = (GAUGE_START_ANGLE * Math.PI) / 180
  const endRadians = ((GAUGE_START_ANGLE + GAUGE_SWEEP) * Math.PI) / 180
  const startX = GAUGE_CENTER + Math.cos(startRadians) * radius
  const startY = GAUGE_CENTER + Math.sin(startRadians) * radius
  const endX = GAUGE_CENTER + Math.cos(endRadians) * radius
  const endY = GAUGE_CENTER + Math.sin(endRadians) * radius
  return `M ${startX} ${startY} A ${radius} ${radius} 0 1 1 ${endX} ${endY}`
}

function gaugeNeedleAngle(probability: number) {
  const clamped = Math.max(0, Math.min(1, probability))
  return GAUGE_START_ANGLE + clamped * GAUGE_SWEEP
}

function clampPosition(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export function FloatingProbabilityGauge({
  probability,
  isDarkMode,
  appShellRef,
  sidebarRef,
  isFeatureImportanceOpen,
  onToggleFeatureImportance,
}: FloatingProbabilityGaugeProps) {
  const panelRef = useRef<HTMLDivElement | null>(null)
  const dragStartMouseRef = useRef<Point | null>(null)
  const dragStartPositionRef = useRef<Point>({ x: 0, y: 0 })
  const hasManualPositionRef = useRef(false)
  const [position, setPosition] = useState<Point>({ x: PANEL_MARGIN, y: INITIAL_TOP })
  const [isDragging, setIsDragging] = useState(false)
  const theme = radialTheme(isDarkMode)
  const gaugeValue = probability ?? 0
  const gaugeAngle = gaugeNeedleAngle(gaugeValue)

  const clampToShell = useMemo(
    () => (nextPosition: Point) => {
      const shell = appShellRef.current
      const panel = panelRef.current
      if (!shell || !panel) {
        return nextPosition
      }

      const maxX = Math.max(PANEL_MARGIN, shell.clientWidth - panel.offsetWidth - PANEL_MARGIN)
      const maxY = Math.max(PANEL_MARGIN, shell.clientHeight - panel.offsetHeight - PANEL_MARGIN)
      return {
        x: clampPosition(nextPosition.x, PANEL_MARGIN, maxX),
        y: clampPosition(nextPosition.y, PANEL_MARGIN, maxY),
      }
    },
    [appShellRef],
  )

  useEffect(() => {
    const updatePosition = () => {
      const shell = appShellRef.current
      const sidebar = sidebarRef.current
      const panel = panelRef.current
      if (!shell || !sidebar || !panel) {
        return
      }

      if (hasManualPositionRef.current) {
        setPosition((current) => clampToShell(current))
        return
      }

      const shellRect = shell.getBoundingClientRect()
      const sidebarRect = sidebar.getBoundingClientRect()
      const anchorX = sidebarRect.right - shellRect.left + 8
      setPosition(
        clampToShell({
          x: anchorX,
          y: INITIAL_TOP,
        }),
      )
    }

    updatePosition()
    window.addEventListener('resize', updatePosition)
    return () => window.removeEventListener('resize', updatePosition)
  }, [appShellRef, clampToShell, sidebarRef])

  useEffect(() => {
    if (!isDragging) {
      return
    }

    const handleMouseMove = (event: MouseEvent) => {
      const dragStartMouse = dragStartMouseRef.current
      if (!dragStartMouse) {
        return
      }

      setPosition(
        clampToShell({
          x: dragStartPositionRef.current.x + (event.clientX - dragStartMouse.x),
          y: dragStartPositionRef.current.y + (event.clientY - dragStartMouse.y),
        }),
      )
    }

    const stopDragging = () => {
      setIsDragging(false)
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
  }, [clampToShell, isDragging])

  const handleDragStart = (event: ReactMouseEvent<HTMLButtonElement>) => {
    if (event.button !== 0) {
      return
    }

    event.preventDefault()
    hasManualPositionRef.current = true
    dragStartMouseRef.current = { x: event.clientX, y: event.clientY }
    dragStartPositionRef.current = position
    setIsDragging(true)
  }

  return (
    <div
      ref={panelRef}
      className={isDragging ? 'floating-gauge-panel dragging' : 'floating-gauge-panel'}
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        background: theme.gaugeCardBackground,
        borderColor: theme.gaugeCardBorder,
        boxShadow: theme.gaugeCardShadow,
      }}
      aria-label="Predicted probability gauge"
    >
      <button
        type="button"
        className="floating-gauge-handle"
        aria-label="Drag probability gauge panel"
        onMouseDown={handleDragStart}
      >
        <span style={{ background: theme.gaugeHandle }} />
        <span style={{ background: theme.gaugeHandle }} />
        <span style={{ background: theme.gaugeHandle }} />
      </button>
      <svg className="floating-radial-gauge" viewBox={`0 0 ${GAUGE_SIZE} ${GAUGE_SIZE}`} role="img">
        <path d={gaugeArcPath(GAUGE_RADIUS)} className="gauge-track" style={{ stroke: theme.gaugeTrack }} />
        <g
          className="gauge-needle"
          style={{
            transform: `rotate(${gaugeAngle}deg)`,
            transformOrigin: `${GAUGE_CENTER}px ${GAUGE_CENTER}px`,
          }}
        >
          <line
            x1={GAUGE_CENTER}
            y1={GAUGE_CENTER}
            x2={GAUGE_CENTER + GAUGE_RADIUS - 8}
            y2={GAUGE_CENTER}
            stroke={theme.gaugeNeedle}
            strokeWidth={3.2}
            strokeLinecap="round"
          />
        </g>
        <circle cx={GAUGE_CENTER} cy={GAUGE_CENTER} r={4.5} fill={theme.gaugeHub} />
        <text x={GAUGE_CENTER} y={GAUGE_CENTER + 22} textAnchor="middle" className="gauge-value" fill={theme.gaugeText}>
          {gaugeValue.toFixed(2)}
        </text>
      </svg>
      <span className="floating-gauge-label" style={{ color: theme.gaugeLabel }}>
        Probability
      </span>
      <button type="button" className="floating-feature-importance-toggle" onClick={onToggleFeatureImportance}>
        {isFeatureImportanceOpen ? 'Hide Importance' : 'Feature Importance'}
      </button>
    </div>
  )
}
