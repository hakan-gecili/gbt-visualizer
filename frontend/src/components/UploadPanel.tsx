import { useState } from 'react'

type UploadPanelProps = {
  sessionId: string | null
  busy: boolean
  onModelUpload: (file: File) => Promise<void>
  onDatasetUpload: (file: File) => Promise<void>
}

export function UploadPanel({
  sessionId,
  busy,
  onModelUpload,
  onDatasetUpload,
}: UploadPanelProps) {
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [datasetFile, setDatasetFile] = useState<File | null>(null)

  async function handleSubmitModel() {
    if (modelFile) {
      await onModelUpload(modelFile)
    }
  }

  async function handleSubmitDataset() {
    if (datasetFile) {
      await onDatasetUpload(datasetFile)
    }
  }

  return (
    <section className="panel upload-panel">
      <div className="eyebrow">Session</div>
      <h1>Gradient Boosting Tree Visualizer</h1>
      <p className="lede">
        Upload a LightGBM binary classifier, then inspect fixed radial tree geometry and live path
        updates in raw margin space.
      </p>

      <label className="file-card">
        <span>Model file</span>
        <input
          type="file"
          accept=".txt,.model,.lgb,.bst"
          onChange={(event) => setModelFile(event.target.files?.[0] ?? null)}
        />
      </label>
      <button type="button" className="action-button" disabled={!modelFile || busy} onClick={handleSubmitModel}>
        {busy ? 'Working...' : 'Upload model'}
      </button>

      <label className="file-card subtle">
        <span>Optional CSV dataset</span>
        <input type="file" accept=".csv,text/csv" onChange={(event) => setDatasetFile(event.target.files?.[0] ?? null)} />
      </label>
      <button
        type="button"
        className="ghost-button"
        disabled={!datasetFile || !sessionId || busy}
        onClick={handleSubmitDataset}
      >
        Load dataset
      </button>

      <div className="session-badge">{sessionId ? `session ${sessionId.slice(0, 8)}` : 'no active session'}</div>
    </section>
  )
}
