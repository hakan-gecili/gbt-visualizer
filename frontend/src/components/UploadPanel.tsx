import { useState } from 'react'

type UploadPanelProps = {
  sessionId: string | null
  modelFamily: string | null
  busy: boolean
  onModelUpload: (file: File) => Promise<void>
  onDatasetUpload: (file: File) => Promise<void>
  onSchemaUpload: (file: File) => Promise<void>
}

export function UploadPanel({
  sessionId,
  modelFamily,
  busy,
  onModelUpload,
  onDatasetUpload,
  onSchemaUpload,
}: UploadPanelProps) {
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [datasetFile, setDatasetFile] = useState<File | null>(null)
  const [schemaFile, setSchemaFile] = useState<File | null>(null)

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

  async function handleSubmitSchema() {
    if (schemaFile) {
      await onSchemaUpload(schemaFile)
    }
  }

  return (
    <section className="panel upload-panel">
      <div className="eyebrow">Session</div>
      <h1>Gradient Boosting Tree Visualizer</h1>
      <p className="lede">
        Upload a LightGBM or XGBoost binary classifier, then inspect fixed radial tree geometry and live path
        updates in raw margin space.
      </p>

      <div className="upload-grid">
        <div className="upload-card">
          <label className="file-card">
            <span>Model File (.txt, .json, .ubj, .bst)</span>
            <input
              type="file"
              accept=".txt,.model,.lgb,.json,.ubj,.xgb,.bst,.pkl,.pickle,.joblib"
              onChange={(event) => setModelFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <button type="button" className="action-button compact-button" disabled={!modelFile || busy} onClick={handleSubmitModel}>
            {busy ? 'Working...' : 'Upload model'}
          </button>
        </div>

        <div className="upload-card">
          <label className="file-card subtle">
            <span>Dataset File (.csv)</span>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => setDatasetFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <button
            type="button"
            className="ghost-button compact-button"
            disabled={!datasetFile || !sessionId || busy}
            onClick={handleSubmitDataset}
          >
            Load dataset
          </button>
        </div>

        <div className="upload-card">
          <label className="file-card subtle">
            <span>Feature Schema File (.json)</span>
            <input
              type="file"
              accept=".json,application/json"
              onChange={(event) => setSchemaFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <button
            type="button"
            className="ghost-button compact-button"
            disabled={!schemaFile || !sessionId || busy}
            onClick={handleSubmitSchema}
          >
            Upload schema
          </button>
        </div>
      </div>

      <div className="session-badge">
        {sessionId ? `${modelFamily ?? 'model'} session ${sessionId.slice(0, 8)}` : 'no active session'}
      </div>
    </section>
  )
}
