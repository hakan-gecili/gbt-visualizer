import type { PreviewPayload } from '../types/api'

type DatasetTableProps = {
  preview: PreviewPayload | null
  selectedRowIndex: number | null
  onSelectRow: (rowIndex: number) => void
}

export function DatasetTable({ preview, selectedRowIndex, onSelectRow }: DatasetTableProps) {
  return (
    <section className="panel dataset-panel">
      <div className="panel-header">
        <h2>Dataset</h2>
        <span className="panel-caption">
          {preview ? `${preview.rows.length} preview rows` : 'Upload CSV to inspect rows'}
        </span>
      </div>
      {preview ? (
        <div className="table-shell">
          <table>
            <thead>
              <tr>
                <th>Row</th>
                {preview.columns.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, rowIndex) => (
                <tr
                  key={rowIndex}
                  className={selectedRowIndex === rowIndex ? 'selected' : ''}
                  onClick={() => onSelectRow(rowIndex)}
                >
                  <td>{rowIndex}</td>
                  {row.map((value, valueIndex) => (
                    <td key={`${rowIndex}-${valueIndex}`}>{value ?? 'null'}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty-state">No dataset loaded. Manual feature editing still works.</div>
      )}
    </section>
  )
}
