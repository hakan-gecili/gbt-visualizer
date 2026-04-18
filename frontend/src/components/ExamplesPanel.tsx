type ExamplesPanelProps = {
  examples: string[]
  selectedExample: string
  busy: boolean
  onSelectExample: (exampleName: string) => void
}

export function ExamplesPanel({
  examples,
  selectedExample,
  busy,
  onSelectExample,
}: ExamplesPanelProps) {
  return (
    <section className="panel examples-panel">
      <div className="panel-header">
        <h2>Examples</h2>
        <span className="panel-caption">{examples.length} available</span>
      </div>
      <p className="lede examples-copy">
        Select a bundled example to automatically load one model and one dataset from its folder.
      </p>
      <label className="examples-select-label" htmlFor="example-select">
        Example
      </label>
      <select
        id="example-select"
        className="examples-select"
        value={selectedExample}
        disabled={busy || examples.length === 0}
        onChange={(event) => onSelectExample(event.target.value)}
      >
        <option value="">Choose an example</option>
        {examples.map((example) => (
          <option key={example} value={example}>
            {example}
          </option>
        ))}
      </select>
    </section>
  )
}
