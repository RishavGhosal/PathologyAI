import type { BatchState } from "../types";
import { DataTable, MetricsCards, PriorityChip } from "./common";
import { effectivePriority } from "../queue";
import { EmbeddingMap } from "./EmbeddingMap";

export function DashboardTab({ batch }: { batch: BatchState }) {
  const metrics = batch.metrics;
  return (
    <section className="panel tab-panel">
      <p className="eyebrow">Batch operations</p>
      <h2>Operational dashboard</h2>
      <MetricsCards metrics={metrics} />
      <EmbeddingMap projection={batch.embedding_projection} />
      <section className="batch-summaries" aria-label="Computed batch summaries">
        <h3>Computed summaries for high-priority images</h3>
        {batch.records.filter((record) => effectivePriority(record) !== "Lower Priority").length ? (
          <div className="summary-list">
            {batch.records
              .filter((record) => effectivePriority(record) !== "Lower Priority")
              .map((record) => (
                <div className="summary-row" key={record.id}>
                  <PriorityChip priority={effectivePriority(record)} />
                  <strong>{record.name}</strong>
                  <span>{record.computed?.summary ?? "Computed region summary unavailable."}</span>
                </div>
              ))}
          </div>
        ) : <p className="empty compact">No high-priority images in this batch.</p>}
      </section>
      <div className="dashboard-grid">
        <section>
          <h3>Effective review priorities</h3>
          <div className="priority-breakdown">
            {Object.entries(metrics.effective_priority_counts).map(([priority, count]) => (
              <div key={priority}><PriorityChip priority={priority as keyof typeof metrics.effective_priority_counts} /><b>{count}</b></div>
            ))}
          </div>
          <h3>Quality findings</h3>
          <DataTable headers={["Issue", "Images"]} rows={Object.entries(metrics.quality_issue_counts)} empty="No blocking quality findings." />
          <h3>Nonblocking advisories</h3>
          <DataTable headers={["Advisory", "Images"]} rows={Object.entries(metrics.quality_advisory_counts)} empty="No nonblocking advisories." />
        </section>
        <section>
          <h3>Reviewer agreement</h3>
          <DataTable
            headers={["Suggested", "Confirmed", "Agreement"]}
            rows={metrics.agreement_by_suggested_priority.map((row) => [
              row.suggested_priority,
              `${row.confirmed_count}/${row.reviewed_count}`,
              row.agreement_percentage == null ? "Not measured" : `${row.agreement_percentage.toFixed(1)}%`,
            ])}
          />
          <h3>Method provenance</h3>
          <dl className="provenance">
            <div><dt>Deterministic</dt><dd>{metrics.deterministic_prediction_count}</dd></div>
            <div><dt>Experimental</dt><dd>{metrics.experimental_model_prediction_count}</dd></div>
            <div><dt>Quality gate</dt><dd>{metrics.quality_gate_count}</dd></div>
            <div><dt>Runtime fallback</dt><dd>{metrics.runtime_fallback_count}</dd></div>
          </dl>
        </section>
      </div>
      <p className="footnote">Estimated time avoided covers only unusable or skipped files, not time saved by ranking. 0s means this batch had no unusable or skipped files.</p>
    </section>
  );
}
