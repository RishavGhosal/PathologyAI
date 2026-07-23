import type { WorkspaceSnapshot } from "../types";
import { DataTable } from "./common";

export function EvaluationTab({ workspace }: { workspace: WorkspaceSnapshot }) {
  const providers = workspace.providers;
  const metrics = Object.entries(providers.review_model.metrics ?? {}).map(([name, value]) => [
    humanize(name),
    typeof value === "number" ? value.toFixed(3) : value,
  ]);
  return (
    <section className="panel tab-panel evaluation">
      <p className="eyebrow">Evidence and provenance</p>
      <h2>Model evaluation &amp; limits</h2>
      <p className="notice">{workspace.disclaimer}</p>
      <section className="evaluation-section">
        <h3>Experimental MHIST agreement-proxy head</h3>
        <p><strong>{providers.review_model.summary}</strong></p>
        <p className="muted">{providers.review_model.detail}</p>
        <DataTable headers={["Metric", "Value"]} rows={metrics} empty="No validated evaluation metrics are available." />
      </section>
      <section className="evaluation-section">
        <h3>Local feature providers</h3>
        <div className="provider-cards">
          <article><span className="provider-label">UNI</span><strong>{providers.uni.summary}</strong><p>{providers.uni.detail}</p></article>
          <article><span className="provider-label">Hibou-B</span><strong>{providers.hibou.summary}</strong><p>{providers.hibou.detail}</p></article>
        </div>
      </section>
      <p className="footnote">These local methods are exploratory. The displayed evaluation, when present, is dataset-specific and is not clinical performance.</p>
    </section>
  );
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}
