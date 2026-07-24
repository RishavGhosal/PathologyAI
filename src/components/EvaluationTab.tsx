import type { WorkspaceSnapshot } from "../types";
import { DataTable } from "./common";

export function EvaluationTab({ workspace }: { workspace: WorkspaceSnapshot }) {
  const providers = workspace.providers;
  const localUni = localStatus(providers.uni, providers.modal_uni, "UNI");
  const localHibou = localStatus(providers.hibou, providers.modal_hibou, "Hibou-B");
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
        <p className="muted">Local weights are optional. On Render, the configured Modal providers are the active model path.</p>
        <div className="provider-cards">
          <article><span className="provider-label">UNI · local</span><strong>{localUni.summary}</strong><p>{localUni.detail}</p></article>
          <article><span className="provider-label">Hibou-B · local</span><strong>{localHibou.summary}</strong><p>{localHibou.detail}</p></article>
        </div>
      </section>
      <section className="evaluation-section">
        <h3>Remote Modal providers</h3>
        <p className="muted">These statuses describe the configured remote model path used by hosted deployments.</p>
        <div className="provider-cards">
          <article><span className="provider-label">UNI · Modal GPU</span><strong>{providers.modal_uni.summary}</strong><p>{providers.modal_uni.detail}</p></article>
          <article><span className="provider-label">Hibou-B · Modal GPU</span><strong>{providers.modal_hibou.summary}</strong><p>{providers.modal_hibou.detail}</p></article>
          <article><span className="provider-label">MHIST proxy · Modal UNI</span><strong>{providers.modal_uni.ready ? "Requested remotely when enabled" : "Unavailable"}</strong><p>The proxy head is requested only with UNI. If the deployed head is unavailable, the batch records a review-priority fallback.</p></article>
        </div>
      </section>
      <p className="footnote">These local methods are exploratory. The displayed evaluation, when present, is dataset-specific and is not clinical performance.</p>
    </section>
  );
}

function localStatus(local: WorkspaceSnapshot["providers"]["uni"], remote: WorkspaceSnapshot["providers"]["modal_uni"], label: string) {
  if (local.ready || !remote.ready) return local;
  return {
    ready: false,
    summary: `Not installed locally (using Modal ${label})`,
    detail: `The hosted app routes ${label} requests through the configured Modal GPU endpoint.`,
  };
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}
