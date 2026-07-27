import type { Providers, ProviderStatus } from "../types";

export function Overview({ disclaimer, providers }: { disclaimer: string; providers: Providers }) {
  const uni = mergeProviderStatus(providers.uni, providers.modal_uni, "UNI");
  const hibou = mergeProviderStatus(providers.hibou, providers.modal_hibou, "Hibou-B");
  const readyCount = [uni, hibou].filter((provider) => provider.ready).length;

  return (
    <div className="landing">
      <div className="hero">
        <section className="hero-copy">
          <p className="eyebrow">Human-in-the-loop image review</p>
          <h1>Human review, organized</h1>
          <p className="hero-disclaimer">{disclaimer}</p>
          <div className="workflow" aria-label="Review workflow">
            <span className="active">1. Upload</span>
            <span>2. Process</span>
            <span>3. Review images</span>
            <span>4. Export confirmed labels</span>
          </div>
        </section>
        <MicroscopyVisual />
      </div>

      <section className="model-status panel" aria-labelledby="model-status-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Model providers</p>
            <h2 id="model-status-title">Choose a feature encoder</h2>
          </div>
          <span className="status-summary"><span className="status-dot" />{readyCount}/2 available</span>
        </div>
        <p className="muted model-status-intro">
          Choose UNI or Hibou-B when you process a batch. The app uses local weights when available and routes through Modal on hosted deployments.
        </p>
        <div className="model-status-grid">
          <ProviderStatusCard label="UNI" tone="cyan" provider={uni} detailLabel="General-purpose embeddings" description="UNI creates general-purpose image embeddings. The experimental MHIST proxy head uses UNI embeddings when enabled." />
          <ProviderStatusCard label="Hibou-B" tone="violet" provider={hibou} detailLabel="General pathology encoder" description="Hibou-B creates general pathology feature embeddings for exploratory visual summaries." />
        </div>
        <p className="model-footnote">The MHIST proxy is an optional UNI-based review-order capability, not a separate feature encoder.</p>
      </section>

      <section className="scope-strip" aria-label="Product guardrails">
        <span className="scope-icon">◎</span>
        <div>
          <strong>Designed for accountable review</strong>
          <span>Suggestions order a review queue only. They do not identify tissue, disease, cancer, or clinical urgency.</span>
          <span>Optional local encoders never download weights at runtime and fall back to the deterministic method when unavailable.</span>
        </div>
      </section>
    </div>
  );
}

function mergeProviderStatus(local: ProviderStatus, remote: ProviderStatus, label: string): ProviderStatus {
  if (local.ready) return local;
  if (remote.ready) {
    return {
      ready: true,
      summary: `${label} available through Modal`,
      detail: remote.detail,
    };
  }
  return local;
}

function ProviderStatusCard({ label, tone, provider, detailLabel, description }: { label: string; tone: string; provider: ProviderStatus; detailLabel: string; description: string }) {
  return (
    <article className={`model-card ${tone} ${provider.ready ? "ready" : "unavailable"}`}>
      <div className="model-card-top"><span className="model-mark">{label === "MHIST head" ? "↗" : "✦"}</span><span className="model-label">{label}</span><span className="model-ready">{provider.ready ? "Ready" : "Unavailable"}</span></div>
      <strong>{provider.ready ? description : provider.summary}</strong>
      <span className="model-detail-label">{detailLabel}</span>
      <p>{provider.detail}</p>
    </article>
  );
}

function MicroscopyVisual() {
  return (
    <section className="micrograph-card" aria-label="Illustration of a local microscopy review pipeline">
      <div className="micrograph-heading"><span className="eyebrow">Visual pipeline</span><span className="local-pill"><span className="status-dot" />Local only</span></div>
      <div className="micrograph-stage">
        <svg viewBox="0 0 560 360" role="img" aria-label="Abstract microscopy tiles with a feature map overlay">
          <defs>
          <linearGradient id="micro-bg" x1="0" x2="1" y1="0" y2="1"><stop stopColor="var(--pipeline-bg-start)" /><stop offset="1" stopColor="var(--pipeline-bg-end)" /></linearGradient>
            <radialGradient id="cell-blue"><stop stopColor="var(--pipeline-blue)" stopOpacity=".8" /><stop offset="1" stopColor="var(--pipeline-blue)" stopOpacity=".08" /></radialGradient>
            <radialGradient id="cell-pink"><stop stopColor="var(--pipeline-pink)" stopOpacity=".78" /><stop offset="1" stopColor="var(--pipeline-pink)" stopOpacity=".08" /></radialGradient>
            <filter id="soft"><feGaussianBlur stdDeviation="8" /></filter>
          </defs>
          <rect width="560" height="360" rx="18" fill="url(#micro-bg)" />
          <g opacity=".34" filter="url(#soft)"><circle cx="104" cy="106" r="70" fill="url(#cell-blue)" /><circle cx="220" cy="218" r="90" fill="url(#cell-pink)" /><circle cx="420" cy="120" r="92" fill="var(--pipeline-focus)" /></g>
          <g fill="none" stroke="var(--pipeline-blue)" strokeOpacity=".38" strokeWidth="2"><path d="M22 82C72 35 112 64 138 101s70 19 80-26" /><path d="M39 269c48-50 80-27 112 8s79 20 99-22" /><path d="M294 72c36 30 67 28 102-4s70-23 137 25" /><path d="M314 285c44-49 79-42 110 0s72 35 108-4" /></g>
          <g fill="none" stroke="var(--pipeline-pink)" strokeOpacity=".5" strokeWidth="1.5"><ellipse cx="95" cy="118" rx="42" ry="28" transform="rotate(-18 95 118)" /><ellipse cx="186" cy="245" rx="55" ry="36" transform="rotate(22 186 245)" /><ellipse cx="406" cy="124" rx="48" ry="29" transform="rotate(-12 406 124)" /><ellipse cx="481" cy="264" rx="54" ry="34" transform="rotate(18 481 264)" /></g>
          <g fill="var(--accent-tertiary)"><circle cx="126" cy="91" r="4" /><circle cx="164" cy="225" r="3" /><circle cx="375" cy="151" r="4" /><circle cx="448" cy="91" r="3" /><circle cx="477" cy="235" r="4" /></g>
          <rect x="330" y="194" width="168" height="102" rx="12" fill="var(--pipeline-stage-bg)" fillOpacity=".68" stroke="var(--blue)" strokeOpacity=".62" />
          <path d="M346 269c18-41 29-27 42-48 14-23 25 18 40-2 17-23 29-11 53-28" fill="none" stroke="var(--blue)" strokeWidth="3" strokeLinecap="round" />
          <path d="M346 278h120" stroke="var(--blue)" strokeOpacity=".25" /><path d="M346 216v66M376 216v66M406 216v66M436 216v66M466 216v66" stroke="var(--blue)" strokeOpacity=".12" />
          <text x="348" y="214" fill="var(--pipeline-label)" fontFamily="system-ui" fontSize="11" fontWeight="700" letterSpacing="1.2">FEATURE MAP</text>
          <text x="26" y="330" fill="var(--pipeline-muted)" fontFamily="system-ui" fontSize="12">image quality</text><text x="176" y="330" fill="var(--pipeline-muted)" fontFamily="system-ui" fontSize="12">local features</text><text x="332" y="330" fill="var(--pipeline-muted)" fontFamily="system-ui" fontSize="12">review order</text>
        </svg>
      </div>
      <p className="micrograph-caption">A visual summary of the flow: inspect the image, extract local features, then surface the next human review.</p>
    </section>
  );
}
