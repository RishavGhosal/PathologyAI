export function Overview({ disclaimer }: { disclaimer: string }) {
  return (
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
      <section className="panel scope-panel">
        <p className="eyebrow">Guardrails</p>
        <h2>Scope and limits</h2>
        <p>Suggestions order a review queue only. They do not identify tissue, disease, cancer, or clinical urgency.</p>
        <p>Optional local encoders never download weights at runtime and fall back to the deterministic method when unavailable.</p>
      </section>
    </div>
  );
}
