export function LandingPageFallback({ onOpenApp }: { onOpenApp: () => void }) {
  return (
    <main className="landing-page landing-page--fallback">
      <div className="landing-shell">
        <header className="landing-nav">
          <div className="landing-brand" aria-label="PathologyAI">
            <span className="landing-brand-mark" aria-hidden="true">P</span>
            <span>PathologyAI</span>
          </div>
          <button className="landing-nav-link" type="button" onClick={onOpenApp}>Open workspace</button>
        </header>
        <section className="landing-hero" aria-labelledby="landing-title">
          <div className="landing-hero-copy">
            <p className="landing-kicker">Research / education prototype</p>
            <h1 id="landing-title">A calmer way to organize image review.</h1>
            <p className="landing-lede">Turn a slide folder into a clear, human-led review queue with transparent visual summaries.</p>
          </div>
          <button className="landing-primary-cta" type="button" onClick={onOpenApp}>Open the review workspace <span aria-hidden="true">→</span></button>
          <div className="landing-fallback-visual" aria-hidden="true"><span /><span /><span /><span /><span /><span /><span /><span /></div>
        </section>
      </div>
    </main>
  );
}
