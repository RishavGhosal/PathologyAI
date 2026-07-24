import { motion, useReducedMotion } from "framer-motion";
import { lazy, Suspense } from "react";
import { LANDING_PROJECTION_POINTS } from "./embeddingProjection";

const ease = [0.22, 1, 0.36, 1] as const;
const EmbeddingSpace3D = lazy(() => import("./EmbeddingSpace3D").then((module) => ({ default: module.EmbeddingSpace3D })));

export function LandingPage({ onOpenApp }: { onOpenApp: () => void }) {
  const reducedMotion = useReducedMotion();
  const reveal = reducedMotion
    ? { initial: { opacity: 1, y: 0 }, whileInView: { opacity: 1, y: 0 } }
    : { initial: { opacity: 0, y: 18 }, whileInView: { opacity: 1, y: 0, transition: { duration: 0.32, ease } } };

  return (
    <main className="landing-page" aria-labelledby="landing-title">
      <div className="landing-shell">
        <header className="landing-nav">
          <a className="landing-brand" href="/" aria-label="PathologyAI home">
            <Logo reducedMotion={Boolean(reducedMotion)} />
            <span>PathologyAI</span>
          </a>
          <button className="landing-nav-link" type="button" onClick={onOpenApp}>Open workspace</button>
        </header>

        <section className="landing-hero">
          <div className="landing-hero-copy">
            <p className="landing-kicker">Research / education prototype</p>
            <motion.h1 id="landing-title" initial={reducedMotion ? false : { opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.34, delay: 0.12, ease }}>
              A calmer way to organize image review.
            </motion.h1>
            <motion.p className="landing-lede" initial={reducedMotion ? false : { opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.32, delay: 0.2, ease }}>
              Turn a slide folder into a clear, human-led review queue with transparent visual summaries.
            </motion.p>
            <motion.div initial={reducedMotion ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.28, ease }}>
              <button className="landing-primary-cta" type="button" onClick={onOpenApp}>Open the review workspace <span aria-hidden="true">→</span></button>
            </motion.div>
          </div>
          <motion.div className="landing-hero-note" initial={reducedMotion ? false : { opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.34, delay: 0.34, ease }}>
            <span className="landing-note-line" aria-hidden="true" />
            <p>Built for accountable review: every suggested queue position stays visible, explainable, and human-confirmed.</p>
          </motion.div>
          <EmbeddingCluster reducedMotion={Boolean(reducedMotion)} />
        </section>

        <motion.section className="landing-how" aria-labelledby="how-title" {...reveal} viewport={{ once: true, amount: 0.18 }}>
          <div className="landing-section-intro">
            <p className="landing-kicker">How it works</p>
            <h2 id="how-title">From visual representation to reviewer triage.</h2>
            <p>Three simple stages keep the model output legible and the reviewer in control.</p>
          </div>
          <div className="landing-steps">
            <Step icon="cluster" title="Create embeddings" text="UNI (recommended) or Hibou-B (alternative) maps each image into a compact representation of visual features." />
            <Step icon="tag" title="Estimate proxy labels" text="The MHIST head turns those representations into exploratory review-order labels." />
            <Step icon="checklist" title="Triage with context" text="A reviewer checks image quality, examines the visual summary, and confirms the queue." />
          </div>
        </motion.section>

        <motion.section className="landing-preview" aria-labelledby="preview-title" {...reveal} viewport={{ once: true, amount: 0.16 }}>
          <div className="landing-section-intro">
            <p className="landing-kicker">Inside the workspace</p>
            <h2 id="preview-title">A review queue you can read at a glance.</h2>
            <p>Keep the original image, visual summary, queue suggestion, and reviewer decision in one calm workspace.</p>
          </div>
          <WorkspacePreview />
        </motion.section>

        <motion.section className="landing-cta" aria-labelledby="cta-title" {...reveal} viewport={{ once: true, amount: 0.24 }}>
          <div>
            <p className="landing-kicker">Ready when you are</p>
            <h2 id="cta-title">Start with a small image batch.</h2>
          </div>
          <button className="landing-secondary-cta" type="button" onClick={onOpenApp}>Enter the workspace <span aria-hidden="true">↗</span></button>
        </motion.section>

        <footer className="landing-footer">
          <span>PathologyAI</span>
          <span>Research and education only · Human review required</span>
        </footer>
      </div>
    </main>
  );
}

function Step({ icon, title, text }: { icon: "cluster" | "tag" | "checklist"; title: string; text: string }) {
  return (
    <article className="landing-step">
      <StepIcon kind={icon} />
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
  );
}

function StepIcon({ kind }: { kind: "cluster" | "tag" | "checklist" }) {
  if (kind === "tag") {
    return <svg className="landing-step-icon" viewBox="0 0 32 32" aria-hidden="true"><path d="M6 7.5h10.2L26 17.3 17.3 26 7.5 16.2V7.5Z" /><circle cx="11.5" cy="12" r="1.5" /><path d="m15 17 2 2 4-4" /></svg>;
  }
  if (kind === "checklist") {
    return <svg className="landing-step-icon" viewBox="0 0 32 32" aria-hidden="true"><rect x="7" y="6" width="18" height="21" rx="2" /><path d="m11 12 1.5 1.5L15 11M18 12h4M11 18l1.5 1.5L15 17M18 18h4M11 24h1M18 24h4" /></svg>;
  }
  return <svg className="landing-step-icon" viewBox="0 0 32 32" aria-hidden="true"><circle cx="8" cy="10" r="2.5" /><circle cx="23" cy="7" r="2.5" /><circle cx="23" cy="23" r="2.5" /><circle cx="10" cy="24" r="2.5" /><path d="m10.3 10 10.2-2M9.5 12l-.1 9.5M12 23.5l8.5-.8M23 9.5v11" /></svg>;
}

function EmbeddingCluster({ reducedMotion }: { reducedMotion: boolean }) {
  return <Suspense fallback={<StaticEmbeddingCluster reducedMotion={reducedMotion} />}><EmbeddingSpace3D reducedMotion={reducedMotion} /></Suspense>;
}

function StaticEmbeddingCluster({ reducedMotion }: { reducedMotion: boolean }) {
  const drift = reducedMotion ? {} : { y: [0, -5, 2, 0], x: [0, 2, -1, 0] };
  return (
    <motion.div className="embedding-cluster" initial={reducedMotion ? false : { opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.7, delay: 0.24, ease }}>
      <svg viewBox="0 0 520 360" role="img" aria-label="Loading an interactive 3D t-SNE projection of real UNI embeddings from an MHIST sample">
        <path className="embedding-axis" d="M55 300H472M55 300V48" />
        <motion.g animate={drift} transition={reducedMotion ? undefined : { duration: 12, repeat: Infinity, ease: "easeInOut" }}>
          {LANDING_PROJECTION_POINTS.map((point, index) => {
            const projectedX = point.x * 0.78 + point.z * 0.22;
            const projectedY = point.y * 0.8 + point.z * 0.2;
            return <circle key={`${point.x}-${point.y}-${point.z}`} className={`embedding-point embedding-point-${point.tone}`} cx={58 + projectedX * 400} cy={58 + (1 - projectedY) * 218} r={3.5 + (index % 4) * 1.1 + point.z * 1.4} opacity={0.5 + point.z * 0.5} />;
          })}
        </motion.g>
        <text x="59" y="326">representation space</text><text x="23" y="54" transform="rotate(-90 23 54)">image features</text>
      </svg>
      <span className="embedding-caption">UNI embeddings · MHIST sample · 3D t-SNE projection · loading interactive view</span>
    </motion.div>
  );
}

function WorkspacePreview() {
  return (
    <div className="workspace-preview" aria-label="Simplified preview of the PathologyAI review workspace">
      <div className="workspace-browser-bar"><span /><span /><span /><div className="workspace-address">pathologyai / review workspace</div></div>
      <div className="workspace-preview-body">
        <aside className="workspace-sidebar"><strong>PathologyAI</strong><span className="workspace-sidebar-active">Review queue</span><span>Dashboard</span><span>Evaluation</span><small>Human review required</small></aside>
        <div className="workspace-main">
          <div className="workspace-main-heading"><div><small>REVIEW QUEUE</small><h3>Batch overview</h3></div><span className="workspace-badge">12 images</span></div>
          <div className="workspace-metrics"><div><small>Awaiting review</small><strong>08</strong></div><div><small>Reviewed</small><strong>04</strong></div><div><small>Next queue</small><strong>03</strong></div></div>
          <div className="workspace-content-row"><div className="workspace-image"><svg viewBox="0 0 260 150" aria-hidden="true"><rect width="260" height="150" rx="8" fill="#d9e3da" /><path d="M0 96c28-35 48-24 69-2s42 17 61-13 39-29 60-9 42 13 70-24v102H0Z" fill="#8db2a1" opacity=".72" /><circle cx="72" cy="57" r="19" fill="#527c70" opacity=".62" /><circle cx="160" cy="74" r="24" fill="#b9d8ca" opacity=".9" /><path d="M23 34c30 19 40 11 67 1M178 35c23 14 40 10 64-4" fill="none" stroke="#2f6f69" strokeWidth="2" opacity=".7" /></svg><small>Original image</small></div><div className="workspace-queue"><div><span className="queue-dot queue-dot-high" /><strong>Review First</strong><small>Image quality and feature summary available</small></div><div><span className="queue-dot queue-dot-mid" /><strong>Needs Better Image</strong><small>Manual check suggested before review</small></div><div><span className="queue-dot queue-dot-low" /><strong>Lower Priority</strong><small>Awaiting reviewer confirmation</small></div></div></div>
        </div>
      </div>
    </div>
  );
}

function Logo({ reducedMotion }: { reducedMotion: boolean }) {
  const initial = reducedMotion ? false : { opacity: 0, scale: 0.92, filter: "blur(5px)" };
  const animate = { opacity: 1, scale: 1, filter: "blur(0px)" };
  return (
    <motion.svg className="landing-logo" viewBox="0 0 34 34" role="img" aria-label="PathologyAI research image review mark" initial={initial} animate={animate} transition={{ duration: 1.5, ease }}>
      <motion.path d="M17 3.5 28.5 10v14L17 30.5 5.5 24V10L17 3.5Z" fill="none" stroke="currentColor" strokeWidth="1.4" initial={reducedMotion ? { pathLength: 1, opacity: 1 } : { pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ pathLength: { duration: 1.35, ease }, opacity: { duration: 0.45, ease } }} />
      <motion.path d="M11 23V11h5.4c3.6 0 5.7 1.7 5.7 4.6s-2.1 4.6-5.7 4.6H13" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" initial={reducedMotion ? { pathLength: 1, opacity: 1 } : { pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 1 }} transition={{ pathLength: { duration: 1.35, delay: 0.1, ease }, opacity: { duration: 0.45, delay: 0.1, ease } }} />
    </motion.svg>
  );
}
