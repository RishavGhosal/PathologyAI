import { motion, useReducedMotion } from "framer-motion";

const ease = [0.22, 1, 0.36, 1] as const;

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
        </section>

        <motion.section className="landing-how" aria-labelledby="how-title" {...reveal} viewport={{ once: true, amount: 0.18 }}>
          <div className="landing-section-intro">
            <p className="landing-kicker">How it works</p>
            <h2 id="how-title">From visual representation to reviewer triage.</h2>
            <p>Three simple stages keep the model output legible and the reviewer in control.</p>
          </div>
          <div className="landing-steps">
            <Step number="01" title="Create embeddings" text="UNI maps each image into a compact representation of visual features." />
            <Step number="02" title="Estimate proxy labels" text="The MHIST head turns those representations into exploratory review-order labels." />
            <Step number="03" title="Triage with context" text="A reviewer checks image quality, examines the visual summary, and confirms the queue." />
          </div>
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

function Step({ number, title, text }: { number: string; title: string; text: string }) {
  return (
    <article className="landing-step">
      <span className="landing-step-number">{number}</span>
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
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
