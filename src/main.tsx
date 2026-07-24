import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { LandingPageFallback } from "./landing/LandingPageFallback";
import "./styles.css";

const MotionRouter = lazy(() => import("./MotionRouter"));

function openAppFromFallback() {
  window.history.pushState({}, "", "/app");
  window.location.reload();
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Suspense fallback={<LandingPageFallback onOpenApp={openAppFromFallback} />}>
      <MotionRouter />
    </Suspense>
  </StrictMode>,
);
