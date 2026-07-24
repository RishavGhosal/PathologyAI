import { AnimatePresence, motion } from "framer-motion";
import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { LandingPageFallback } from "./landing/LandingPageFallback";

const LandingPage = lazy(() => import("./landing/LandingPage").then((module) => ({ default: module.LandingPage })));
const ToolApp = lazy(() => import("./App"));

function routeForPath(pathname: string): "landing" | "app" {
  return pathname === "/app" || pathname.startsWith("/app/") ? "app" : "landing";
}

export default function MotionRouter() {
  const [route, setRoute] = useState(() => routeForPath(window.location.pathname));

  useEffect(() => {
    const handlePopState = () => setRoute(routeForPath(window.location.pathname));
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const openApp = useCallback(() => {
    if (route === "app") return;
    window.history.pushState({}, "", "/app");
    setRoute("app");
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [route]);

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        className="route-transition"
        key={route}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
      >
        {route === "app" ? (
          <Suspense fallback={<div className="loading-screen"><p>Opening the review workspace…</p></div>}><ToolApp /></Suspense>
        ) : (
          <Suspense fallback={<LandingPageFallback onOpenApp={openApp} />}><LandingPage onOpenApp={openApp} /></Suspense>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
