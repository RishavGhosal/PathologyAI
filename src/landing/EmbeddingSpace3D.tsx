import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useRef, useState } from "react";
import type { LandingProjectionPoint } from "./embeddingProjection";
import { LANDING_PROJECTION_POINTS } from "./embeddingProjection";

const COLORS = {
  teal: "#2f6f69",
  sage: "#739f8b",
};

export function EmbeddingSpace3D({ reducedMotion }: { reducedMotion: boolean }) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const controlsRef = useRef<any>(null);
  const resumeTimerRef = useRef<number | null>(null);
  const [paused, setPaused] = useState(false);

  function pauseForInteraction() {
    if (resumeTimerRef.current !== null) window.clearTimeout(resumeTimerRef.current);
    setPaused(true);
    wrapperRef.current?.setAttribute("data-user-interacted", "true");
  }

  function resumeAfterIdle() {
    if (resumeTimerRef.current !== null) window.clearTimeout(resumeTimerRef.current);
    resumeTimerRef.current = window.setTimeout(() => setPaused(false), 3200);
  }

  useEffect(() => () => {
    if (resumeTimerRef.current !== null) window.clearTimeout(resumeTimerRef.current);
  }, []);

  function recordOrbitChange() {
    const angle = controlsRef.current?.getAzimuthalAngle?.();
    if (typeof angle === "number") {
      wrapperRef.current?.setAttribute("data-orbit-angle", angle.toFixed(3));
    }
  }

  return (
    <div
      ref={wrapperRef}
      className="embedding-cluster embedding-space-3d"
      data-user-interacted="false"
      role="img"
      aria-label="Interactive 3D t-SNE projection of real UNI embeddings from an MHIST sample. Drag to rotate and scroll to zoom."
    >
      <Canvas camera={{ position: [0, 0, 4.8], fov: 34 }} dpr={[1, 1.5]} gl={{ alpha: true, antialias: true }}>
        <color attach="background" args={["#f7f6f1"]} />
        <ambientLight intensity={1} />
        <PointCloud points={LANDING_PROJECTION_POINTS} />
        <gridHelper args={[3.7, 8, "#b9d8ca", "#d8ddd5"]} position={[0, -1.22, 0]} rotation={[Math.PI / 2, 0, 0]} />
        <OrbitControls
          ref={controlsRef}
          makeDefault
          enablePan={false}
          enableZoom
          minDistance={2.7}
          maxDistance={7}
          enableDamping
          dampingFactor={0.08}
          autoRotate={!reducedMotion && !paused}
          autoRotateSpeed={0.3}
          onStart={pauseForInteraction}
          onEnd={resumeAfterIdle}
          onChange={recordOrbitChange}
        />
      </Canvas>
      <span className="embedding-3d-hint">3D t-SNE projection · drag to rotate · scroll to zoom</span>
    </div>
  );
}

function PointCloud({ points }: { points: LandingProjectionPoint[] }) {
  return (
    <group>
      {points.map((point, index) => (
        <mesh
          key={`${point.x}-${point.y}-${point.z}`}
          position={[(point.x - 0.5) * 2.8, (point.y - 0.5) * 2.15, (point.z - 0.5) * 1.9]}
          scale={0.72 + (index % 4) * 0.15 + point.z * 0.18}
        >
          <sphereGeometry args={[0.065, 10, 10]} />
          <meshBasicMaterial color={COLORS[point.tone]} transparent opacity={0.5 + point.z * 0.5} />
        </mesh>
      ))}
    </group>
  );
}
