import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { useEffect, useMemo, useRef, useState } from "react";
import type { LandingProjectionPoint } from "./embeddingProjection";
import { LANDING_PROJECTION_POINTS } from "./embeddingProjection";

const COLORS = {
  teal: "#00a7a0",
  // This projection has two categorical tones; use a coral counterpart so
  // the groups remain distinguishable against the existing off-white scene.
  sage: "#f47767",
};

const LANDING_RENDER_CAP = 1200;

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
      <Canvas camera={{ position: [0, 0, 6.2], fov: 36 }} dpr={[1, 1.5]} gl={{ alpha: true, antialias: true }}>
        <color attach="background" args={["#f7f6f1"]} />
        <ambientLight intensity={1} />
        <PointCloud points={LANDING_PROJECTION_POINTS.slice(0, LANDING_RENDER_CAP)} />
        <OrbitControls
          ref={controlsRef}
          makeDefault
          enablePan={false}
          enableZoom
          minDistance={3.4}
          maxDistance={8.5}
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={1.7}
          autoRotate={!reducedMotion && !paused}
          autoRotateSpeed={0.55}
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
  const tealPoints = useMemo(() => points.filter((point) => point.tone === "teal"), [points]);
  const sagePoints = useMemo(() => points.filter((point) => point.tone === "sage"), [points]);

  return (
    <group position={[0, 0.04, 0]} scale={[1.2, 1.12, 1.04]}>
      <InstancedPoints points={tealPoints} color={COLORS.teal} />
      <InstancedPoints points={sagePoints} color={COLORS.sage} />
    </group>
  );
}

function InstancedPoints({ points, color }: { points: LandingProjectionPoint[]; color: string }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const helperRef = useRef(new THREE.Object3D());

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const helper = helperRef.current;
    points.forEach((point, index) => {
      helper.position.set((point.x - 0.5) * 2.8, (point.y - 0.5) * 2.15, (point.z - 0.5) * 1.9);
      helper.scale.setScalar(0.72 + (index % 4) * 0.15 + point.z * 0.18);
      helper.updateMatrix();
      mesh.setMatrixAt(index, helper.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  }, [points]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, points.length]} frustumCulled={false}>
      <sphereGeometry args={[0.065, 10, 10]} />
      <meshBasicMaterial color={color} transparent opacity={0.86} />
    </instancedMesh>
  );
}
