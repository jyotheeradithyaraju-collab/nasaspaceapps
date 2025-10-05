import React, { useEffect, useRef, useState, useMemo } from "react";
import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import * as THREE from "three";
import { TextureLoader } from "three";
import * as satellite from "satellite.js";
// Optional: on-device inference library (only used if you supply TF models)
// import * as tf from '@tensorflow/tfjs';

/**
 * PredictiveSpaceDisasterGlobe (updated)
 * - Keeps the 3D visualization from your original file
 * - Adds a lightweight "PredictionEngine" that can operate in two modes:
 *   1) Orchestrator mode: asks a backend to run heavy ML/pipeline jobs and returns GeoJSON predictions
 *   2) On-device mode (optional): runs supplied TF.js models in-browser for quick prototypes
 *
 * Notes:
 * - This file focuses on the front-end orchestration and minimal local inference hooks.
 * - Real operational prediction still requires backend data sources, preprocessing, and trained models.
 * - See the README comments below for recommended backend endpoints and data sources.
 */

/* ----------------------- Utility: lat/lon → 3D vector ---------------------- */
function latLongToVector3(lat, lon, alt = 0.01) {
  const radius = 1 + alt;
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
}

/* ---------------------------- Atmosphere ----------------------------- */
function Atmosphere() {
  return (
    <mesh>
      <sphereGeometry args={[1.03, 64, 64]} />
      <meshBasicMaterial
        color="#2e8bff"
        transparent
        opacity={0.18}
        side={THREE.BackSide}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  );
}

/* --------------------------------- Stars --------------------------------- */
function Stars() {
  const [positions] = useState(() =>
    Array.from({ length: 1500 }, () => [
      (Math.random() - 0.5) * 25,
      (Math.random() - 0.5) * 25,
      (Math.random() - 0.5) * 25,
    ])
  );
  const flat = useMemo(() => new Float32Array(positions.flat()), [positions]);
  return (
    <points>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length}
          array={flat}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.02} sizeAttenuation />
    </points>
  );
}

/* ---------------------------------- Earth --------------------------------- */
function Earth() {
  const texture = useLoader(
    TextureLoader,
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74477/world.topo.bathy.200412.3x5400x2700.jpg"
  );
  const earthRef = useRef();
  useFrame(() => {
    if (earthRef.current) earthRef.current.rotation.y += 0.0004;
  });
  return (
    <group>
      <mesh ref={earthRef}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshPhongMaterial map={texture} />
      </mesh>
      <Atmosphere />
    </group>
  );
}

/* ------------------------------ SatelliteMarker --------------------------- */
function SatelliteMarker({ sat, color = "#ffcc00", onHover = () => {} }) {
  const [pos, setPos] = useState({ lat: 0, lon: 0, alt: 0, vel: 0 });
  const markerRef = useRef();

  useEffect(() => {
    if (!sat?.tle) return;
    const satrec = satellite.twoline2satrec(sat.tle[0], sat.tle[1]);
    const updatePosition = () => {
      const now = new Date();
      const pv = satellite.propagate(satrec, now);
      if (!pv.position) return;
      const gmst = satellite.gstime(now);
      const geo = satellite.eciToGeodetic(pv.position, gmst);
      const velocity = Math.sqrt(
        pv.velocity.x ** 2 + pv.velocity.y ** 2 + pv.velocity.z ** 2
      );
      setPos({
        lat: (geo.latitude * 180) / Math.PI,
        lon: (geo.longitude * 180) / Math.PI,
        alt: geo.height,
        vel: (velocity * 3600) / 1000,
      });
    };
    updatePosition();
    const intv = setInterval(updatePosition, 3000);
    return () => clearInterval(intv);
  }, [sat]);

  useFrame(() => {
    if (markerRef.current) {
      const v = latLongToVector3(pos.lat, pos.lon, 0.05);
      markerRef.current.position.copy(v);
    }
  });

  return (
    <mesh
      ref={markerRef}
      onPointerOver={() => onHover({ ...sat, ...pos })}
      onPointerOut={() => onHover(null)}
    >
      <sphereGeometry args={[0.015, 16, 16]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

/* ------------------------------- GeoPoint -------------------------------- */
function GeoPoint({ lat, lon, size = 0.01, color = "red", altitude = 0.02 }) {
  const ref = useRef();
  useFrame(() => {
    if (ref.current) {
      const v = latLongToVector3(lat, lon, altitude);
      ref.current.position.copy(v);
    }
  });
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[size, 8, 8]} />
      <meshStandardMaterial color={color} transparent opacity={0.95} />
    </mesh>
  );
}

/* ------------------------------- GeoPolygon ------------------------------- */
function GeoPolygon({ coordinates, color = "orange", opacity = 0.5, altitude = 0.01 }) {
  const pts = useMemo(() => {
    if (!coordinates || !Array.isArray(coordinates)) return [];
    const ring =
      Array.isArray(coordinates[0][0]) && typeof coordinates[0][0][0] === "number"
        ? coordinates[0]
        : coordinates[0][0] || [];
    return ring.map(([lon, lat]) => latLongToVector3(lat, lon, altitude));
  }, [coordinates, altitude]);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    if (pts.length) g.setFromPoints(pts);
    return g;
  }, [pts]);
  if (!pts.length) return null;
  return (
    <line geometry={geometry}>
      <lineBasicMaterial color={color} transparent opacity={opacity} linewidth={1} />
    </line>
  );
}

/* ------------------------- Prediction Engine (frontend) ------------------- */
/**
 * PredictionEngine responsibilities:
 * 1) Acquire input data (optionally) from public sources or a backend ingest service
 * 2) Call either an on-device model (if provided) or orchestrate backend inference jobs
 * 3) Return standardized GeoJSON predictions for the front-end visualizer
 *
 * Modes:
 * - mode: 'orchestrator' (default)
 *     - POST /api/ai/infer  { disaster: 'fires', horizonHours: 24 }
 *     - The backend responds with GeoJSON predictions.
 * - mode: 'local' (optional)
 *     - Load TF.js model URL(s) and run in-browser quick inference for small areas.
 *
 * Security/scale notes: do heavy work on server-side. On-device inference is only for experiments.
 */

function usePredictionEngine({ mode = "orchestrator", apiBase = "", modelUrls = {} } = {}) {
  const abortControllers = useRef({});

  // fetch predictions for a disaster/horizon
  const infer = async (disaster, horizonHours, options = {}) => {
    if (mode === "local") {
      // Basic on-device stub: user must supply modelUrls[disaster]
      // This is a *placeholder* demonstrating how to plug TF.js inference.
      if (!modelUrls || !modelUrls[disaster]) {
        throw new Error("No local model URL supplied for disaster: " + disaster);
      }
      // Example pseudo-steps (do not actually import tf here unless you bundle it):
      // const model = await tf.loadGraphModel(modelUrls[disaster]);
      // const featuresTensor = preprocessLocalInputs(options.inputs);
      // const logits = model.predict(featuresTensor);
      // const geojson = postprocessToGeoJSON(logits);
      // return geojson;
      return { type: "FeatureCollection", features: [] }; // stub
    }

    // Default: orchestrator mode - ask backend to produce predictions
    const controller = new AbortController();
    const signal = controller.signal;
    const key = `${disaster}-${horizonHours}`;
    abortControllers.current[key] = controller;

    try {
      const res = await fetch(`${apiBase}/api/ai/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal,
        body: JSON.stringify({ disaster, horizonHours, options }),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Inference request failed: ${res.status} ${txt}`);
      }
      const json = await res.json();
      // expect { geojson: {...} }
      return json.geojson || json;
    } finally {
      delete abortControllers.current[key];
    }
  };

  const cancel = (disaster, horizonHours) => {
    const key = `${disaster}-${horizonHours}`;
    const c = abortControllers.current[key];
    if (c) c.abort();
  };

  // cleanup
  useEffect(() => {
    return () => {
      Object.values(abortControllers.current).forEach((c) => c.abort());
      abortControllers.current = {};
    };
  }, []);

  return { infer, cancel };
}

/* -------------------------- Predictive Layer Component --------------------- */
function RiskForecastOverlay({ enabled = true, hoursList = [6, 12, 24], pollInterval = 30 * 60 * 1000 /* 30m */ }) {
  const API_BASE = process.env.REACT_APP_API_BASE || "";

  // use enhanced prediction engine (orchestrator mode pointing to API_BASE)
  const { infer } = usePredictionEngine({ mode: "orchestrator", apiBase: API_BASE });

  const [predictions, setPredictions] = useState({ fires: {}, floods: {}, landslides: {} });
  const groupRefs = useRef(hoursList.map(() => React.createRef()));

  useEffect(() => {
    if (!enabled) return;
    let mounted = true;

    const fetchAll = async () => {
      try {
        const disasters = ["fires", "floods", "landslides"];
        const newPred = { fires: {}, floods: {}, landslides: {} };

        // For each disaster/horizon ask the PredictionEngine to infer
        for (const d of disasters) {
          for (const h of hoursList) {
            try {
              // infer() will POST to /api/ai/infer {disaster, horizonHours}
              const geojson = await infer(d, h, { realtime: true });
              const features = geojson && Array.isArray(geojson.features) ? geojson.features : [];
              newPred[d][h] = features;
            } catch (err) {
              console.warn(`Prediction fetch failed for ${d} ${h}h:`, err.message || err);
              newPred[d][h] = [];
            }
          }
        }

        if (!mounted) return;
        setPredictions(newPred);
      } catch (e) {
        console.error("RiskForecastOverlay fetch error", e);
      }
    };

    // initial fetch + polling
    fetchAll();
    const intv = setInterval(fetchAll, pollInterval);
    return () => {
      mounted = false;
      clearInterval(intv);
    };
  }, [enabled, JSON.stringify(hoursList), pollInterval]);

  useFrame((state, delta) => {
    const speeds = [0.0012, -0.0008, 0.0005];
    groupRefs.current.forEach((ref, i) => {
      const g = ref.current;
      if (g) g.rotation.y += speeds[i % speeds.length] * (delta * 60);
    });
  });

  if (!enabled) return null;

  const disasterStyle = {
    fires: { color: "#ff6600", altOffset: 0.02 },
    floods: { color: "#0066ff", altOffset: 0.025 },
    landslides: { color: "#ffcc00", altOffset: 0.018 },
  };

  return (
    <>
      {hoursList.map((h, horizonIdx) => (
        <group key={`horizon-${h}`} ref={groupRefs.current[horizonIdx]}>
          {Object.keys(disasterStyle).map((d) => {
            const features = (predictions[d] && predictions[d][h]) || [];
            const style = disasterStyle[d] || { color: "#fff", altOffset: 0.02 };
            return (
              <group key={`${d}-${h}`}>
                {features.map((feat, i) => {
                  if (!feat || !feat.geometry) return null;
                  const geom = feat.geometry;
                  if (geom.type === "Point") {
                    const [lon, lat] = geom.coordinates;
                    const risk = feat.properties?.risk ?? 0.5;
                    const size = 0.006 + Math.min(0.03, risk * 0.08);
                    return (
                      <GeoPoint
                        key={`${d}-${h}-pt-${i}`}
                        lat={lat}
                        lon={lon}
                        size={size}
                        color={style.color}
                        altitude={1 + style.altOffset ? 1 + style.altOffset : 0.02}
                      />
                    );
                  } else if (geom.type === "Polygon" || geom.type === "MultiPolygon") {
                    return (
                      <GeoPolygon
                        key={`${d}-${h}-poly-${i}`}
                        coordinates={geom.type === "Polygon" ? geom.coordinates : geom.coordinates[0]}
                        color={style.color}
                        opacity={0.22}
                        altitude={0.01 + (horizonIdx * 0.002)}
                      />
                    );
                  } else {
                    return null;
                  }
                })}
              </group>
            );
          })}
        </group>
      ))}
    </>
  );
}

/* -------------------------- Top-level component --------------------------- */
export default function PredictiveSpaceDisasterGlobe() {
  const [satellites, setSatellites] = useState([]);
  const [hovered, setHovered] = useState(null);
  const [disastersEnabled, setDisastersEnabled] = useState(true);
  const [forecastEnabled, setForecastEnabled] = useState(true);
  const [habitatEnabled, setHabitatEnabled] = useState(true);

  useEffect(() => {
    const fetchTLE = async () => {
      try {
        const res = await fetch("https://celestrak.org/NORAD/elements/active.txt");
        const txt = await res.text();
        const lines = txt.split("\n").filter(Boolean);
        const sats = [];
        for (let i = 0; i < lines.length; i += 3) {
          if (lines[i + 1] && lines[i + 2]) {
            sats.push({
              name: lines[i].trim(),
              tle: [lines[i + 1], lines[i + 2]],
            });
          }
          if (sats.length >= 6) break;
        }
        setSatellites(sats);
      } catch (e) {
        console.warn("TLE fetch failed", e);
      }
    };
    fetchTLE();
  }, []);

  return (
    <div style={{ position: "relative", width: "100%", height: "100vh", overflow: "hidden" }}>
      <Canvas camera={{ position: [0, 0, 3] }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[5, 3, 5]} intensity={1.2} />
        <Stars />
        <Earth />
        {satellites.map((s, i) => (
          <SatelliteMarker
            key={s.name + i}
            sat={s}
            color={["#ffcc00", "#00ffff", "#ff66cc", "#33ff33", "#ff3333", "#aa66ff"][i % 6]}
            onHover={setHovered}
          />
        ))}

        <RiskForecastOverlay enabled={forecastEnabled} hoursList={[6, 12, 24]} pollInterval={5 * 60 * 1000} />

        {habitatEnabled && (
          <group position={[1.5, -0.8, 0.8]} rotation={[0.2, -0.6, 0]}>
            <mesh>
              <boxGeometry args={[0.8, 0.56, 0.4]} />
              <meshStandardMaterial color="#222" transparent opacity={0.15} />
            </mesh>
          </group>
        )}
      </Canvas>

      <div
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          width: 360,
          padding: 12,
          borderRadius: 10,
          background: "rgba(8,10,20,0.85)",
          color: "#cfe",
          fontFamily: "system-ui, monospace",
          zIndex: 5,
          border: "1px solid rgba(0,200,255,0.3)",
        }}
      >
        <h3 style={{ margin: "0 0 8px 0", color: "#8ff" }}>Predictive Mission Control</h3>
        <label style={{ display: "block", marginBottom: 6 }}>
          <input type="checkbox" checked={disastersEnabled} onChange={(e) => setDisastersEnabled(e.target.checked)} /> Show Real-time Events
        </label>
        <label style={{ display: "block", marginBottom: 6 }}>
          <input type="checkbox" checked={forecastEnabled} onChange={(e) => setForecastEnabled(e.target.checked)} /> Show Predictive Forecasts (6 / 12 / 24h)
        </label>
        <label style={{ display: "block", marginBottom: 6 }}>
          <input type="checkbox" checked={habitatEnabled} onChange={(e) => setHabitatEnabled(e.target.checked)} /> Habitat Monitor
        </label>

        <hr style={{ borderColor: "rgba(255,255,255,0.06)" }} />
        <div style={{ fontSize: 13 }}>
          <strong>Legend</strong>
          <div>• Fires: orange (predicted)</div>
          <div>• Floods: blue (predicted)</div>
          <div>• Landslides: yellow (predicted)</div>
          <div style={{ marginTop: 6, color: "#9ff", fontSize: 12 }}>
            Layers: 6h (fast orbit), 12h (mid), 24h (slow).
          </div>
        </div>

        <hr style={{ borderColor: "rgba(255,255,255,0.06)" }} />
        <div style={{ fontSize: 13, minHeight: 56 }}>
          {hovered ? (
            <>
              <div style={{ color: "#ffd" }}><strong>{hovered.name}</strong></div>
              <div>Lat: {hovered.lat?.toFixed(2)}°</div>
              <div>Lon: {hovered.lon?.toFixed(2)}°</div>
              <div>Alt: {hovered.alt?.toFixed(2)} km</div>
              <div>Vel: {hovered.vel?.toFixed(0)} km/h</div>
            </>
          ) : (
            <div>Hover satellites for telemetry.</div>
          )}
        </div>

        <hr style={{ borderColor: "rgba(255,255,255,0.06)" }} />
        <div style={{ fontSize: 12, marginTop: 6 }}>
          <strong>Notes</strong>
          <ul style={{ paddingLeft: 16 }}>
            <li>Set <code>REACT_APP_API_BASE</code> to your backend host (e.g. https://example.com) which must implement /api/ai/infer.</li>
            <li>Backend should accept POST {"{disaster,horizonHours,options}"} and return GeoJSON: {"{type:'FeatureCollection',features:[...] }"}.</li>
            <li>On-device TF.js inference is optional for prototypes — supply model URLs in usePredictionEngine.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

/* -------------------------- README (short) ----------------------------- */
/*
Recommended backend endpoints (server-side responsibilities):

POST /api/ai/infer
  body: { disaster: 'fires'|'floods'|'landslides', horizonHours: number, options?: {} }
  response: { geojson: GeoJSON.FeatureCollection }

Why server-side is required:
 - Ingest large satellite imagery (MODIS/VIIRS), weather forecast grids (GFS/ECMWF), hydrology models, DEMs
 - Preprocess (tiles, radiometric correction, resample), compute features (NDVI, soil moisture index, antecedent precipitation)
 - Run heavy ML models (CNNs over imagery, physics-based hydrology, ensemble models)
 - Post-process outputs into probabilities and geojson predictions

Suggested data sources and pipeline components:
 - NASA FIRMS (active fire hotspots)
 - MODIS/VIIRS imagery
 - NOAA GFS / ECMWF forecasts
 - Digital Elevation Models for slope/landslide susceptibility
 - Local rainfall / gauge networks if available
 - Model frameworks: PyTorch/TensorFlow/Keras on the server; containerize with Docker + GPU if required

Operational notes:
 - Prediction uncertainty is critical. Include probabilistic outputs and lead-time metrics.
 - Validate using historical events and a holdout region-based evaluation.
 - For real warnings, hook outputs to alerting and human verification workflows.
*/
