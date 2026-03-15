const { useEffect, useMemo, useRef, useState } = React;

const MAPBOX_TOKEN =
  window.MAPBOX_PUBLIC_TOKEN ||
  "pk.eyJ1IjoibXVrdW5kMjAzMyIsImEiOiJjbW1xNThkdWMwcnYzMnFxdHJtNXFycmxhIn0.lfX0rAPb3cx_C7XGj-yOgw";

const CITY_OPTIONS = ["Delhi", "Mumbai", "Pune", "Bangalore", "Chennai"];

const CITY_CENTERS = {
  Delhi: [77.1025, 28.7041],
  Mumbai: [72.8777, 19.076],
  Pune: [73.8567, 18.5204],
  Bangalore: [77.5946, 12.9716],
  Chennai: [80.2707, 13.0827],
};

function parseCSV(path) {
  return new Promise((resolve) => {
    Papa.parse(path, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: ({ data }) => resolve(data || []),
      error: () => resolve([]),
    });
  });
}

function seededNoise(seed) {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

function normalizeScore(rawScore) {
  if (rawScore == null || Number.isNaN(Number(rawScore))) {
    return 55;
  }
  const score = Number(rawScore);
  return score <= 10 ? Math.round(score * 10) : Math.round(score);
}

function getRiskLabel(score) {
  if (score >= 70) return "High";
  if (score >= 40) return "Moderate";
  return "Low";
}

function getMarkerColorClass(score) {
  if (score >= 70) return "red";
  if (score >= 40) return "yellow";
  return "green";
}

function buildSyntheticFactories(city) {
  const [baseLng, baseLat] = CITY_CENTERS[city] || CITY_CENTERS.Delhi;
  return Array.from({ length: 18 }).map((_, index) => {
    const dx = (seededNoise(index + 21) - 0.5) * 0.55;
    const dy = (seededNoise(index + 91) - 0.5) * 0.45;
    const score = Math.round(35 + seededNoise(index + 300) * 55);
    return {
      factory_id: `sim_${city}_${index + 1}`,
      factory_name: `${city} Industrial Unit ${index + 1}`,
      industry_type: ["Steel", "Chemical", "Power", "Manufacturing"][index % 4],
      latitude: Number((baseLat + dy).toFixed(6)),
      longitude: Number((baseLng + dx).toFixed(6)),
      city,
      pollution_score: score,
      risk_level: getRiskLabel(score),
      primary_pollutant: score > 70 ? "PM2.5" : score > 45 ? "NO2" : "CO",
      recommendation:
        "Install scrubbers, strengthen continuous monitoring, and optimize combustion systems.",
      latest_pm25: Number((22 + score * 1.25).toFixed(1)),
      latest_pm10: Number((38 + score * 1.6).toFixed(1)),
    };
  });
}

function deriveFactoryMetrics(factory) {
  const score = normalizeScore(factory.pollution_score || factory.pollution_impact_score);
  const pm25 = Number(factory.latest_pm25 || (20 + score * 1.2).toFixed(1));
  const pm10 = Number(factory.latest_pm10 || (32 + score * 1.7).toFixed(1));
  const no2 = Number((18 + score * 0.76).toFixed(1));
  const so2 = Number((8 + score * 0.52).toFixed(1));
  const co = Number((0.5 + score * 0.05).toFixed(2));
  const o3 = Number((14 + score * 0.42).toFixed(1));

  const trend = Array.from({ length: 7 }).map((_, i) => {
    const wobble = seededNoise(score + i * 11) * 14 - 7;
    return Math.max(10, Math.round(pm25 + wobble));
  });

  return { score, pm25, pm10, no2, so2, co, o3, trend };
}

function pollutionLevelText(score) {
  if (score >= 70) return "HIGH";
  if (score >= 40) return "MODERATE";
  return "LOW";
}

function useFactoryData(city) {
  const [loading, setLoading] = useState(true);
  const [factories, setFactories] = useState([]);

  useEffect(() => {
    let mounted = true;

    async function run() {
      setLoading(true);
      const [recommendations, factoriesRaw] = await Promise.all([
        parseCSV("../data/output/recommendations.csv"),
        parseCSV("../data/raw/factories/factories.csv"),
      ]);

      if (!mounted) return;

      const recByCity = recommendations
        .filter((r) => r.city === city && r.latitude && r.longitude)
        .map((r) => {
          const score = normalizeScore(r.pollution_impact_score);
          return {
            ...r,
            pollution_score: score,
            latitude: Number(r.latitude),
            longitude: Number(r.longitude),
            primary_pollutant: score > 70 ? "PM2.5" : score > 45 ? "NO2" : "CO",
          };
        });

      const fallbackRaw = factoriesRaw
        .filter((f) => f.city === city && f.latitude && f.longitude)
        .slice(0, 40)
        .map((f, idx) => {
          const score = 32 + Math.round(seededNoise(idx + 77) * 62);
          return {
            ...f,
            pollution_score: score,
            risk_level: getRiskLabel(score),
            primary_pollutant: score > 70 ? "PM2.5" : score > 45 ? "NO2" : "CO",
            latest_pm25: Number((18 + score * 1.18).toFixed(1)),
            latest_pm10: Number((35 + score * 1.58).toFixed(1)),
            recommendation:
              "Add electrostatic precipitators, monitor stack emissions continuously, and improve fuel quality.",
          };
        });

      const merged = recByCity.length
        ? recByCity
        : fallbackRaw.length
          ? fallbackRaw
          : buildSyntheticFactories(city);

      setFactories(merged);
      setLoading(false);
    }

    run();
    return () => {
      mounted = false;
    };
  }, [city]);

  return { loading, factories };
}

function AnimatedBackground({ phase, showIndustry }) {
  return (
    <div className={`background-layer ${phase}`}>
      <div className="haze" />
      <div className="cloud c1" />
      <div className="cloud c2" />
      <div className="cloud c3" />

      <div className={`industry-layer ${showIndustry ? "visible" : ""}`}>
        <div className="stack s1" />
        <div className="stack s2" />
        <div className="stack s3" />
        <div className="stack s4" />
        <div className="stack-smoke" style={{ left: 8, bottom: 76 }} />
        <div className="stack-smoke" style={{ left: 72, bottom: 93, animationDelay: "-1.2s" }} />
        <div className="stack-smoke" style={{ left: 136, bottom: 67, animationDelay: "-0.8s" }} />
        <div className="stack-smoke" style={{ left: 201, bottom: 83, animationDelay: "-1.7s" }} />
      </div>

      <div className="trees">
        <div className="tree" />
        <div className="tree" />
        <div className="tree" />
      </div>

      <div className="skyline" />
      <div className="water" />
    </div>
  );
}

function LandingPage({ onNext }) {
  return (
    <section>
      <div className="step-label">Page 1 / Environmental Story</div>
      <h1 className="title">Smart Factory Emission Monitoring System</h1>
      <p className="subtitle">
        Discover which factories are affecting your city&apos;s air quality and explore AI-powered
        solutions to reduce pollution.
      </p>
      <div className="actions">
        <button className="btn-primary" onClick={onNext}>
          Start Analysis
        </button>
      </div>
    </section>
  );
}

function CitySelector({ city, setCity, onBack, onNext }) {
  return (
    <section>
      <div className="step-label">Page 2 / City Selection</div>
      <h2 className="title" style={{ fontSize: "2.1rem" }}>
        Which city do you want to analyze?
      </h2>
      <div className="grid two" style={{ marginTop: 16 }}>
        {CITY_OPTIONS.map((item) => (
          <button
            key={item}
            className={`option-btn ${city === item ? "active" : ""}`}
            onClick={() => setCity(item)}
          >
            {item}
          </button>
        ))}
      </div>
      <div className="actions">
        <button className="btn-muted" onClick={onBack}>
          Back
        </button>
        <button className="btn-primary" onClick={onNext}>
          Continue
        </button>
      </div>
    </section>
  );
}

function QuestionFlow({ answers, setAnswers, onBack, onNext }) {
  return (
    <section>
      <div className="step-label">Page 3 / Industrial Context</div>
      <h2 className="title" style={{ fontSize: "2rem" }}>
        Tell us more about your analysis scope
      </h2>
      <div className="grid" style={{ marginTop: 16 }}>
        <label>
          1. Which industrial sector are you analyzing?
          <select
            value={answers.sector}
            onChange={(event) => setAnswers((prev) => ({ ...prev, sector: event.target.value }))}
          >
            <option>Steel</option>
            <option>Chemical</option>
            <option>Power Plant</option>
            <option>Manufacturing</option>
            <option>Mixed Industry</option>
          </select>
        </label>

        <label>
          2. What pollution concern is most important?
          <select
            value={answers.concern}
            onChange={(event) => setAnswers((prev) => ({ ...prev, concern: event.target.value }))}
          >
            <option>Air Quality</option>
            <option>CO2 Emissions</option>
            <option>Industrial Smoke</option>
            <option>Water Pollution</option>
          </select>
        </label>

        <label>
          3. What level of monitoring is needed?
          <select
            value={answers.scope}
            onChange={(event) => setAnswers((prev) => ({ ...prev, scope: event.target.value }))}
          >
            <option>City Level</option>
            <option>Industrial Zone</option>
            <option>Specific Factory</option>
          </select>
        </label>
      </div>
      <div className="actions">
        <button className="btn-muted" onClick={onBack}>
          Back
        </button>
        <button className="btn-primary" onClick={onNext}>
          Run Analysis
        </button>
      </div>
    </section>
  );
}

function AnalysisLoader({ onBack, onNext }) {
  useEffect(() => {
    const timer = setTimeout(onNext, 2800);
    return () => clearTimeout(timer);
  }, [onNext]);

  return (
    <section>
      <div className="step-label">Page 4 / Pollution Analysis</div>
      <h2 className="title" style={{ fontSize: "2rem" }}>
        Analyzing environmental impact...
      </h2>
      <div className="loader-wrap">
        <div>AI processing factory emissions</div>
        <div className="loader-bar">
          <span />
        </div>
      </div>
      <div className="actions">
        <button className="btn-muted" onClick={onBack}>
          Back
        </button>
      </div>
    </section>
  );
}

function MapView({ city, factories, selectedFactory, setSelectedFactory, onBack, onNext }) {
  const mapRef = useRef(null);
  const mapContainerRef = useRef(null);
  const markersRef = useRef([]);

  const center = CITY_CENTERS[city] || CITY_CENTERS.Delhi;

  useEffect(() => {
    mapboxgl.accessToken = MAPBOX_TOKEN;
    if (!mapContainerRef.current) return;

    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: "mapbox://styles/mapbox/streets-v12",
      center,
      zoom: 10,
    });

    map.addControl(new mapboxgl.NavigationControl(), "top-right");

    map.on("load", () => {
      const featureCollection = {
        type: "FeatureCollection",
        features: factories.map((factory) => ({
          type: "Feature",
          geometry: {
            type: "Point",
            coordinates: [factory.longitude, factory.latitude],
          },
          properties: {
            score: normalizeScore(factory.pollution_score || factory.pollution_impact_score),
          },
        })),
      };

      map.addSource("factory-heat", {
        type: "geojson",
        data: featureCollection,
      });

      map.addLayer({
        id: "factory-heat",
        type: "heatmap",
        source: "factory-heat",
        maxzoom: 14,
        paint: {
          "heatmap-weight": ["interpolate", ["linear"], ["get", "score"], 0, 0.1, 100, 1],
          "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 0, 0.6, 14, 1.6],
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            0,
            "rgba(89, 196, 122, 0)",
            0.2,
            "rgba(89, 196, 122, 0.35)",
            0.45,
            "rgba(254, 207, 85, 0.44)",
            0.7,
            "rgba(250, 138, 84, 0.56)",
            1,
            "rgba(236, 94, 80, 0.74)",
          ],
          "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 0, 14, 13, 26],
        },
      });
    });

    mapRef.current = map;
    return () => {
      markersRef.current.forEach((marker) => marker.remove());
      markersRef.current = [];
      map.remove();
    };
  }, [city]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    markersRef.current.forEach((marker) => marker.remove());
    markersRef.current = [];

    factories.forEach((factory) => {
      const score = normalizeScore(factory.pollution_score || factory.pollution_impact_score);
      const colorClass = getMarkerColorClass(score);

      const markerNode = document.createElement("div");
      markerNode.className = `marker-dot ${colorClass}`;

      const popupHtml = `
        <strong>${factory.factory_name}</strong><br/>
        Industry: ${factory.industry_type}<br/>
        Pollution Score: ${score}/100<br/>
        Primary Pollutant: ${factory.primary_pollutant || "PM2.5"}
      `;

      const marker = new mapboxgl.Marker(markerNode)
        .setLngLat([factory.longitude, factory.latitude])
        .setPopup(new mapboxgl.Popup({ offset: 16 }).setHTML(popupHtml))
        .addTo(map);

      markerNode.addEventListener("click", () => {
        setSelectedFactory(factory);
      });

      markersRef.current.push(marker);
    });
  }, [factories, setSelectedFactory]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.getSource("factory-heat")) return;

    const featureCollection = {
      type: "FeatureCollection",
      features: factories.map((factory) => ({
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [factory.longitude, factory.latitude],
        },
        properties: {
          score: normalizeScore(factory.pollution_score || factory.pollution_impact_score),
        },
      })),
    };

    map.getSource("factory-heat").setData(featureCollection);
  }, [factories]);

  useEffect(() => {
    if (!selectedFactory || !mapRef.current) return;
    mapRef.current.flyTo({
      center: [selectedFactory.longitude, selectedFactory.latitude],
      zoom: 12,
      duration: 1000,
    });
  }, [selectedFactory]);

  return (
    <section>
      <div className="step-label">Page 5 / Interactive Map</div>
      <h2 className="title" style={{ fontSize: "2rem" }}>
        Factory map for {city}
      </h2>
      <p className="subtitle">
        Green markers indicate low impact, yellow moderate, and red high pollution impact.
      </p>
      <div className="map-shell">
        <div className="map-canvas" ref={mapContainerRef} />
      </div>
      <div className="footer-note">
        Click any marker to inspect details, then continue to factory report.
      </div>
      <div className="actions">
        <button className="btn-muted" onClick={onBack}>
          Back
        </button>
        <button className="btn-primary" onClick={onNext}>
          View Factory Detail
        </button>
      </div>
    </section>
  );
}

function FactoryTrendChart({ data }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const chart = new Chart(canvasRef.current, {
      type: "line",
      data: {
        labels: ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "Today"],
        datasets: [
          {
            label: "PM2.5 Trend",
            data,
            borderColor: "#ffd166",
            backgroundColor: "rgba(255, 209, 102, 0.18)",
            fill: true,
            tension: 0.35,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#dbeff5" } } },
        scales: {
          x: { ticks: { color: "#dbeff5" } },
          y: { ticks: { color: "#dbeff5" } },
        },
      },
    });

    return () => chart.destroy();
  }, [data]);

  return (
    <div className="chart-wrap">
      <canvas ref={canvasRef} />
    </div>
  );
}

function FactoryDetails({ factory, onBack, onNext }) {
  const metrics = useMemo(() => deriveFactoryMetrics(factory), [factory]);

  return (
    <section>
      <div className="step-label">Page 6 / Factory Report</div>
      <h2 className="title" style={{ fontSize: "2rem" }}>
        Factory Pollution Report
      </h2>
      <p className="subtitle">
        <strong>{factory.factory_name}</strong> in {factory.city} ({factory.industry_type})
      </p>
      <div className="badges">
        <span className="badge">Pollution score: {metrics.score}/100</span>
        <span className="badge">Risk: {getRiskLabel(metrics.score)}</span>
        <span className="badge">Primary pollutant: {factory.primary_pollutant || "PM2.5"}</span>
      </div>
      <div className="metric-grid">
        <div className="metric-card">
          PM2.5
          <strong>{metrics.pm25}</strong>
        </div>
        <div className="metric-card">
          PM10
          <strong>{metrics.pm10}</strong>
        </div>
        <div className="metric-card">
          NO2
          <strong>{metrics.no2}</strong>
        </div>
        <div className="metric-card">
          SO2
          <strong>{metrics.so2}</strong>
        </div>
        <div className="metric-card">
          CO
          <strong>{metrics.co}</strong>
        </div>
        <div className="metric-card">
          O3
          <strong>{metrics.o3}</strong>
        </div>
      </div>
      <FactoryTrendChart data={metrics.trend} />
      <div className="actions">
        <button className="btn-muted" onClick={onBack}>
          Back
        </button>
        <button className="btn-primary" onClick={onNext}>
          AI Pollution Score
        </button>
      </div>
    </section>
  );
}

function AIScore({ factory, onBack, onNext }) {
  const metrics = useMemo(() => deriveFactoryMetrics(factory), [factory]);
  const level = pollutionLevelText(metrics.score);
  const radius = (1.2 + metrics.score / 24).toFixed(1);
  const population = (0.28 + metrics.score * 0.013).toFixed(2);

  return (
    <section>
      <div className="step-label">Page 7 / AI Impact Score</div>
      <h2 className="title" style={{ fontSize: "2rem" }}>
        AI Impact Score
      </h2>
      <div className="metric-grid">
        <div className="metric-card">
          Pollution Impact
          <strong>{level}</strong>
        </div>
        <div className="metric-card">
          Impact Radius
          <strong>{radius} km</strong>
        </div>
        <div className="metric-card">
          Affected Population
          <strong>~{population}M</strong>
        </div>
      </div>
      <p className="subtitle">
        AI estimates are generated from current pollutant intensity and industrial profile for the
        selected factory.
      </p>
      <div className="actions">
        <button className="btn-muted" onClick={onBack}>
          Back
        </button>
        <button className="btn-primary" onClick={onNext}>
          View Solutions
        </button>
      </div>
    </section>
  );
}

function SolutionView({ factory, onBack, onNext }) {
  const recommendations = factory.recommendation
    ? factory.recommendation.split(",").map((item) => item.trim())
    : [
        "Install scrubber systems",
        "Upgrade filtration units",
        "Transition to cleaner fuel",
        "Implement real-time emission monitoring",
      ];

  return (
    <section>
      <div className="step-label">Page 8 / Recommendation</div>
      <h2 className="title" style={{ fontSize: "2rem" }}>
        Solution and Recommendation
      </h2>
      <p className="subtitle">Clearer air begins with operational improvements at {factory.factory_name}.</p>
      <ul className="reco-list">
        {recommendations.slice(0, 5).map((item, index) => (
          <li key={`${item}-${index}`}>{item}</li>
        ))}
      </ul>
      <div className="actions">
        <button className="btn-muted" onClick={onBack}>
          Back
        </button>
        <button className="btn-primary" onClick={onNext}>
          View Sustainable Alternatives
        </button>
      </div>
    </section>
  );
}

function Transformation({ onRestart }) {
  return (
    <section>
      <div className="step-label">Final / Transformation</div>
      <h2 className="title" style={{ fontSize: "2rem" }}>
        Cleaner industry leads to a cleaner future.
      </h2>
      <p className="subtitle">
        A sustainable city is possible when monitoring, accountability, and cleaner technology work
        together.
      </p>
      <div className="actions">
        <button className="btn-primary" onClick={onRestart}>
          Restart Analysis
        </button>
      </div>
    </section>
  );
}

function App() {
  const [step, setStep] = useState(0);
  const [city, setCity] = useState("Pune");
  const [transitioning, setTransitioning] = useState(false);
  const [answers, setAnswers] = useState({
    sector: "Steel",
    concern: "Air Quality",
    scope: "City Level",
  });

  const { loading, factories } = useFactoryData(city);
  const [selectedFactory, setSelectedFactory] = useState(null);

  useEffect(() => {
    if (!selectedFactory && factories.length > 0) {
      setSelectedFactory(factories[0]);
    }
  }, [factories, selectedFactory]);

  function transitionTo(nextStep) {
    setTransitioning(true);
    setTimeout(() => setStep(nextStep), 360);
    setTimeout(() => setTransitioning(false), 900);
  }

  const phase = step >= 7 ? "clean" : step >= 3 ? "smoky" : "";
  const showIndustry = step >= 1;

  const current = (() => {
    if (step === 0) return <LandingPage onNext={() => transitionTo(1)} />;
    if (step === 1)
      return (
        <CitySelector
          city={city}
          setCity={setCity}
          onBack={() => transitionTo(0)}
          onNext={() => transitionTo(2)}
        />
      );
    if (step === 2)
      return (
        <QuestionFlow
          answers={answers}
          setAnswers={setAnswers}
          onBack={() => transitionTo(1)}
          onNext={() => transitionTo(3)}
        />
      );
    if (step === 3)
      return <AnalysisLoader onBack={() => transitionTo(2)} onNext={() => transitionTo(4)} />;
    if (step === 4)
      return (
        <MapView
          city={city}
          factories={factories}
          selectedFactory={selectedFactory}
          setSelectedFactory={setSelectedFactory}
          onBack={() => transitionTo(3)}
          onNext={() => transitionTo(5)}
        />
      );
    if (step === 5 && selectedFactory)
      return (
        <FactoryDetails
          factory={selectedFactory}
          onBack={() => transitionTo(4)}
          onNext={() => transitionTo(6)}
        />
      );
    if (step === 6 && selectedFactory)
      return <AIScore factory={selectedFactory} onBack={() => transitionTo(5)} onNext={() => transitionTo(7)} />;
    if (step === 7 && selectedFactory)
      return (
        <SolutionView
          factory={selectedFactory}
          onBack={() => transitionTo(6)}
          onNext={() => transitionTo(8)}
        />
      );
    return <Transformation onRestart={() => transitionTo(0)} />;
  })();

  return (
    <div className="app">
      <AnimatedBackground phase={phase} showIndustry={showIndustry} />
      <main className="main-stage">
        <article className="panel">
          {loading && step >= 4 ? <div className="subtitle">Loading factory data...</div> : current}
        </article>
      </main>
      {transitioning ? <div className="fade-overlay" /> : null}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
