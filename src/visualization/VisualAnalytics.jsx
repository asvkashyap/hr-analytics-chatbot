import React, { useState, useRef } from "react";

/*─────────────────────────────────────────────────────────────────────────────
  VisualAnalytics.jsx
  Dynamic chart generator — sends query to /api/visualize,
  Groq interprets the query against real CSV data and returns
  a Chart.js config which we render directly. No hardcoded charts.
─────────────────────────────────────────────────────────────────────────────*/

const SUGGESTIONS = [
  "Show recognition category distribution",
  "Which subcategories are most common?",
  "Compare give vs receive by role group",
  "Show recognition gaps — least recognized behaviors",
  "Monthly recognition trend over time",
  "What does Engineering get recognized for?",
  "Show a scatter plot of category count vs subcategory count",
  "Which role groups are most isolated?",
];

// ---------------------------------------------------------------------------
// Chart.js — load once globally
// ---------------------------------------------------------------------------
let _chartJsReady = null;
function loadChartJs() {
  if (_chartJsReady) return _chartJsReady;
  _chartJsReady = new Promise((resolve) => {
    if (window.Chart) { resolve(); return; }
    const s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js";
    s.onload = resolve;
    document.head.appendChild(s);
  });
  return _chartJsReady;
}

// ---------------------------------------------------------------------------
// ChartRenderer — destroys old instance, mounts new one from Groq config
// ---------------------------------------------------------------------------
function ChartRenderer({ config }) {
  const canvasRef = useRef(null);
  const instanceRef = useRef(null);

  React.useEffect(() => {
    if (!config) return;
    loadChartJs().then(() => {
      if (!canvasRef.current) return;
      if (instanceRef.current) { instanceRef.current.destroy(); instanceRef.current = null; }
      try {
        instanceRef.current = new window.Chart(canvasRef.current, config);
      } catch (e) {
        console.error("Chart render error:", e);
      }
    });
    return () => {
      if (instanceRef.current) { instanceRef.current.destroy(); instanceRef.current = null; }
    };
  }, [config]);

  if (!config) return null;

  const isHorizontal = config.options?.indexAxis === "y";
  const labelCount = config.data?.labels?.length || 6;
  const height = isHorizontal ? Math.max(300, labelCount * 42 + 80) : 320;

  return (
    <div style={{ position: "relative", width: "100%", height: `${height}px` }}>
      <canvas ref={canvasRef} role="img" aria-label={config._title || "HR analytics chart"} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// LoadingSkeleton
// ---------------------------------------------------------------------------
function LoadingSkeleton({ message }) {
  return (
    <div style={{ padding: "24px 0" }}>
      <div style={{ fontSize: 12, color: "#9ca3af", marginBottom: 18, display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ display: "flex", gap: 4 }}>
          {[0, 1, 2].map(i => (
            <div key={i} style={{
              width: 6, height: 6, borderRadius: "50%", background: "#7F77DD",
              animation: `vapulse 1.2s ease-in-out ${i * 0.2}s infinite`,
            }} />
          ))}
        </div>
        {message}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {[90, 65, 80, 50, 72].map((w, i) => (
          <div key={i} style={{
            height: 10, width: `${w}%`, borderRadius: 6,
            background: "rgba(127,119,221,0.1)",
            animation: `vapulse 1.4s ease-in-out ${i * 0.15}s infinite`,
          }} />
        ))}
      </div>
      <style>{`@keyframes vapulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MetricCards
// ---------------------------------------------------------------------------
function MetricCards({ metrics }) {
  if (!metrics || metrics.length === 0) return null;
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
      {metrics.map((m, i) => (
        <div key={i} style={{
          background: "#f9fafb", borderRadius: 10, padding: "10px 14px", minWidth: 90, flex: 1,
        }}>
          <div style={{ fontSize: 11, color: "#9ca3af", marginBottom: 3 }}>{m.label}</div>
          <div style={{ fontSize: 18, fontWeight: 500, color: "#111827" }}>{m.value}</div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------
function Legend({ config }) {
  if (!config) return null;
  const datasets = config.data?.datasets || [];
  const type = config.type;
  if (datasets.length <= 1 && type !== "doughnut" && type !== "pie") return null;

  let items = [];
  if (type === "doughnut" || type === "pie") {
    const labels = config.data?.labels || [];
    const colors = datasets[0]?.backgroundColor || [];
    items = labels.map((l, i) => ({ label: l, color: Array.isArray(colors) ? colors[i] : colors }));
  } else {
    items = datasets.map(d => ({
      label: d.label,
      color: Array.isArray(d.backgroundColor) ? d.backgroundColor[0] : (d.backgroundColor || d.borderColor),
    }));
  }

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 10, fontSize: 12, color: "#6b7280" }}>
      {items.map((item, i) => (
        <span key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: item.color, flexShrink: 0 }} />
          {String(item.label).length > 35 ? String(item.label).slice(0, 35) + "…" : item.label}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main VisualAnalytics component
// ---------------------------------------------------------------------------
export function VisualAnalytics({ stats }) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("Analyzing your question...");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const runQuery = async (q) => {
    const text = (q || query).trim();
    if (!text || loading) return;
    setQuery(text);
    setLoading(true);
    setError(null);
    setResult(null);

    const msgs = [
      "Analyzing your question...",
      "Reading recognition data...",
      "Asking Groq to build the chart...",
      "Rendering visualization...",
    ];
    let idx = 0;
    setLoadingMsg(msgs[0]);
    const timer = setInterval(() => {
      idx = Math.min(idx + 1, msgs.length - 1);
      setLoadingMsg(msgs[idx]);
    }, 900);

    try {
      const res = await fetch("/api/visualize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "#f9fafb", overflow: "hidden" }}>

      {/* ── Query bar ── */}
      <div style={{
        padding: "14px 20px 12px", borderBottom: "0.5px solid rgba(0,0,0,0.07)",
        background: "white", flexShrink: 0,
      }}>
        <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && runQuery()}
            placeholder="Ask anything — e.g. show a radar chart of subcategory spread, or trend over time..."
            disabled={loading}
            style={{
              flex: 1, height: 36, padding: "0 14px",
              border: "0.5px solid rgba(0,0,0,0.12)", borderRadius: 8,
              fontSize: 13, fontFamily: "inherit", outline: "none",
              background: loading ? "#f9fafb" : "white", color: "#111827",
            }}
          />
          <button
            onClick={() => runQuery()}
            disabled={loading || !query.trim()}
            style={{
              height: 36, padding: "0 16px", borderRadius: 8,
              background: loading || !query.trim() ? "#d1d5db" : "#7F77DD",
              border: "none", color: "white",
              fontSize: 13, fontWeight: 500,
              cursor: loading || !query.trim() ? "default" : "pointer",
              fontFamily: "inherit", transition: "background 0.15s",
            }}
          >
            {loading ? "Generating..." : "Generate"}
          </button>
        </div>

        {/* Suggestion chips */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {SUGGESTIONS.map((s, i) => (
            <button
              key={i}
              onClick={() => runQuery(s)}
              disabled={loading}
              style={{
                padding: "4px 10px", borderRadius: 20, fontSize: 11,
                border: "0.5px solid rgba(0,0,0,0.1)",
                background: "white", color: "#6b7280",
                cursor: loading ? "default" : "pointer",
                fontFamily: "inherit",
              }}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* ── Chart area ── */}
      <div style={{ flex: 1, overflowY: "auto", padding: "20px 20px 32px" }}>

        {/* Empty state */}
        {!loading && !result && !error && (
          <div style={{
            display: "flex", flexDirection: "column", alignItems: "center",
            justifyContent: "center", height: "80%", gap: 12,
            color: "#9ca3af", textAlign: "center", padding: "40px 20px",
          }}>
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" style={{ opacity: 0.35 }}>
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <path d="M3 9h18M9 21V9"/><path d="M7 15h4M7 18h6"/>
            </svg>
            <div style={{ fontSize: 14, fontWeight: 500, color: "#6b7280" }}>Ask anything about your recognition data</div>
            <div style={{ fontSize: 12 }}>Groq picks the best chart type and builds it from your real CSV — not hardcoded</div>
          </div>
        )}

        {/* Loading */}
        {loading && <LoadingSkeleton message={loadingMsg} />}

        {/* Error */}
        {error && !loading && (
          <div style={{
            padding: "16px 20px", borderRadius: 10,
            background: "#fef2f2", border: "0.5px solid #fca5a5",
            fontSize: 13, color: "#dc2626",
          }}>
            <strong style={{ fontWeight: 500 }}>Error: </strong>{error}
            <div style={{ marginTop: 6, fontSize: 12, color: "#ef4444" }}>
              Make sure the FastAPI server is running and GROQ_API_KEY is set in your .env
            </div>
          </div>
        )}

        {/* Result */}
        {result && !loading && (
          <div>
            {/* Title + badge */}
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
              <span style={{ fontSize: 15, fontWeight: 500, color: "#111827" }}>
                {result.title || "Chart"}
              </span>
              <span style={{
                fontSize: 11, padding: "2px 8px", borderRadius: 20,
                background: "#eeecfc", color: "#7F77DD",
              }}>
                {result.chart_type || "chart"} · {result.record_count || "—"} records
              </span>
            </div>

            <MetricCards metrics={result.metrics} />
            <Legend config={result.chart_config} />
            <ChartRenderer config={result.chart_config} />

            {result.insight && (
              <div style={{
                marginTop: 14, padding: "12px 16px",
                background: "white", borderRadius: 10,
                border: "0.5px solid rgba(0,0,0,0.08)",
                fontSize: 13, color: "#4b5563", lineHeight: 1.65,
              }}>
                {result.insight}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
