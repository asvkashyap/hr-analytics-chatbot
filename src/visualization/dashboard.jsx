import React, { useState, useRef, useEffect, useCallback } from "react";
import { VisualAnalytics } from "./VisualAnalytics.jsx";
/*─────────────────────────────────────────────────────────────────────────────
  Recognition Intelligence — Executive Briefing Dashboard
  Connects to FastAPI backend: /api/chat, /api/stats, /api/clear
  Tableau Pulse-inspired insight cards, chat-first layout
─────────────────────────────────────────────────────────────────────────────*/

const API_BASE = "/api";

const ROUTE_COLORS = {
  ANALYTICAL:   { bg: "#fef3c7", fg: "#92400e" },
  SEMANTIC:     { bg: "#e0e7ff", fg: "#3730a3" },
  HYBRID:       { bg: "#fce7f3", fg: "#9d174d" },
  SUBSTITUTION: { bg: "#d1fae5", fg: "#065f46" },
  GRAPH:        { bg: "#ede9fe", fg: "#5b21b6" },
  DIRECT:       { bg: "#f1f5f9", fg: "#475569" },
};

const THEME_COLORS = ["#6366f1", "#f59e0b", "#10b981", "#f43f5e"];

const SUGGESTIONS = [
  { icon: "◎", label: "Theme overview",     q: "What are people most recognized for?" },
  { icon: "△", label: "Recognition gaps",    q: "Where are the recognition gaps?" },
  { icon: "⇄", label: "Give vs receive",     q: "Which roles give the most recognition?" },
  { icon: "⬡", label: "Team connections",    q: "How connected are teams?" },
  { icon: "◇", label: "Engineering focus",   q: "What does Engineering get recognized for?" },
  { icon: "↻", label: "Role substitution",   q: "Who could replace a Senior Director?" },
];

const FOLLOW_UPS = {
  HYBRID:       ["Which teams are isolated?", "Show mutual recognition pairs", "Deep dive on top flow"],
  SEMANTIC:     ["Show me specific examples", "Compare with other themes", "Which roles drive this?"],
  ANALYTICAL:   ["Break down by subcategory", "Show trend over time", "Compare role groups"],
  SUBSTITUTION: ["Compare their recognition profiles", "What skills overlap?", "Show recognition gaps"],
  GRAPH:        ["Show bridge roles", "Strongest mutual pairs", "Most isolated group"],
  DIRECT:       [],
};

/* ── Helpers ───────────────────────────────────────────────────────────── */

function parseMarkdown(text) {
  if (!text) return "";
  let h = text
    .replace(/\*\*(.+?)\*\*/g, '<strong style="font-weight:500;color:var(--color-text-primary)">$1</strong>')
    .replace(/\(Relevance\s+([\d.]+%?)\)/g,
      '<span style="display:inline-block;background:rgba(99,102,241,0.1);color:#6366f1;font-size:11px;padding:1px 6px;border-radius:8px;margin-left:2px;font-weight:500">$1</span>');
  const lines = h.split("\n");
  let result = [];
  let inList = false;
  for (const line of lines) {
    const bullet = line.match(/^\s*[•*-]\s+(.+)$/);
    if (bullet) {
      if (!inList) { result.push("<ul>"); inList = true; }
      result.push(`<li>${bullet[1]}</li>`);
    } else {
      if (inList) { result.push("</ul>"); inList = false; }
      if (line.trim()) result.push(`<p>${line}</p>`);
    }
  }
  if (inList) result.push("</ul>");
  return result.join("");
}

function extractKeyMetric(text, route) {
  if (!text) return null;

  if (route === "SUBSTITUTION") {
    const matches = [...text.matchAll(/(\d{1,3}(?:\.\d+)?)\s*%\s*(?:match|similarity)/gi)];
    if (matches.length > 0) return { value: matches[0][1] + "%", label: "Top match score" };
    return null;
  }

  if (route === "GRAPH") {
    const numMatch = text.match(/(\d{2,5})\s*(?:recognitions|connections|edges)/i);
    if (numMatch) return { value: numMatch[1], label: "Connections mapped" };
    return null;
  }

  if (route === "ANALYTICAL" || route === "HYBRID") {
    const firstChunk = text.substring(0, 200);
    const headlinePct = firstChunk.match(/(\d{1,3}(?:\.\d+)?)\s*%\s+of\s+(\w[\w\s]{3,20})/i);
    if (headlinePct) {
      return { value: headlinePct[1] + "%", label: headlinePct[2].trim() };
    }
    const headlineCount = firstChunk.match(/(\d{3,5})\s+(records|awards|recognitions)/i);
    if (headlineCount) {
      return { value: parseInt(headlineCount[1]).toLocaleString(), label: headlineCount[2] };
    }
  }

  return null;
}

function extractDrivers(text) {
  if (!text) return [];
  const drivers = [];
  const patterns = [
    /(?:^|\n)\s*[•*-]\s*\*?\*?(.+?)\*?\*?\s*(?::|—|–)\s*(\d+)\s/gm,
    /(\w[\w\s&/]+?)\s*→\s*(\w[\w\s&/]+?)\s*(?::|—|–|\()\s*(\d+)/g,
  ];
  for (const p of patterns) {
    let m;
    while ((m = p.exec(text)) !== null && drivers.length < 5) {
      if (m[3]) drivers.push({ label: `${m[1].trim()} → ${m[2].trim()}`, value: parseInt(m[3]) });
      else drivers.push({ label: m[1].trim(), value: parseInt(m[2]) });
    }
  }
  return drivers;
}

/* ── Components ────────────────────────────────────────────────────────── */

function IconButton({ children, active, onClick, title }) {
  return (
    <button onClick={onClick} title={title} style={{
      width: 36, height: 36, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center",
      color: active ? "#6366f1" : "#9ca3af", background: active ? "rgba(99,102,241,0.1)" : "transparent",
      border: "none", cursor: "pointer", transition: "all 0.15s",
    }}>
      {children}
    </button>
  );
}

function RouteBadge({ route }) {
  const c = ROUTE_COLORS[route] || ROUTE_COLORS.DIRECT;
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, padding: "2px 7px", borderRadius: 4,
      background: c.bg, color: c.fg, letterSpacing: "0.04em",
    }}>{route?.toLowerCase()}</span>
  );
}

function CragScore({ crag }) {
  if (!crag) return null;
  const score = typeof crag === "object" ? crag.score : crag;
  const numScore = typeof score === "number" ? score : parseFloat(score) || 0;
  const passed = numScore >= 0.25;
  return (
    <span style={{
      fontSize: 10, color: passed ? "#059669" : "#d97706", fontWeight: 500,
    }}>crag {(numScore * 100).toFixed(0)}</span>
  );
}

function TraceToggle({ trace }) {
  const [open, setOpen] = useState(false);
  if (!trace || trace.length === 0) return null;
  return (
    <>
      <button onClick={() => setOpen(!open)} style={{
        fontSize: 10, color: "#9ca3af", background: "none", border: "none", cursor: "pointer",
        fontFamily: "inherit", textDecoration: "underline", textDecorationStyle: "dotted",
        textUnderlineOffset: 2,
      }}>{open ? "hide trace" : "execution trace"}</button>
      {open && (
        <div style={{
          marginTop: 6, padding: "8px 12px", background: "rgba(0,0,0,0.03)", borderRadius: 8,
          fontSize: 11, color: "#6b7280", lineHeight: 1.6, fontFamily: "monospace",
          maxHeight: 160, overflowY: "auto",
        }}>
          {trace.map((t, i) => <div key={i}>{typeof t === "string" ? t : JSON.stringify(t)}</div>)}
        </div>
      )}
    </>
  );
}

function DriversSection({ drivers }) {
  if (!drivers || drivers.length === 0) return null;
  const max = Math.max(...drivers.map(d => d.value));
  return (
    <div style={{
      borderTop: "0.5px solid rgba(0,0,0,0.06)", padding: "12px 20px",
      display: "flex", flexDirection: "column", gap: 8,
    }}>
      <span style={{ fontSize: 11, color: "#9ca3af", fontWeight: 500, letterSpacing: "0.04em" }}>Key drivers</span>
      {drivers.map((d, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 12, color: "#6b7280", minWidth: 140 }}>{d.label}</span>
          <div style={{ flex: 1, height: 6, background: "rgba(0,0,0,0.04)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{
              height: "100%", width: `${(d.value / max) * 100}%`, borderRadius: 3,
              background: `rgba(99,102,241,${1 - i * 0.15})`,
            }} />
          </div>
          <span style={{ fontSize: 12, fontWeight: 500, color: "#374151", minWidth: 28, textAlign: "right" }}>{d.value}</span>
        </div>
      ))}
    </div>
  );
}

function FollowUpChips({ route, onSend }) {
  const chips = FOLLOW_UPS[route] || [];
  if (chips.length === 0) return null;
  return (
    <div style={{
      borderTop: "0.5px solid rgba(0,0,0,0.06)", padding: "10px 20px",
      display: "flex", gap: 6, flexWrap: "wrap",
    }}>
      {chips.map((c, i) => (
        <button key={i} onClick={() => onSend(c)} style={{
          fontSize: 11, padding: "4px 12px", borderRadius: 12,
          background: "rgba(0,0,0,0.03)", border: "0.5px solid rgba(0,0,0,0.08)",
          color: "#6b7280", cursor: "pointer", fontFamily: "inherit",
        }}>{c}</button>
      ))}
    </div>
  );
}

function PulseCard({ msg, onSendFollowUp }) {
  const { answer, route, tools_used, confidence, trace, crag } = msg;
  const metric = extractKeyMetric(answer, route);
  const drivers = extractDrivers(answer);
  const isSubstitution = route === "SUBSTITUTION";

  return (
    <div style={{ alignSelf: "flex-start", maxWidth: 580, display: "flex", flexDirection: "column", gap: 0 }}>
      <div style={{
        background: "white", border: "0.5px solid rgba(0,0,0,0.08)",
        borderLeft: isSubstitution ? "3px solid #10b981" : undefined,
        borderRadius: isSubstitution ? "0 12px 12px 0" : 12,
        overflow: "hidden",
      }}>
        {metric && (
          <div style={{ padding: "16px 20px 8px", display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
            <div>
              <div style={{ fontSize: 12, color: "#9ca3af", fontWeight: 500, letterSpacing: "0.03em" }}>{metric.label}</div>
              <div style={{ fontSize: 32, fontWeight: 500, color: "#111827", lineHeight: 1.1, marginTop: 2 }}>{metric.value}</div>
            </div>
            <svg width="72" height="32" viewBox="0 0 72 32" style={{ flexShrink: 0, marginTop: 4 }}>
              <defs>
                <linearGradient id="spk" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0" stopColor="#6366f1" stopOpacity="0.12" />
                  <stop offset="1" stopColor="#6366f1" stopOpacity="0" />
                </linearGradient>
              </defs>
              <polyline points="0,26 10,22 20,18 30,20 40,12 50,9 60,5 72,7" fill="none" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" />
              <polygon points="0,26 10,22 20,18 30,20 40,12 50,9 60,5 72,7 72,32 0,32" fill="url(#spk)" />
            </svg>
          </div>
        )}

        <div style={{ padding: metric ? "4px 20px 16px" : "16px 20px" }}>
          <div
            style={{ fontSize: 13, color: "#4b5563", lineHeight: 1.65 }}
            dangerouslySetInnerHTML={{ __html: parseMarkdown(answer) }}
          />
        </div>

        <DriversSection drivers={drivers} />
        <FollowUpChips route={route} onSend={onSendFollowUp} />
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 4px 0", flexWrap: "wrap" }}>
        {route && <RouteBadge route={route} />}
        {tools_used && tools_used.length > 0 && (
          <span style={{ fontSize: 10, color: "#9ca3af" }}>{tools_used.join(" + ")}</span>
        )}
        {crag && <><span style={{ fontSize: 10, color: "#d1d5db" }}>·</span><CragScore crag={crag} /></>}
        {confidence && (
          <><span style={{ fontSize: 10, color: "#d1d5db" }}>·</span>
          <span style={{ fontSize: 10, color: "#9ca3af" }}>{confidence.toLowerCase()}</span></>
        )}
        {trace && trace.length > 0 && (
          <><span style={{ fontSize: 10, color: "#d1d5db" }}>·</span><TraceToggle trace={trace} /></>
        )}
      </div>
    </div>
  );
}

function UserBubble({ text }) {
  return (
    <div style={{
      alignSelf: "flex-end", maxWidth: 400, padding: "10px 16px",
      background: "#6366f1", color: "#fff", borderRadius: "16px 16px 4px 16px",
      fontSize: 14, lineHeight: 1.5,
    }}>{text}</div>
  );
}

function EmptyState({ onSend }) {
  return (
    <div style={{
      flex: 1, display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", gap: 24, padding: "40px 20px",
    }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 18, fontWeight: 500, color: "#111827", marginBottom: 4 }}>Recognition Intelligence</div>
        <div style={{ fontSize: 13, color: "#9ca3af" }}>Ask anything about recognition patterns, team dynamics, or culture signals</div>
      </div>
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8,
        maxWidth: 480, width: "100%",
      }}>
        {SUGGESTIONS.map((s, i) => (
          <button key={i} onClick={() => onSend(s.q)} style={{
            display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 4,
            padding: "12px 14px", borderRadius: 10,
            background: "white", border: "0.5px solid rgba(0,0,0,0.08)",
            cursor: "pointer", fontFamily: "inherit", textAlign: "left",
            transition: "border-color 0.15s",
          }}
          onMouseEnter={e => e.currentTarget.style.borderColor = "rgba(99,102,241,0.3)"}
          onMouseLeave={e => e.currentTarget.style.borderColor = "rgba(0,0,0,0.08)"}
          >
            <span style={{ fontSize: 16 }}>{s.icon}</span>
            <span style={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>{s.label}</span>
            <span style={{ fontSize: 11, color: "#9ca3af", lineHeight: 1.4 }}>{s.q}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function SignalCard({ label, value, detail, color }) {
  return (
    <div style={{ padding: "10px 12px", borderRadius: 8, background: "rgba(0,0,0,0.02)" }}>
      <div style={{ fontSize: 11, fontWeight: 500, color, marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 500, color: "#111827" }}>{value}</div>
      {detail && <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 1 }}>{detail}</div>}
    </div>
  );
}

function ContextPanel({ stats }) {
  const categories = stats?.categories || [
    { name: "Team Enablement", pct: 44.2 },
    { name: "Ops Excellence", pct: 34.0 },
    { name: "Strategic Outcomes", pct: 16.4 },
    { name: "Creative & Brand", pct: 5.3 },
  ];

  const totalRecords = stats?.total_records || 999;

  return (
    <div style={{
      width: 220, flexShrink: 0, borderLeft: "0.5px solid rgba(0,0,0,0.08)",
      background: "white", padding: "20px 14px",
      display: "flex", flexDirection: "column", gap: 18, overflowY: "auto",
    }}>
      <div>
        <div style={{ fontSize: 11, fontWeight: 500, color: "#9ca3af", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 8 }}>
          Today's pulse
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <SignalCard label="Strongest bond" value="Eng ↔ Eng" detail="158 recognitions" color="#6366f1" />
          <SignalCard label="Most central hub" value="Marketing" detail="Bridges 8 role groups" color="#f59e0b" />
          <SignalCard label="Needs attention" value="Creative & Brand" detail="Only 5.3% of recognition" color="#f43f5e" />
        </div>
      </div>

      <div style={{ height: "0.5px", background: "rgba(0,0,0,0.06)" }} />

      <div>
        <div style={{ fontSize: 11, fontWeight: 500, color: "#9ca3af", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 8 }}>
          Themes
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          {categories.map((c, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "3px 0" }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: THEME_COLORS[i] || "#9ca3af", flexShrink: 0 }} />
              <span style={{ fontSize: 12, color: "#6b7280", flex: 1 }}>{c.name}</span>
              <span style={{ fontSize: 12, fontWeight: 500, color: "#111827" }}>{c.pct}%</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ height: "0.5px", background: "rgba(0,0,0,0.06)" }} />

      <div>
        <div style={{ fontSize: 11, fontWeight: 500, color: "#9ca3af", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 8 }}>
          Data scope
        </div>
        {[
          ["Records", totalRecords],
          ["Role groups", stats?.role_groups || 11],
          ["Behaviors", stats?.subcategories || 26],
        ].map(([label, val], i) => (
          <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0" }}>
            <span style={{ fontSize: 12, color: "#9ca3af" }}>{label}</span>
            <span style={{ fontSize: 12, fontWeight: 500, color: "#111827" }}>{val}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Main App ──────────────────────────────────────────────────────────── */

export default function RecognitionIntelligence() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [activeTab, setActiveTab] = useState("chat");
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/stats`)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!d) return;
        let records = [];
        if (Array.isArray(d)) {
          records = d;
        } else {
          for (const key of Object.keys(d)) {
            if (Array.isArray(d[key]) && d[key].length > 10) {
              records = d[key];
              break;
            }
          }
        }
        if (records.length > 0) {
          const counts = {};
          records.forEach(r => {
            const cat = r.category_short || r.category_name || "Unknown";
            counts[cat] = (counts[cat] || 0) + 1;
          });
          const total = records.length;
          const categories = Object.entries(counts)
            .map(([name, count]) => ({ name, pct: Math.round(count / total * 1000) / 10 }))
            .sort((a, b) => b.pct - a.pct);
          setStats({ categories, total_records: total });
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return;
    const userMsg = { role: "user", content: text.trim() };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text.trim(), session_id: "dashboard" }),
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();

      setMessages(prev => [...prev, {
        role: "assistant",
        answer: data.answer || data.response || "No response received.",
        route: data.route,
        tools_used: data.tools_used,
        confidence: data.confidence,
        trace: data.trace,
        crag: data.crag,
        chart_type: data.chart_type,
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: "assistant",
        answer: `Something went wrong: ${err.message}. Check that the FastAPI server is running on port 8000.`,
        route: "DIRECT",
      }]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [loading]);

  const handleClear = async () => {
    setMessages([]);
    try { await fetch(`${API_BASE}/clear`, { method: "POST" }); } catch {}
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div style={{
      height: "100vh", display: "flex", flexDirection: "column",
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      background: "#f9fafb", color: "#111827",
    }}>
      {/* Top bar */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "10px 20px", borderBottom: "0.5px solid rgba(0,0,0,0.08)",
        background: "white", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <span style={{ fontSize: 15, fontWeight: 500 }}>Recognition Intelligence</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 11, color: "#9ca3af" }}>Workhuman Capstone · Spring 2026</span>
          {messages.length > 0 && activeTab === "chat" && (
            <button onClick={handleClear} style={{
              fontSize: 11, padding: "4px 10px", borderRadius: 6,
              background: "transparent", border: "0.5px solid rgba(0,0,0,0.1)",
              color: "#9ca3af", cursor: "pointer", fontFamily: "inherit",
            }}>Clear</button>
          )}
        </div>
      </div>

      {/* Body */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* Icon rail */}
        <div style={{
          width: 52, flexShrink: 0, borderRight: "0.5px solid rgba(0,0,0,0.08)",
          background: "white", display: "flex", flexDirection: "column",
          alignItems: "center", padding: "14px 0", gap: 4,
        }}>
          <IconButton active={activeTab === "chat"} onClick={() => setActiveTab("chat")} title="Chat">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
            </svg>
          </IconButton>
          <IconButton active={activeTab === "overview"} onClick={() => setActiveTab("overview")} title="Visual Analytics">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
              <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
            </svg>
          </IconButton>
          <IconButton active={activeTab === "graph"} onClick={() => setActiveTab("graph")} title="Graph">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
              <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
            </svg>
          </IconButton>
        </div>

        {/* ── VISUAL ANALYTICS TAB ── */}
        {activeTab === "overview" && (
          <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <VisualAnalytics stats={stats} />
          </div>
        )}

        {/* ── CHAT TAB ── */}
        {activeTab === "chat" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", background: "#f9fafb", minWidth: 0 }}>
            <div ref={scrollRef} style={{
              flex: 1, overflowY: "auto", padding: "24px 28px",
              display: "flex", flexDirection: "column", gap: 20,
            }}>
              {messages.length === 0 && !loading ? (
                <EmptyState onSend={sendMessage} />
              ) : (
                messages.map((msg, i) => (
                  msg.role === "user" ? (
                    <UserBubble key={i} text={msg.content} />
                  ) : (
                    <PulseCard key={i} msg={msg} onSendFollowUp={sendMessage} />
                  )
                ))
              )}
              {loading && (
                <div style={{ alignSelf: "flex-start", display: "flex", gap: 4, padding: "12px 16px" }}>
                  {[0, 1, 2].map(i => (
                    <div key={i} style={{
                      width: 8, height: 8, borderRadius: "50%", background: "#6366f1",
                      opacity: 0.4, animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`,
                    }} />
                  ))}
                  <style>{`@keyframes pulse { 0%,80%,100% { opacity:0.2; transform:scale(0.8); } 40% { opacity:1; transform:scale(1); } }`}</style>
                </div>
              )}
            </div>

            {messages.length > 0 && !loading && (
              <div style={{ display: "flex", gap: 8, padding: "0 28px 10px", flexWrap: "wrap" }}>
                {SUGGESTIONS.slice(0, 4).map((s, i) => (
                  <button key={i} onClick={() => sendMessage(s.q)} style={{
                    fontSize: 12, padding: "6px 14px", borderRadius: 16,
                    background: "white", border: "0.5px solid rgba(0,0,0,0.08)",
                    color: "#6b7280", cursor: "pointer", fontFamily: "inherit",
                  }}>{s.q}</button>
                ))}
              </div>
            )}

            <div style={{
              padding: "10px 16px", borderTop: "0.5px solid rgba(0,0,0,0.08)",
              background: "white", display: "flex", alignItems: "center", gap: 8,
            }}>
              <input
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about recognition patterns, team dynamics..."
                disabled={loading}
                style={{
                  flex: 1, padding: "10px 16px", borderRadius: 20,
                  background: "#f3f4f6", border: "0.5px solid rgba(0,0,0,0.06)",
                  fontSize: 13, color: "#111827", fontFamily: "inherit",
                  outline: "none",
                }}
              />
              <button
                onClick={() => sendMessage(input)}
                disabled={loading || !input.trim()}
                style={{
                  width: 34, height: 34, borderRadius: "50%",
                  background: input.trim() ? "#6366f1" : "#d1d5db",
                  border: "none", display: "flex", alignItems: "center",
                  justifyContent: "center", cursor: input.trim() ? "pointer" : "default",
                  transition: "background 0.15s",
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                  <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* ── GRAPH TAB (placeholder) ── */}
        {activeTab === "graph" && (
          <div style={{
            flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
            background: "#f9fafb", flexDirection: "column", gap: 8,
          }}>
            <div style={{ fontSize: 14, fontWeight: 500, color: "#6b7280" }}>Graph view</div>
            <div style={{ fontSize: 12, color: "#9ca3af" }}>Neo4j network visualization coming soon</div>
          </div>
        )}

        {/* Context panel — only on chat tab */}
        {activeTab === "chat" && <ContextPanel stats={stats} />}

      </div>
    </div>
  );
}
