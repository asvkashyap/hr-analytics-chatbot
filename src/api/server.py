"""
Recognition Analytics API
==========================
FastAPI backend that serves:
  GET  /api/stats      — aggregated stats computed from annotated_results.csv
  POST /api/chat       — routes to the LangGraph agent (src/rag/agent_graph.py)
  POST /api/clear      — clears agent session memory
  POST /api/visualize  — Groq-powered dynamic chart generator

Run:
    cd "C:\\Users\\kashy\\OneDrive\\Desktop\\Masters Project"
    conda activate masters-project
    uvicorn src.api.server:app --reload --port 8000
"""

import sys
import warnings
import os
import json as _json
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Import the agent ────────────────────────────────────────────────────────────────
# LangGraph agent with guardrails (the rule-based router agent was retired)
AGENT_BACKEND = "graph"
try:
    from src.rag.agent_graph import chat_agent as agent_chat, clear_session
    AGENT_AVAILABLE = True
except Exception as e:
    AGENT_AVAILABLE = False
    AGENT_ERROR = str(e)

# The frontend shows a "route" badge. The LangGraph agent picks tools instead
# of routes, so map its tool choices onto the same route names.
_TOOL_ROUTES = {
    "recognition_statistics":      "analytical",
    "search_recognition_messages": "semantic",
    "find_similar_roles":          "substitution",
    "recognition_network":         "graph",
}


def _route_for(result: dict) -> str:
    if result.get("guard") == "refused":     # blocked by the input guard
        return "guarded"
    tools = result.get("tools_used", [])
    if not tools:
        return "direct"
    if len(tools) > 1:
        return "hybrid"
    return _TOOL_ROUTES.get(tools[0], "direct")

app = FastAPI(title="Recognition Analytics API", version="1.0.0")

# Allow the React dev server (port 5173) and any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data loading ──────────────────────────────────────────────────────────────

DATA_PATHS = [
    Path("output/reports/annotated_results.csv"),
    Path("data/annotated_results.csv"),
    Path("output/annotated_results.csv"),
    Path("annotated_results.csv"),
]

CATEGORY_SHORT = {
    "Organizational & Team Enablement": "Team Enablement",
    "Operational Excellence & Delivery": "Ops Excellence",
    "Strategic Business Outcomes":       "Strategic Outcomes",
    "Creative & Brand Impact":           "Creative & Brand",
}

CATEGORY_COLORS = {
    "Organizational & Team Enablement": "#6C63FF",
    "Operational Excellence & Delivery": "#3ECFCF",
    "Strategic Business Outcomes":       "#F97316",
    "Creative & Brand Impact":           "#EC4899",
}

def classify_role(title: str) -> str:
    """Department for a job title. Uses the SAME rules as the agent
    (config/schema.yaml via src/rag/schema.py), so the dashboard and the
    agent always agree. Previously this file kept its own outdated copy."""
    from src.rag.schema import classify_title
    if pd.isna(title):
        return classify_title("")
    return classify_title(str(title))


def load_df() -> pd.DataFrame:
    for p in DATA_PATHS:
        if p.exists():
            df = pd.read_csv(p)
            df["msg_len"]        = df["message"].str.len()
            df["category_short"] = df["category_name"].map(CATEGORY_SHORT)
            df["role_group"]     = df["recipient_title"].apply(classify_role)
            return df
    raise FileNotFoundError("annotated_results.csv not found in any expected location.")


# ── /api/stats ────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats():
    """
    Returns all aggregated stats the frontend needs to render its charts.
    Called once on load — no repeated hits.
    """
    try:
        df = load_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    total = len(df)

    # ── Category breakdown ────────────────────────────────────────────────────
    cat_summary = (
        df.groupby("category_name")
        .agg(count=("message", "size"), avg_len=("msg_len", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    cat_summary["pct"]   = (cat_summary["count"] / total * 100).round(1)
    cat_summary["short"] = cat_summary["category_name"].map(CATEGORY_SHORT)
    cat_summary["color"] = cat_summary["category_name"].map(CATEGORY_COLORS)

    categories = cat_summary.to_dict(orient="records")

    # ── Subcategory breakdown ─────────────────────────────────────────────────
    sub_summary = (
        df.groupby(["category_name", "subcategory_name", "subcategory_id"])
        .agg(count=("message", "size"), avg_len=("msg_len", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    sub_summary["pct"]   = (sub_summary["count"] / total * 100).round(1)
    sub_summary["color"] = sub_summary["category_name"].map(CATEGORY_COLORS)

    subcategories = sub_summary.to_dict(orient="records")

    # ── Role × category cross-tab ─────────────────────────────────────────────
    role_cat = pd.crosstab(df["role_group"], df["category_name"])
    role_cat = role_cat[role_cat.sum(axis=1) >= 15]
    role_cat_pct = (role_cat.div(role_cat.sum(axis=1), axis=0) * 100).round(1)

    role_theme = []
    for role in role_cat_pct.index:
        row_data = {"role": role}
        for cat in role_cat_pct.columns:
            row_data[CATEGORY_SHORT.get(cat, cat)] = float(role_cat_pct.loc[role, cat])
        row_data["total"] = int(role_cat.loc[role].sum())
        role_theme.append(row_data)

    role_theme.sort(key=lambda x: x.get("Team Enablement", 0), reverse=True)

    # ── Give vs receive ───────────────────────────────────────────────────────
    df["nom_group"] = df["nominator_title"].apply(classify_role)

    gives    = df.groupby("nom_group").size().rename("gives")
    receives = df.groupby("role_group").size().rename("receives")
    gr = pd.DataFrame({"gives": gives, "receives": receives}).fillna(0)
    gr_total = gr["gives"] + gr["receives"]
    gr = gr[gr_total >= 10]
    gr["gives_pct"]    = (gr["gives"]    / gr_total * 100).round(0).astype(int)
    gr["receives_pct"] = (gr["receives"] / gr_total * 100).round(0).astype(int)
    gr = gr.sort_values("gives_pct", ascending=False).reset_index()
    gr = gr.rename(columns={"index": "role"})

    give_receive = gr[["role", "gives_pct", "receives_pct"]].to_dict(orient="records")

    # ── Top recognition flows ─────────────────────────────────────────────────
    flow_df = (
        df.groupby(["nom_group", "role_group"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(8)
    )
    flow_df["pct"] = (flow_df["count"] / total * 100).round(1)

    flows = []
    for _, row in flow_df.iterrows():
        reverse_exists = any(
            (f["from_role"] == row["role_group"] and f["to_role"] == row["nom_group"])
            for f in flows
        )
        flows.append({
            "from_role": row["nom_group"],
            "to_role":   row["role_group"],
            "count":     int(row["count"]),
            "pct":       float(row["pct"]),
            "mutual":    reverse_exists,
        })

    # ── Message length by category ────────────────────────────────────────────
    msg_len_by_cat = []
    for cat in df["category_name"].unique():
        lens = df[df["category_name"] == cat]["msg_len"].tolist()
        msg_len_by_cat.append({
            "category":  cat,
            "short":     CATEGORY_SHORT.get(cat, cat),
            "color":     CATEGORY_COLORS.get(cat, "#6C63FF"),
            "median":    int(np.median(lens)),
            "q1":        int(np.percentile(lens, 25)),
            "q3":        int(np.percentile(lens, 75)),
            "min":       int(np.min(lens)),
            "max":       int(np.min([np.max(lens), int(np.percentile(lens, 95))])),
        })

    # ── KPIs ──────────────────────────────────────────────────────────────────
    top_cat          = cat_summary.iloc[0]
    bottom_cat       = cat_summary.iloc[-1]
    top_sub          = sub_summary.iloc[0]
    top2_pct         = round(float(cat_summary.iloc[0]["pct"]) + float(cat_summary.iloc[1]["pct"]), 1)
    ratio_top_bottom = round(float(top_cat["count"]) / float(bottom_cat["count"]), 1)

    # ── Pulse panel: computed by the same graph code the agent uses ──────────
    pulse = {"least_recognized": {
        "name": str(bottom_cat["category_name"]), "pct": float(bottom_cat["pct"]),
    }}
    try:
        from src.rag.graph_tool import central_departments, top_relationships, _dept_graph
        hub = central_departments(1)
        link = top_relationships(1, cross_department_only=True)
        pulse["most_central"] = hub[0][0] if hub else None
        pulse["strongest_link"] = (
            {"from": link[0]["from"], "to": link[0]["to"], "count": int(link[0]["count"])}
            if link else None
        )
        pulse["departments"] = _dept_graph().number_of_nodes()
    except Exception as e:
        pulse["error"] = str(e)

    # ── Sample messages for explorer ──────────────────────────────────────────
    sample_cols = ["message", "award_title", "recipient_title", "nominator_title",
                   "category_name", "category_short", "subcategory_name",
                   "subcategory_id", "msg_len"]
    available_cols = [c for c in sample_cols if c in df.columns]
    messages = df[available_cols].fillna("").to_dict(orient="records")

    return {
        "total":           total,
        "n_categories":    int(df["category_name"].nunique()),
        "n_subcategories": int(df["subcategory_name"].nunique()),
        "median_msg_len":  int(df["msg_len"].median()),
        "top_category":    {
            "name":  str(top_cat["category_name"]),
            "short": str(top_cat["short"]),
            "pct":   float(top_cat["pct"]),
            "count": int(top_cat["count"]),
        },
        "bottom_category": {
            "name":  str(bottom_cat["category_name"]),
            "short": str(bottom_cat["short"]),
            "pct":   float(bottom_cat["pct"]),
            "count": int(bottom_cat["count"]),
        },
        "top_subcategory": {
            "name":  str(top_sub["subcategory_name"]),
            "pct":   float(top_sub["pct"]),
            "count": int(top_sub["count"]),
        },
        "top2_pct":              top2_pct,
        "ratio_top_bottom":      ratio_top_bottom,
        "categories":            categories,
        "subcategories":         subcategories,
        "role_theme":            role_theme,
        "give_receive":          give_receive,
        "flows":                 flows,
        "msg_len_by_cat":        msg_len_by_cat,
        "messages":              messages,
        "pulse":                 pulse,
        "agent_available":       AGENT_AVAILABLE,
    }


# ── Groq caller (used by /api/visualize) ──────────────────────────────────────

_viz_limiter = None   # created on first use, for the configured model

def _call_groq(prompt: str) -> str:
    """
    Calls Groq with the model from .env (GROQ_MODEL), the same model as the
    agent. llama-3.3-70b-versatile was retired by Groq on 2026-08-16.
    Returns raw text response. Raises HTTPException on failure.
    """
    import time
    from groq import Groq
    from src.utils.llm_usage import RateLimiter, record_usage, extract_usage, estimate_tokens

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="No GROQ_API_KEY found in .env"
        )

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    global _viz_limiter
    if _viz_limiter is None:
        _viz_limiter = RateLimiter("groq", model)

    try:
        client = Groq(api_key=api_key)
        # gpt-oss reasons before answering, and reasoning uses output tokens,
        # so the limit is higher than the 1,500 the old model needed
        _viz_limiter.wait(estimate_tokens(prompt, expected_output=2000))
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000,
        )
        in_tok, out_tok = extract_usage(response)
        _viz_limiter.correct(in_tok + out_tok)
        record_usage("visualize", model, in_tok, out_tok, time.perf_counter() - start)
        return response.choices[0].message.content
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "rate_limit" in err_str.lower():
            raise HTTPException(status_code=429, detail=f"Groq rate limit hit: {err_str[:300]}")
        raise HTTPException(status_code=502, detail=f"Groq API error: {err_str[:300]}")


# ── /api/visualize ────────────────────────────────────────────────────────────

class VisualizeRequest(BaseModel):
    query: str


@app.post("/api/visualize")
def visualize_endpoint(req: VisualizeRequest):
    """
    1. Reads annotated_results.csv and computes a rich data summary
    2. Sends the query + data summary to Groq (llama-3.3-70b-versatile)
    3. Groq picks the best chart type and returns a Chart.js v4 config
    4. Returns it to the frontend to render — no hardcoded charts
    """

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        df = load_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    total = len(df)

    # ── Build data summary ─────────────────────────────────────────────────
    cat_counts = df["category_name"].value_counts().to_dict()
    sub_counts = df["subcategory_name"].value_counts().head(20).to_dict()

    # Give vs receive
    df["nom_group"] = df["nominator_title"].apply(classify_role)
    given    = df["nom_group"].value_counts().to_dict()
    received = df["role_group"].value_counts().to_dict()
    all_groups = sorted(set(list(given.keys()) + list(received.keys())))
    give_receive = {
        g: {"given": int(given.get(g, 0)), "received": int(received.get(g, 0))}
        for g in all_groups
    }

    # Monthly trend
    monthly = {}
    if "award_date" in df.columns:
        df["award_date"] = pd.to_datetime(df["award_date"], errors="coerce")
        monthly_series = df.dropna(subset=["award_date"]).groupby(
            df["award_date"].dt.to_period("M")
        ).size()
        monthly = {str(k): int(v) for k, v in monthly_series.items()}

    data_summary = {
        "total_records": total,
        "category_distribution": cat_counts,
        "subcategory_distribution_top10": dict(list(sub_counts.items())[:10]),
        "give_receive_by_role_group": give_receive,
        "monthly_trend": monthly,
    }

    # ── Build prompt ───────────────────────────────────────────────────────
    prompt = f"""You are a data visualization expert. Given HR recognition data and a user question,
return a Chart.js v4 configuration object as valid JSON.

USER QUESTION: {req.query}

DATA SUMMARY:
{_json.dumps(data_summary, indent=2)}

INSTRUCTIONS:
1. Choose the best chart type for the question (bar, line, doughnut, pie, radar, scatter, bubble, or bar with indexAxis:y for horizontal).
2. Use the actual numbers from the data summary above — do not invent data.
3. Use these hex colors: ["#7F77DD","#1D9E75","#BA7517","#D85A30","#378ADD","#E24B4A","#888780","#EF9F27","#5DCAA5"].
4. Truncate long labels to 30 characters.
5. For horizontal bar charts set options.indexAxis to "y".
6. Always set options.responsive = true and options.maintainAspectRatio = false.
7. Disable Chart.js default legend: plugins.legend.display = false.
8. Add subtle grid lines: color "rgba(128,128,128,0.08)".
9. Set tick colors to "#9ca3af".

ALSO return these fields alongside chart_config:
- title: short chart title (max 8 words)
- chart_type: human-readable type string e.g. "horizontal bar", "doughnut", "line"
- insight: 1-2 sentence plain-English insight about what the chart shows
- metrics: array of up to 4 objects with "label" (string) and "value" (string) for summary cards
- record_count: {total}

Respond ONLY with a single valid JSON object — no markdown, no backticks, no explanation outside the JSON:
{{
  "title": "...",
  "chart_type": "...",
  "record_count": {total},
  "insight": "...",
  "metrics": [{{"label": "...", "value": "..."}}],
  "chart_config": {{
    "type": "...",
    "data": {{}},
    "options": {{}}
  }}
}}"""

    # ── Call Groq ──────────────────────────────────────────────────────────
    raw_text = _call_groq(prompt)

    # ── Parse response ─────────────────────────────────────────────────────
    raw_text = raw_text.strip()

    # Strip markdown fences if model wraps in ```json ... ```
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:])
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
    raw_text = raw_text.strip()

    # Find outermost JSON object
    start = raw_text.find("{")
    end   = raw_text.rfind("}") + 1
    if start == -1 or end == 0:
        raise HTTPException(
            status_code=502,
            detail=f"No JSON found in Groq response: {raw_text[:300]}"
        )

    try:
        result = _json.loads(raw_text[start:end])
    except _json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"JSON parse error: {str(e)} — raw snippet: {raw_text[start:start+300]}"
        )

    if "chart_config" not in result:
        raise HTTPException(
            status_code=502,
            detail="Groq response missing 'chart_config' field"
        )

    return result


# ── Chart type inference ──────────────────────────────────────────────────────

ROLE_KEYWORDS = [
    "engineering", "leadership", "customer service", "marketing",
    "people/hr", "people", "hr", "finance", "data/analytics",
    "data", "analytics", "design", "creative", "product", "sales",
]

def infer_chart_type(route: str, tools_used: list, query: str) -> Optional[str]:
    lq = query.lower()

    if route == "direct" or route == "guarded":
        return None
    if route == "graph":
        return "flows"
    if route == "substitution":
        for r in ROLE_KEYWORDS:
            if r in lq:
                return f"role_theme:{r}"
        return "role_theme:engineering"
    if route == "semantic":
        for r in ROLE_KEYWORDS:
            if r in lq:
                return f"role_theme:{r}"
        return "themes"
    if route in ("analytical", "hybrid"):
        if any(k in lq for k in ["give", "receiv", "imbalance", "gap", "underrecogniz",
                                   "invisible", "missing", "least", "dark spot"]):
            return "give_receive"
        if any(k in lq for k in ["connect", "flow", "between", "each other",
                                   "cross", "direction", "top-down", "lateral"]):
            return "flows"
        for r in ROLE_KEYWORDS:
            if r in lq:
                return f"role_theme:{r}"
        return "themes"

    return "themes"


# ── /api/chat ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str
    session_id: str

class ClearRequest(BaseModel):
    session_id: str

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {AGENT_ERROR}")

    try:
        result     = agent_chat(user_input=req.message, session_id=req.session_id)
        route      = _route_for(result)
        tools_used = result.get("tools_used", [])
        chart_type = infer_chart_type(route, tools_used, req.message)

        return {
            "answer":       result.get("answer", ""),
            "route":        route,
            "tools_used":   tools_used,
            "confidence":   result.get("confidence", ""),
            "trace":        result.get("trace", []),
            "crag":         result.get("crag", None),
            "chart_type":   chart_type,
            "backend":      AGENT_BACKEND,
            "guard":        result.get("guard"),
            "input_label":  result.get("input_label"),
            "cost_usd":     result.get("cost_usd"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
def clear_endpoint(req: ClearRequest):
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent unavailable")
    try:
        clear_session(req.session_id)
        return {"cleared": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok", "agent": AGENT_AVAILABLE, "backend": AGENT_BACKEND}
