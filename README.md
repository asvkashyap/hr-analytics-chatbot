# HR Recognition Analytics — Agentic RAG Pipeline

> Northeastern University × Workhuman Capstone — Spring 2026

End-to-end NLP pipeline that classifies 1,000 employee recognition messages into a hierarchical topic taxonomy using LLM-driven grounded theory, then builds an agentic RAG chatbot on top for natural language HR analytics.

---

## What it does
## Demo
[![Watch the demo](https://img.shields.io/badge/Loom-Watch_Demo-625DF5?style=for-the-badge&logo=loom&logoColor=white)](https://www.loom.com/share/[YOUR_VIDEO_ID](https://www.loom.com/share/04c68b3e16d9421f86810523502f52b3))

**Stage 1 — Taxonomy creation:** Processes raw recognition messages in batches of 20 through a Gemini LLM. Uses grounded theory — no predefined labels. Taxonomy compresses every 5 batches to remove redundancy. Output: 4 categories, 26 subcategories with hierarchical IDs (A, A1, A2...).

**Stage 2 — Annotation:** Classifies all 999 records against the taxonomy. Batch size 20 with retry logic. Failed records saved separately and retried. Output: `annotated_results.csv`.

**Stage 3 — Agentic RAG:** Chatbot answers natural language HR analytics questions using 5 specialized tools, a dynamic 6-route query router, CRAG self-correction, Neo4j graph database, and session memory.

**Stage 4 — Visual analytics:** Natural language → LLM picks chart type → builds Chart.js config from real CSV data → renders in the browser. No hardcoded charts.

---

## Architecture

```
Raw CSV (1000 messages)
       │
       ▼
┌─────────────────────────────────────┐
│         PIPELINE                    │
│  Dedup → Taxonomy Creation          │
│  (Gemini, batch=20, compress/5)     │
│         ↓                           │
│  Annotation (999 records)           │
│  annotated_results.csv              │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
  ChromaDB          Neo4j AuraDB
  (384-dim           (413 nodes
  embeddings)        906 relationships)
       │                │
       └───────┬────────┘
               ▼
┌─────────────────────────────────────┐
│         QUERY ROUTER                │
│  DIRECT → SEMANTIC → ANALYTICAL     │
│  HYBRID → GRAPH → SUBSTITUTION      │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│         5 AGENT TOOLS               │
│  vector_search    pandas_analytics  │
│  substitution     neo4j_query       │
│  graph_query (NetworkX fallback)    │
└──────────────┬──────────────────────┘
               ▼
         CRAG checker
    (keyword overlap scoring)
               ▼
        Groq LLM answer
               ▼
┌─────────────────────────────────────┐
│         SERVING LAYER               │
│  FastAPI backend  (/api/chat,       │
│  /api/stats, /api/visualize)        │
│  React + Chart.js frontend          │
└─────────────────────────────────────┘
```

---

## Taxonomy

Built from data — not predefined:

```
A. Strategic Business Outcomes
   A1. Business Growth & Client Success
   A2. Customer Experience & Engagement
   A3. Strategic Direction & Innovation

B. Operational Excellence & Delivery
   B1. Project & Process Management
   B2. Technical & System Operations
   B3–B9. (7 more subcategories)

C. Creative & Brand Impact
   C1. Content Strategy & Creation
   C2. Visual Design & Branding
   C3–C5. (3 more subcategories)

D. Organizational & Team Enablement
   D1. Team Collaboration & Integration
   D2. Learning & Professional Development
   D3–D9. (7 more subcategories)
```

**Distribution across 999 records:**

| Category | Records | % |
|---|---|---|
| Organizational & Team Enablement | 442 | 44.2% |
| Operational Excellence & Delivery | 340 | 34.0% |
| Strategic Business Outcomes | 164 | 16.4% |
| Creative & Brand Impact | 53 | 5.3% |

---

## RAG System

### Query Router — 6 routes

| Route | Trigger | Tools |
|---|---|---|
| `DIRECT` | Greetings, meta questions | None |
| `SEMANTIC` | Themes, examples, qualitative | `vector_search` |
| `ANALYTICAL` | Counts, rankings, distributions | `pandas_analytics` |
| `HYBRID` | Role + qualitative combined | `pandas_analytics` + `vector_search` |
| `GRAPH` | Team relationships, network | `neo4j_query` / `graph_query` |
| `SUBSTITUTION` | "Who could replace X" | `substitution` |

### CRAG Self-Correction

Keyword overlap scoring between query and retrieved context:
- Score ≥ 0.25 → proceed
- Score ≥ 0.125 → rerank
- Score < 0.125 → fallback (adds `vector_search` as backup)

### Neo4j Graph

- 413 nodes: 4 categories, 26 subcategories, 11 role groups, 372 individual roles
- 906 relationships: `RECOGNIZED` (with count, top_category) + `BELONGS_TO`
- Most central: Engineering (4,655 connections)
- Strongest cross-team: Product/Program → Engineering (21)
- NetworkX in-memory graph as fallback if Neo4j is unavailable

### RAGAS Evaluation

| Metric | Score |
|---|---|
| Answer Relevancy | **0.776** |
| Faithfulness | Partial (rate limited) |
| Context Precision | Partial (rate limited) |

---

## Visual Analytics

`/api/visualize` endpoint:
1. Loads `annotated_results.csv`
2. Builds data summary (category counts, subcategory top 20, give/receive by role group, monthly trend)
3. Sends to LLM with Chart.js prompt — returns pure JSON config
4. Frontend renders any chart type: bar, horizontal bar, doughnut, line, radar, scatter

No hardcoded charts. Everything generated from real data at query time.

---

## Project Structure

```
├── src/
│   ├── pipeline/
│   │   ├── create_taxonomy.py     # Stage 1: build taxonomy
│   │   ├── annotate.py            # Stage 2: classify records
│   │   └── retry_failed.py        # Retry failed batches
│   ├── rag/
│   │   ├── ingest.py              # ChromaDB ingestion
│   │   ├── tools.py               # 5 agent tools
│   │   ├── router.py              # Dynamic router + CRAG
│   │   ├── agent.py               # Main agent pipeline
│   │   ├── graph_tool.py          # NetworkX in-memory graph
│   │   ├── neo4j_loader.py        # Load graph into Neo4j
│   │   ├── neo4j_tool.py          # 7 Cypher query types
│   │   └── ragas_eval.py          # RAGAS evaluation
│   ├── api/
│   │   └── server.py              # FastAPI: /chat /stats /visualize
│   └── visualization/
│       ├── dashboard.jsx          # React chat UI
│       └── VisualAnalytics.jsx    # Chart generation UI
├── data/raw/
│   └── mockup_awards.csv          # 1000 recognition records
├── output/reports/
│   ├── annotated_results.csv      # 999 annotated records
│   └── compressed_taxonomy_with_ids.json
├── config/
│   └── config.yaml                # All settings
└── requirements.txt
```

---

## Running Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/asvkashyap/hr-analytics-chatbot
cd hr-analytics-chatbot
conda create -n masters-project python=3.11
conda activate masters-project
pip install -r requirements.txt

# 2. Add environment variables
cp .env.example .env
# Fill in GROQ_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# 3. Build the vector store
python -m src.rag.ingest

# 4. Load Neo4j graph (optional)
python -m src.rag.neo4j_loader

# 5. Start FastAPI backend
uvicorn src.api.server:app --reload --port 8000

# 6. Start React frontend (separate terminal)
npm install
npm run dev
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM (agent) | Groq API — Llama 3.3 70B Versatile |
| LLM (taxonomy + viz) | Google Gemini 2.5 Flash |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Vector DB | ChromaDB (local persistent, cosine similarity) |
| Graph DB | Neo4j AuraDB (cloud) + NetworkX (fallback) |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite + Chart.js |
| Evaluation | RAGAS 0.4.3 |
| Environment | Conda, Python 3.11 |

---

## Key Technical Decisions

**Raw urllib for annotation (not SDK)** — more reliable on free-tier APIs with rate limits. SDK abstraction adds failure modes.

**API key rotation** — reads all keys from `config.yaml`, rotates on 429. Handles free-tier limits without manual intervention across 50 batches.

**Grounded theory taxonomy** — labels emerge from the data rather than being predefined. Each batch adds only new categories; compression every 5 batches keeps the schema clean.

**NetworkX fallback for Neo4j** — app works fully even if cloud DB is down. Graph is rebuilt in memory from the CSV on startup.

**CRAG over naive RAG** — retrieval quality is scored before passing context to the LLM. Low-quality retrievals trigger a fallback tool call rather than hallucinating from bad context.

---

*Northeastern University — MS Data Analytics Engineering · Workhuman Capstone · Spring 2026*
