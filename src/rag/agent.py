"""
src/rag/agent.py — Agentic RAG with Groq API
================================================
Uses Groq API with llama-3.3-70b-versatile (free tier, 14400 req/day).
Router, tools, CRAG, session memory, trace all unchanged.

Requires:
    pip install groq
    .env: GROQ_API_KEY=gsk_...
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")

from groq import Groq

from src.rag.tools import vector_search, pandas_analytics, substitution
from src.rag.router import QueryRouter, RouteType, CRAGChecker

# Try Neo4j first, fall back to NetworkX
NEO4J_AVAILABLE = False
try:
    from src.rag.neo4j_tool import neo4j_query
    neo4j_query("test")
    NEO4J_AVAILABLE = True
except Exception:
    pass

# Always import graph_tool as fallback
try:
    from src.rag.graph_tool import graph_query
except ImportError:
    def graph_query(q):
        return "Graph tool not available."
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_MODEL  = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3
MAX_TOKENS  = 2000
MAX_HISTORY = 10
RETRY_SLEEP = 3

SYSTEM_PROMPT = """You are an expert HR analytics assistant for a recognition intelligence platform.

You have access to 999 recognition records classified into:
- 4 main categories: Organizational & Team Enablement, Operational Excellence & Delivery,
  Strategic Business Outcomes, Creative & Brand Impact
- 26 subcategories
- A Neo4j graph database with 372 role nodes, 11 role group nodes, and 819 relationships

Your job is to answer questions clearly and insightfully using the context provided.

Guidelines:
- Be concise but thorough
- Use specific numbers and percentages when available
- Highlight patterns and insights, not just raw data
- For graph/network questions, explain what the relationships mean organizationally
- If context is insufficient, say so honestly
- Format responses clearly with headers when answering complex questions
"""

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No GROQ_API_KEY found in .env")
        _client = Groq(api_key=api_key)
        print(f"[Groq] Client initialized — model: {GROQ_MODEL}")
    return _client


def _call_groq(messages: list, trace: list, max_retries: int = 3) -> str:
    client = _get_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            trace.append(f"[GROQ] Sending request (attempt {attempt + 1})...")
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            trace.append("[GROQ] Response received.")
            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            err_str = str(e)

            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = RETRY_SLEEP * (attempt + 1)
                trace.append(f"[GROQ] Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            elif "503" in err_str or "unavailable" in err_str.lower():
                trace.append(f"[GROQ] Service unavailable — waiting {RETRY_SLEEP}s...")
                time.sleep(RETRY_SLEEP)
            else:
                raise

    raise RuntimeError(f"Groq request failed after {max_retries} attempts. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------

_sessions: dict[str, list] = {}


def _get_history(session_id: str) -> list:
    return _sessions.get(session_id, [])


def _update_history(session_id: str, role: str, content: str):
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({"role": role, "content": content})
    if len(_sessions[session_id]) > MAX_HISTORY:
        _sessions[session_id] = _sessions[session_id][-MAX_HISTORY:]


def clear_session(session_id: str = "default"):
    _sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Tool runner
# ---------------------------------------------------------------------------

def _run_tools(tool_names: list, user_input: str) -> dict:
    outputs = {}
    for name in tool_names:
        try:
            if name == "vector_search":
                outputs[name] = vector_search(user_input)
            elif name == "pandas_analytics":
                outputs[name] = pandas_analytics(user_input)
            elif name == "substitution":
                outputs[name] = substitution(user_input)
            elif name == "neo4j_query":
                if NEO4J_AVAILABLE:
                    outputs[name] = neo4j_query(user_input)
                else:
                    outputs["graph_query"] = graph_query(user_input)
            elif name == "graph_query":
                outputs[name] = graph_query(user_input)
            else:
                outputs[name] = f"Unknown tool: {name}"
        except Exception as e:
            outputs[name] = f"Tool error: {str(e)}"
    return outputs


# ---------------------------------------------------------------------------
# Message builder (OpenAI-compatible format for Groq)
# ---------------------------------------------------------------------------

def _build_messages(history: list, user_input: str, context: str | None) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history:
        role = msg["role"]
        # Groq uses "assistant" not "model"
        if role == "model":
            role = "assistant"
        messages.append({"role": role, "content": msg["content"]})

    if context:
        current_text = (
            f"Context retrieved from the HR recognition database:\n\n"
            f"{context}\n\n"
            f"---\n"
            f"User question: {user_input}\n\n"
            f"Please answer the question using the context above."
        )
    else:
        current_text = user_input

    messages.append({"role": "user", "content": current_text})
    return messages


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------

def chat(user_input: str, session_id: str = "default") -> dict:
    """
    Main entry point. Routes query, retrieves context, returns answer.
    """
    router = QueryRouter()
    crag   = CRAGChecker()
    trace  = []

    # --- Step 1: Route ---
    decision = router.route(user_input)
    trace.append(f"[ROUTER] Route: {decision.route.value.upper()}")
    trace.append(f"[ROUTER] Tools: {decision.tools or ['none']}")
    trace.append(f"[ROUTER] Confidence: {decision.confidence.value}")
    trace.append(f"[ROUTER] Reasoning: {decision.reasoning}")
    trace.append(f"[LLM] Using Groq — {GROQ_MODEL}")
    if NEO4J_AVAILABLE:
        trace.append("[GRAPH] Neo4j AuraDB available")
    else:
        trace.append("[GRAPH] Neo4j unavailable — using NetworkX fallback")

    # --- Step 2: Direct (no tool) ---
    if decision.route == RouteType.DIRECT:
        history  = _get_history(session_id)
        messages = _build_messages(history, user_input, context=None)
        answer   = _call_groq(messages, trace)
        _update_history(session_id, "user", user_input)
        _update_history(session_id, "assistant", answer)
        trace.append("[DIRECT] No tool called — answered directly.")

        return {
            "answer":     answer,
            "route":      decision.route.value,
            "tools_used": [],
            "confidence": decision.confidence.value,
            "trace":      trace,
            "crag":       None,
        }

    # --- Step 3: Run tools ---
    trace.append(f"[TOOLS] Running: {decision.tools}")
    tool_outputs = _run_tools(decision.tools, user_input)

    for tool_name, output in tool_outputs.items():
        preview = output[:150].replace("\n", " ")
        trace.append(f"[{tool_name.upper()}] Retrieved: {preview}...")

    # --- Step 4: CRAG check (if enabled) ---
    crag_result = None
    if decision.use_crag and tool_outputs:
        combined_output = " ".join(tool_outputs.values())
        crag_result = crag.check(user_input, combined_output)
        trace.append(f"[CRAG] Score: {crag_result['score']:.2f} | Action: {crag_result['action']}")

        if crag_result["action"] == "fallback" and "vector_search" not in tool_outputs:
            trace.append("[CRAG] Fallback triggered — running vector_search as backup")
            tool_outputs["vector_search_fallback"] = vector_search(user_input)

    # --- Step 5: Build context ---
    context_parts = []
    for tool_name, output in tool_outputs.items():
        context_parts.append(f"--- {tool_name.upper()} RESULTS ---\n{output}")
    context = "\n\n".join(context_parts)

    # --- Step 6: Generate answer with Groq ---
    history  = _get_history(session_id)
    messages = _build_messages(history, user_input, context=context)
    answer   = _call_groq(messages, trace)

    # --- Step 7: Update memory ---
    _update_history(session_id, "user", user_input)
    _update_history(session_id, "assistant", answer)

    trace.append("[DONE] Answer generated.")

    return {
        "answer":     answer,
        "route":      decision.route.value,
        "tools_used": decision.tools,
        "confidence": decision.confidence.value,
        "trace":      trace,
        "crag":       crag_result,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic RAG — Groq API Test")
    print(f"  Model: {GROQ_MODEL}")
    print(f"  Neo4j: {'available' if NEO4J_AVAILABLE else 'unavailable (NetworkX fallback)'}")
    print("=" * 60)

    test_questions = [
        "Hello!",
        "How many awards are in each category?",
        "What kind of messages do engineering teams write?",
        "Who could replace a Senior Director?",
        "Which departments recognize each other most?",
        "Who are the bridge roles that span multiple departments?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("-" * 60)
        result = chat(q, session_id="test")
        print(f"Route:      {result['route'].upper()}")
        print(f"Tools:      {result['tools_used']}")
        print(f"Confidence: {result['confidence']}")
        if result["crag"]:
            print(f"CRAG:       score={result['crag']['score']:.2f} action={result['crag']['action']}")
        print(f"\nAnswer:\n{result['answer']}")
        print("\nTrace:")
        for step in result["trace"]:
            print(f"  {step}")
