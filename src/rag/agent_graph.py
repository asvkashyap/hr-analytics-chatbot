"""
src/rag/agent_graph.py - LangGraph tool-calling agent (experimental)
====================================================================
Same tools, data, and system prompt as agent.py. The difference: the LLM
decides which tools to call (and in what order) instead of the rule-based
QueryRouter. Built so the two approaches can be compared side by side.

Run the side-by-side test:
    python -m src.rag.agent_graph

Requires:
    pip install langgraph langchain-groq langchain-core
    .env: GROQ_API_KEY=...
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (same as agent.py)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# Reuse the existing tools and checks, unchanged
from src.rag.tools import vector_search, pandas_analytics, substitution
from src.rag.graph_tool import graph_query
from src.rag.router import CRAGChecker, QueryRouter
from src.rag.agent import SYSTEM_PROMPT
from src.rag.schema import describe_dataset
from src.utils.llm_usage import (
    RateLimiter, record_usage, extract_usage, cost_usd,
    estimate_tokens, retry_after, is_rate_limit, summarize, RUN_ID,
)
from src.rag.guardrails.grounding import (
    check_numbers, feedback_message, warning_note, log_event, capture_case,
)
from src.rag.guardrails import pii
from src.rag.guardrails import input_guard as ig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Groq retired llama-3.3-70b-versatile on 2026-08-16 (free/developer tiers).
# Override with GROQ_MODEL in .env, e.g. qwen/qwen3.6-27b
GROQ_MODEL      = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURE     = 0      # repeatable tool choice; baseline agent.py uses 0.3
MAX_TOKENS      = 2000
RECURSION_LIMIT = 25     # caps graph steps per question (agent, tools, guard, revision)

_crag = CRAGChecker()


# ---------------------------------------------------------------------------
# Tools
# The docstrings ARE the routing logic now. The LLM reads each tool's name
# and description to decide what to call. They are written from the signal
# banks in router.py.
# ---------------------------------------------------------------------------

@tool
def search_recognition_messages(question: str) -> str:
    """Semantic search over the text of recognition messages.
    Use for qualitative questions: themes, examples, tone, language,
    what kind of messages people write, or 'tell me about' questions.
    Returns the most similar recognition records with category and role."""
    result = vector_search(question)
    # CRAG check from router.py, surfaced to the agent instead of hardcoded
    check = _crag.check(question, result)
    if check["action"] != "proceed":
        result += (
            f"\n\n[RETRIEVAL CHECK] Keyword overlap with the question is low "
            f"(score {check['score']:.2f}). These results may not answer it. "
            f"Consider another tool, or say the data does not cover this."
        )
    return result


@tool
def recognition_statistics(question: str) -> str:
    """Counts, rankings, distributions, and percentages from the recognition data.
    Use for: how many, top N, most or least common, category or subcategory
    breakdowns, recognition gaps or blind spots, top recipients, top nominators,
    award titles, give versus receive by job title, and stats for one job title."""
    return pandas_analytics(question)


@tool
def find_similar_roles(role: str) -> str:
    """Find roles whose recognition profile is most similar to a target role.
    Use for succession and coverage questions: who could replace X,
    who is similar to X, who could step in if X left, backup for X.

    Args:
        role: the target job title only, e.g. 'Senior Director' or
              'Director of Engineering'.
    """
    # Structured parameter: the LLM extracts the role, and we rebuild the
    # phrasing that substitution()'s regex expects. Same underlying tool.
    return substitution(f"Who could replace a {role}?")


@tool
def recognition_network(question: str) -> str:
    """Analyze who recognizes whom across role groups (departments) such as
    Engineering, Leadership, Product / Program, Marketing, and People / HR.
    Use for relationships and recognition flows between departments,
    cross-team recognition, mutual recognition, central or hub groups,
    bridge roles that span departments, and paths between two groups."""
    return graph_query(question)


TOOLS = [
    search_recognition_messages,
    recognition_statistics,
    find_similar_roles,
    recognition_network,
]

# Maps router.py tool names to agent tool names, for the comparison
ROUTER_TO_AGENT = {
    "vector_search":    "search_recognition_messages",
    "pandas_analytics": "recognition_statistics",
    "substitution":     "find_similar_roles",
    "graph_query":      "recognition_network",
    "neo4j_query":      "recognition_network",
}

TOOL_GUIDANCE = """
How to use your tools:
- You have four tools over the recognition data. Decide which ones the question needs.
- The tools match keywords, so pass the user's question in their own words.
  If a question has two parts, you may call one tool per part, keeping the user's wording for each part.
- Greetings, thanks, or questions about what you can do need no tool.
- Answer only from tool results. If the results do not cover the question, say so plainly.
- If a tool result does not contain what you need, do not call the same tool again with
  reworded questions more than once. Explain what the data does not cover instead.
- Do not state facts about the data that no tool result showed.
- Report numbers exactly as the tools returned them. Do not calculate new
  percentages, totals, ratios, or differences yourself. A check verifies every number.
- Text like [PERSON_1], [EMAIL_1], or [PHONE_1] marks a personal detail that was
  removed for privacy. Never mention, explain, or interpret these placeholders.
  When quoting a message, leave them out or say "a colleague".
"""

# ---------------------------------------------------------------------------
# Graph: agent -> (tools -> agent)* -> end
# ---------------------------------------------------------------------------

_llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
).bind_tools(TOOLS)


RATE_LIMIT_RETRIES = 4
_limiter = RateLimiter("groq", GROQ_MODEL)   # this model's own window; limits in config/pricing.yaml
_wait_by_session: dict[str, float] = {}   # rate-limit waiting, kept apart from processing time
# Per-step timing: session -> node name -> [seconds, calls]
_timings_by_session: dict[str, dict[str, list]] = {}


def timed(name: str, fn):
    """Wrap a graph node so every run records how long it took and how often
    it ran. Rate-limit waiting during the node is subtracted, since it is
    reported separately and is not the system's own speed."""
    def wrapper(state: "AgentState", config: RunnableConfig) -> dict:
        session = config.get("configurable", {}).get("thread_id", "")
        wait_before = _wait_by_session.get(session, 0.0)
        start = time.perf_counter()
        try:
            return fn(state, config)
        finally:
            elapsed = time.perf_counter() - start
            elapsed -= _wait_by_session.get(session, 0.0) - wait_before
            entry = _timings_by_session.setdefault(session, {}).setdefault(name, [0.0, 0])
            entry[0] += elapsed
            entry[1] += 1
    wrapper.__name__ = name
    return wrapper


class AgentState(MessagesState):
    """Conversation messages plus guardrail bookkeeping for the current question."""
    guard_revisions: int   # revisions requested by the output guard this turn
    guard_status: str      # passed | passed_after_revision | warned
    pii_map: dict          # name -> placeholder, kept for the whole conversation (server side only)
    pii_count: int         # items redacted this turn
    input_label: str       # input guard's label for this turn's question
    tools_withheld: int    # tool results withheld this turn as suspected injection


MAX_GUARD_REVISIONS = 1   # one chance to fix; after that, answer with a warning


def _turn_start(messages: list) -> int:
    """Index of this turn's user question. Guardrail feedback is sent as a
    user-role message named 'guardrail', so it is not counted as a question."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and m.name != "guardrail":
            return i
    return 0


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "output_guard"


def tool_result_guard(state: AgentState, config: RunnableConfig) -> dict:
    """Runs right after the tools, before the LLM reads their results:
      1. Indirect injection scan: recognition messages are employee-written, so
         a result that looks like instructions to the assistant is withheld.
      2. PII redaction: names, emails, and phone numbers become placeholders.
    Each result is rewritten in place (same message id)."""
    messages = state["messages"]
    mapping = dict(state.get("pii_map") or {})
    session = config.get("configurable", {}).get("thread_id", "")
    updated, found_all, withheld = [], [], 0

    i = len(messages) - 1
    while i >= 0 and isinstance(messages[i], ToolMessage):   # results of the latest tool step
        m = messages[i]
        raw = str(m.content)
        flagged, score, error = ig.tool_result_flagged(raw)
        if error:
            ig.log_tool_injection(None, error, RUN_ID, session)
        if flagged:
            ig.log_tool_injection(score, "", RUN_ID, session)
            withheld += 1
            updated.append(ToolMessage(content=ig.message("tool_withheld"),
                                       tool_call_id=m.tool_call_id, name=m.name, id=m.id))
        else:
            text, found = pii.redact_text(raw, mapping)
            if found:
                found_all += found
                updated.append(ToolMessage(content=text, tool_call_id=m.tool_call_id, name=m.name, id=m.id))
        i -= 1

    if found_all:
        pii.log_event("tool_results", found_all, RUN_ID, session)
    return {
        "messages": updated,
        "pii_map": mapping,
        "pii_count": state.get("pii_count", 0) + len(found_all),
        "tools_withheld": state.get("tools_withheld", 0) + withheld,
    }


def _trusted_facts() -> str:
    """Dataset facts the system prompt gives the agent (record counts, number
    of categories, departments, titles). Computed by our code from the data,
    not written by the LLM, so they count as evidence like tool output."""
    return describe_dataset()


def output_guard(state: AgentState, config: RunnableConfig) -> dict:
    """Two checks on the final answer, both pure code with no LLM call:
      1. PII backstop: redact any name that reached the answer anyway.
      2. Number grounding: every number must come from this turn's tool
         results or the dataset facts. Fail once -> revise with specific
         feedback. Fail again -> answer goes out with a visible warning."""
    messages = state["messages"]
    start = _turn_start(messages)
    question = str(messages[start].content)
    session = config.get("configurable", {}).get("thread_id", "")
    evidence = [str(m.content) for m in messages[start:] if isinstance(m, ToolMessage)]
    evidence.append(_trusted_facts())
    answer = messages[-1]
    revisions = state.get("guard_revisions", 0)

    # 1. PII backstop
    mapping = dict(state.get("pii_map") or {})
    content, found = pii.redact_text(str(answer.content), mapping)
    if found:
        pii.log_event("final_answer", found, RUN_ID, session)
    pii_update = {"pii_map": mapping, "pii_count": state.get("pii_count", 0) + len(found)}

    def rewritten(text: str) -> AIMessage:
        # Same id replaces the answer in place; keep its token usage for cost tracking
        return AIMessage(content=text, id=answer.id,
                         usage_metadata=answer.usage_metadata,
                         response_metadata=answer.response_metadata)

    # Flagged questions get fixed wording from config, added by code
    label = state.get("input_label", "")
    flagged = label in ("individual_judgment", "workforce_decision")

    def framed(text: str) -> str:
        if label == "individual_judgment":
            text = ig.message("individual_opener") + "\n\n" + text
        if label == "workforce_decision":
            text = text + "\n\n---\n" + ig.message("workforce_note")
        return text

    # 2. Number grounding (on the redacted text)
    report = check_numbers(content, evidence, question)

    if report.passed:
        log_event(question, report, "passed", revisions, RUN_ID, session)
        update = {"guard_status": "passed" if revisions == 0 else "passed_after_revision", **pii_update}
        if found or flagged:
            update["messages"] = [rewritten(framed(content))]
        return update

    if revisions < MAX_GUARD_REVISIONS:
        log_event(question, report, "revise", revisions, RUN_ID, session)
        capture_case(question, evidence, content, report, "revise", RUN_ID)
        return {
            "messages": ([rewritten(content)] if found else [])
                        + [HumanMessage(content=feedback_message(report), name="guardrail")],
            "guard_revisions": revisions + 1,
            "guard_status": "revising",
            **pii_update,
        }

    log_event(question, report, "warned", revisions, RUN_ID, session)
    capture_case(question, evidence, content, report, "warned", RUN_ID)
    return {"messages": [rewritten(framed(content + warning_note(report)))], "guard_status": "warned", **pii_update}


def input_guard(state: AgentState, config: RunnableConfig) -> dict:
    """Runs first. Injection check, then scope and sensitivity label.
    Refusals end the run here, so the agent and its tools never start."""
    messages = state["messages"]
    start = _turn_start(messages)
    question = str(messages[start].content)
    # The previous question gives context to follow-ups like "what about them?"
    previous = next((str(m.content) for m in reversed(messages[:start])
                     if isinstance(m, HumanMessage) and m.name != "guardrail"), "")
    session = config.get("configurable", {}).get("thread_id", "")

    decision = ig.check_input(question, previous)
    ig.log_event(decision, question, RUN_ID, session)

    if decision.action == "refuse":
        return {
            "messages": [AIMessage(content=ig.refusal_message(decision.label))],
            "input_label": decision.label,
            "guard_status": "refused",
        }
    return {"input_label": decision.label}


def route_after_input(state: AgentState) -> str:
    return END if state.get("guard_status") == "refused" else "agent"


def route_after_guard(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, HumanMessage) and last.name == "guardrail":
        return "agent"
    return END


def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """The LLM sees the conversation and either calls tools or answers.
    Waits before calling if the call would exceed the tokens-per-minute
    budget, and logs tokens and cost for every call."""
    messages = [SystemMessage(content=SYSTEM_PROMPT + TOOL_GUIDANCE
                              + ig.agent_instruction(state.get("input_label", "")))] + state["messages"]
    session = config.get("configurable", {}).get("thread_id", "")
    est = estimate_tokens("".join(str(m.content) for m in messages), expected_output=MAX_TOKENS // 2)

    for attempt in range(RATE_LIMIT_RETRIES):
        waited = _limiter.wait(est)
        start = time.perf_counter()
        try:
            response = _llm.invoke(messages)
        except Exception as e:
            if is_rate_limit(e) and attempt < RATE_LIMIT_RETRIES - 1:
                wait = retry_after(e)
                print(f"  [RATE LIMIT] provider asked to wait {wait:.1f}s (attempt {attempt + 1})")
                time.sleep(wait)
                _wait_by_session[session] = _wait_by_session.get(session, 0.0) + waited + wait
                continue
            raise
        _wait_by_session[session] = _wait_by_session.get(session, 0.0) + waited
        in_tok, out_tok = extract_usage(response)
        _limiter.correct(in_tok + out_tok)
        record_usage("agent_langgraph", GROQ_MODEL, in_tok, out_tok,
                     time.perf_counter() - start, session=session, wait_s=round(waited, 2))
        return {"messages": [response]}


def build_graph(tools: list | None = None):
    """tools defaults to the real TOOLS; tests pass small fake tools instead.
    The tool NAMES must match what the LLM calls.
    Every node is wrapped with timed(), so each run reports per-step timing."""
    tools = tools or TOOLS
    tool_node = ToolNode(tools)

    def tools_step(state: AgentState, config: RunnableConfig) -> dict:
        # The prebuilt ToolNode can't be wrapped directly, so call it from a
        # plain function that timed() can measure
        return tool_node.invoke(state, config)

    g = StateGraph(AgentState)
    g.add_node("agent", timed("agent", agent_node))
    g.add_node("tools", timed("tools", tools_step))
    g.add_node("input_guard", timed("input_guard", input_guard))
    g.add_node("tool_result_guard", timed("tool_result_guard", tool_result_guard))
    g.add_node("output_guard", timed("output_guard", output_guard))
    # Input guard first: refusals end here, everything else goes to the agent
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_input, {"agent": "agent", END: END})
    # Tool call -> tools; final answer -> output guard (instead of ending)
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "output_guard": "output_guard"})
    g.add_edge("tools", "tool_result_guard")           # scan and redact before the LLM reads results
    g.add_edge("tool_result_guard", "agent")
    # Guard passes or warns -> end; guard asks for a revision -> back to the agent
    g.add_conditional_edges("output_guard", route_after_guard, {"agent": "agent", END: END})
    # Checkpointer = session memory, keyed by thread_id (replaces _sessions dict)
    return g.compile(checkpointer=InMemorySaver())


app = build_graph()


def clear_session(session_id: str) -> None:
    """Forget a conversation: its messages and its PII placeholder map."""
    app.checkpointer.delete_thread(session_id)


# ---------------------------------------------------------------------------
# Main entry point (same return shape as agent.chat, plus latency and tokens)
# ---------------------------------------------------------------------------

def chat_agent(user_input: str, session_id: str = "default") -> dict:
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    # Count existing messages so we only report this turn's activity
    before = len(app.get_state(config).values.get("messages", []))

    start = time.perf_counter()
    _wait_by_session.pop(session_id, None)
    _timings_by_session.pop(session_id, None)
    try:
        result = app.invoke(
            {"messages": [("user", user_input)], "guard_revisions": 0, "guard_status": "",
             "pii_count": 0, "input_label": "", "tools_withheld": 0},
            config,
        )
    except Exception as e:
        return {
            "answer":     f"Agent error: {e}",
            "tools_used": [],
            "tool_calls": [],
            "trace":      [f"[ERROR] {type(e).__name__}: {e}"],
            "latency_s":  round(time.perf_counter() - start, 2),
            "wait_s":     round(_wait_by_session.pop(session_id, 0.0), 2),
            "tokens":     {},
            "timings":    _format_timings(_timings_by_session.pop(session_id, {})),
            "error":      str(e),
        }
    latency = time.perf_counter() - start
    waited = _wait_by_session.pop(session_id, 0.0)
    timings = _format_timings(_timings_by_session.pop(session_id, {}))

    new_msgs = result["messages"][before:]
    trace, tool_calls = [], []
    in_tok = out_tok = 0

    for m in new_msgs:
        if isinstance(m, AIMessage):
            m_in, m_out = extract_usage(m)
            in_tok  += m_in
            out_tok += m_out
            for tc in m.tool_calls:
                tool_calls.append({"tool": tc["name"], "args": tc["args"]})
                trace.append(f"[AGENT] Calls {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            preview = str(m.content)[:150].replace("\n", " ")
            trace.append(f"[{m.name.upper()}] Returned: {preview}...")
        elif isinstance(m, HumanMessage) and m.name == "guardrail":
            trace.append(f"[GUARD] Revision requested: {str(m.content)[:160]}...")

    guard = result.get("guard_status", "")
    pii_count = result.get("pii_count", 0)
    label = result.get("input_label", "")
    withheld = result.get("tools_withheld", 0)
    trace.insert(0, f"[INPUT] Label: {label}" + (" -> refused" if guard == "refused" else ""))
    if withheld:
        trace.append(f"[GUARD] Tool results withheld as suspected injection: {withheld}")
    trace.append(f"[GUARD] PII redacted: {pii_count} item(s)")
    trace.append(f"[GUARD] Number grounding: {guard}")
    trace.append("[TIMING] " + " | ".join(
        f"{node} {t['seconds']:.2f}s" + (f" (x{t['calls']})" if t["calls"] > 1 else "")
        for node, t in timings.items()
    ))

    answer = new_msgs[-1].content if new_msgs else ""
    cost = cost_usd(GROQ_MODEL, in_tok, out_tok)
    trace.append(
        f"[DONE] {len(tool_calls)} tool call(s), {latency - waited:.2f}s processing"
        f" + {waited:.1f}s rate-limit wait, {in_tok} input + {out_tok} output tokens, ${cost:.5f}"
    )

    return {
        "answer":       answer,
        "tools_used":   sorted({t["tool"] for t in tool_calls}),
        "tool_calls":   tool_calls,
        "trace":        trace,
        "latency_s":    round(latency, 2),           # total, including waits
        "wait_s":       round(waited, 2),            # rate-limit waiting only
        "processing_s": round(latency - waited, 2),  # use this for latency metrics
        "tokens":       {"input": in_tok, "output": out_tok},
        "cost_usd":     round(cost, 6),
        "guard":        guard,
        "pii_redacted": pii_count,
        "input_label":  label,
        "timings":      timings,
        # Tool outputs the agent actually saw (after injection scan and PII
        # redaction). The evaluation's RAGAS metrics need these as context.
        "contexts":     [str(m.content) for m in new_msgs if isinstance(m, ToolMessage)],
    }


def _format_timings(raw: dict) -> dict:
    """{node: [seconds, calls]} -> {node: {"seconds": s, "calls": n}}, in graph order."""
    order = ["input_guard", "agent", "tools", "tool_result_guard", "output_guard"]
    nodes = [n for n in order if n in raw] + [n for n in raw if n not in order]
    return {n: {"seconds": round(raw[n][0], 3), "calls": raw[n][1]} for n in nodes}


# ---------------------------------------------------------------------------
# Side-by-side test: rule-based router vs LLM agent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    test_questions = [
        "Hello!",
        "How many awards are in each category?",
        "What kind of messages do engineering teams write?",
        "Who could replace a Senior Director?",
        "Which departments recognize each other most?",
        "Who are the bridge roles that span multiple departments?",
        # Multi-step: the rule-based router cannot sequence these
        "Who could replace a Senior Director, and how does Leadership connect to other departments?",
        "Which category is least recognized, and show me example messages from it",
    ]

    parser = argparse.ArgumentParser(description="Router vs LangGraph agent test")
    parser.add_argument("--only", type=int, nargs="+", help="run only these question numbers (see --list)")
    parser.add_argument("--q", help="run one custom question instead of the test set")
    parser.add_argument("--list", action="store_true", help="list the test questions and exit")
    args = parser.parse_args()

    if args.list:
        for i, q in enumerate(test_questions):
            print(f"  {i}: {q}")
        raise SystemExit
    if args.q:
        selected = [(0, args.q)]
    elif args.only:
        selected = [(i, test_questions[i]) for i in args.only]
    else:
        selected = list(enumerate(test_questions))

    router = QueryRouter()

    print("=" * 60)
    print("  Router vs LangGraph Agent")
    print(f"  Model: {GROQ_MODEL} (temperature {TEMPERATURE})")
    print("=" * 60)

    # No fixed pause needed: the rate limiter waits only when the budget is full
    for i, q in selected:
        decision = router.route(q)
        router_tools = sorted({ROUTER_TO_AGENT.get(t, t) for t in decision.tools})

        # Fresh session per question so answers don't leak into each other
        result = chat_agent(q, session_id=f"test-{i}")

        if result.get("error"):
            match = "ERROR"
        else:
            match = "MATCH" if router_tools == result["tools_used"] else "DIFFERENT"
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print("-" * 60)
        print(f"Router: {decision.route.value.upper()} -> {router_tools or ['none']}")
        print(f"Agent:  {result['tools_used'] or ['none']}   [{match}]   "
              f"Input: {result.get('input_label', '')}   Guard: {result.get('guard', '')}")
        print(f"Processing: {result.get('processing_s', result['latency_s'])}s "
              f"(+{result.get('wait_s', 0)}s waiting) | Tokens: {result['tokens']} | "
              f"Cost: ${result.get('cost_usd', 0):.5f}")
        print(f"\nAnswer:\n{result['answer']}")
        print("\nTrace:")
        for step in result["trace"]:
            print(f"  {step}")

    print("\n" + "=" * 60)
    print(summarize())
