"""
src/rag/guardrails/input_guard.py - Checks that run BEFORE the agent
====================================================================
1. Injection detection: Prompt Guard 2 (a small classifier on Groq) returns
   the probability that text is a jailbreak or prompt injection. Used on the
   user's question AND on tool results (recognition messages are written by
   employees, so they can carry "indirect" injections).
2. Scope and sensitivity: a small LLM sorts the question into one label.
   Code, not the LLM, decides what each label does (see ACTIONS), and every
   refusal is a fixed message from config/guardrails.yaml.

Failure policy: if a check errors, the question proceeds (fail open) and the
error is logged. The tools are read-only and the output guards still run, so
a guard outage should not take the whole assistant down.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml

from src.utils.llm_usage import (
    RateLimiter, record_usage, extract_usage, estimate_tokens, retry_after, is_rate_limit,
)

CONFIG_PATH = Path(os.getenv("GUARDRAILS_CONFIG", "config/guardrails.yaml"))
GUARD_LOG = Path("logs/guardrail_events.jsonl")

LABELS = (
    "in_scope", "help", "off_topic", "hr_general",
    "workforce_decision", "individual_judgment", "protected_attribute",
)

# What each label does. Decided by code, so the LLM cannot be talked out of it.
ACTIONS = {
    "in_scope":            "proceed",
    "help":                "proceed",
    "workforce_decision":  "proceed_flagged",   # answer, reframed, with a note
    "individual_judgment": "proceed_flagged",   # answer at team level, with an opener
    "off_topic":           "refuse",
    "hr_general":          "refuse",
    "protected_attribute": "refuse",
    "injection":           "refuse",
}

# Extra instructions for the agent on flagged questions
AGENT_INSTRUCTIONS = {
    "workforce_decision": (
        "\nThis question concerns decisions about roles. Describe results only as roles "
        "with similar recognition patterns. Do not call them candidates, successors, or "
        "replacements, and do not recommend anyone for a role."
    ),
    "individual_judgment": (
        "\nThis question asks about a specific person. Do not evaluate, rank, or "
        "characterize that person. Answer only at the team or role level, using "
        "aggregate patterns."
    ),
}

POLICY = """You classify questions sent to an HR recognition analytics assistant.
The assistant answers questions about ONE organization's employee recognition data:
recognition messages, categories, counts, departments, job titles, and who recognizes whom.

Return ONLY a JSON object, no other text:
{"label": "<one label>", "reason": "<one short sentence>"}

Labels:
- in_scope: questions about the organization's recognition data (categories, counts,
  trends, departments, roles, recognition messages, network patterns).
- help: greetings, thanks, or questions about what the assistant can do.
- off_topic: unrelated to recognition data: math, coding, general knowledge,
  writing tasks, personal advice.
- hr_general: HR or recognition topics that are NOT about this organization's data,
  such as best practices, program design, or policies.
- workforce_decision: uses recognition data to inform decisions about roles or people:
  succession, replacements, promotions, backups, who could step in.
- individual_judgment: asks to evaluate a specific named person's performance,
  value, or standing.
- protected_attribute: about gender, age, race, ethnicity, religion, disability, or
  other protected characteristics, including inferring them from names.

If a question fits more than one label, choose the MORE restrictive one. From most to
least restrictive: protected_attribute, individual_judgment, workforce_decision,
off_topic, hr_general, in_scope, help.
Use the previous question only to understand follow-ups like "what about them?".
Classify the current question. Do not answer it."""


@dataclass
class Decision:
    label: str
    action: str
    reason: str = ""
    score: float | None = None      # injection probability for the question
    error: str = ""                  # set when a check failed and we failed open


@lru_cache(maxsize=1)
def settings() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def message(key: str) -> str:
    return settings()["messages"][key].strip()


def agent_instruction(label: str) -> str:
    return AGENT_INSTRUCTIONS.get(label, "")


# ---------------------------------------------------------------------------
# Groq calls, with per-model rate limiting and cost logging
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _client():
    from groq import Groq
    return Groq()


_limiters: dict[str, RateLimiter] = {}


def _chat(model: str, messages: list[dict], component: str, **options) -> str:
    limiter = _limiters.setdefault(model, RateLimiter("groq", model))
    est = estimate_tokens("".join(m["content"] for m in messages),
                          expected_output=options.get("max_tokens", 50))
    for attempt in range(3):
        limiter.wait(est)
        start = time.perf_counter()
        try:
            resp = _client().chat.completions.create(model=model, messages=messages, **options)
        except Exception as e:
            if is_rate_limit(e) and attempt < 2:
                time.sleep(retry_after(e))
                continue
            raise
        in_tok, out_tok = extract_usage(resp)
        limiter.correct(in_tok + out_tok)
        record_usage(component, model, in_tok, out_tok, time.perf_counter() - start)
        return resp.choices[0].message.content or ""
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# 1. Injection detection
# ---------------------------------------------------------------------------

def parse_score(text: str) -> float:
    """Prompt Guard returns the probability as text, e.g. '0.9995'."""
    try:
        return float(text.strip())
    except ValueError:
        m = re.search(r"\d*\.?\d+(?:[eE]-?\d+)?", text)
        if not m:
            raise ValueError(f"No score in Prompt Guard response: {text!r}")
        return float(m.group(0))


def chunks(text: str, size: int) -> list[str]:
    """Prompt Guard reads short inputs, so long text is scored in pieces."""
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def injection_score(text: str) -> float:
    """Highest injection probability across chunks: one bad chunk flags all."""
    cfg = settings()["injection"]
    scores = [
        parse_score(_chat(cfg["model"], [{"role": "user", "content": c}], "guard_injection"))
        for c in chunks(text, cfg.get("chunk_chars", 1500))
    ]
    return max(scores, default=0.0)


def tool_result_flagged(text: str) -> tuple[bool, float | None, str]:
    """(flagged, score, error) for one tool result. Fails open on error."""
    cfg = settings()["injection"]
    if not cfg.get("scan_tool_results", True):
        return False, None, ""
    try:
        score = injection_score(text)
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"
    return score >= cfg.get("threshold", 0.5), score, ""


# ---------------------------------------------------------------------------
# 2. Scope and sensitivity
# ---------------------------------------------------------------------------

def parse_label(text: str) -> tuple[str, str]:
    """Pull {"label", "reason"} out of the model's reply, tolerating code
    fences or stray text. Raises ValueError on anything unusable."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON in classifier response: {text[:200]!r}")
    data = json.loads(m.group(0))
    label = str(data.get("label", "")).strip().lower()
    if label not in LABELS:
        raise ValueError(f"Unknown label from classifier: {label!r}")
    return label, str(data.get("reason", "")).strip()


def classify(question: str, previous: str = "") -> tuple[str, str]:
    cfg = settings()["scope"]
    user = f"Previous question: {previous or '(none)'}\nCurrent question: {question}"
    reply = _chat(
        cfg["model"],
        [{"role": "system", "content": POLICY}, {"role": "user", "content": user}],
        "guard_scope",
        temperature=0,
        max_tokens=cfg.get("max_tokens", 1000),
    )
    return parse_label(reply)


def check_input(question: str, previous: str = "") -> Decision:
    """Injection check first (cheap). If it flags, the classifier never runs."""
    errors = []
    score = None
    try:
        score = injection_score(question)
        if score >= settings()["injection"].get("threshold", 0.5):
            return Decision("injection", "refuse", "prompt injection detected", score)
    except Exception as e:
        errors.append(f"injection check: {type(e).__name__}: {e}")

    try:
        label, reason = classify(question, previous)
    except Exception as e:
        errors.append(f"scope check: {type(e).__name__}: {e}")
        if settings()["scope"].get("fail_open", True):
            return Decision("in_scope", "proceed", "scope check failed; failing open", score, "; ".join(errors))
        return Decision("off_topic", "refuse", "scope check failed; failing closed", score, "; ".join(errors))

    return Decision(label, ACTIONS[label], reason, score, "; ".join(errors))


def refusal_message(label: str) -> str:
    return message(label)


def log_event(decision: Decision, question: str, run_id: str = "", session: str = "") -> None:
    from src.rag.guardrails.grounding import _redact_for_log
    GUARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "session": session,
        "guard": "input",
        "label": decision.label,
        "action": decision.action,
        "injection_score": decision.score,
        "reason": _redact_for_log(decision.reason),
        "question": _redact_for_log(question),
        "error": decision.error,
    }
    if decision.error:
        print(f"  [GUARD ERROR] {decision.error} (failed open)")
    with open(GUARD_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_tool_injection(score: float | None, error: str, run_id: str = "", session: str = "") -> None:
    GUARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "session": session,
        "guard": "tool_injection",
        "action": "withheld" if not error else "scan_failed_open",
        "injection_score": score,
        "error": error,
    }
    if error:
        print(f"  [GUARD ERROR] tool result scan: {error} (failed open)")
    with open(GUARD_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
