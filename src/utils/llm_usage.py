"""
src/utils/llm_usage.py - Cost tracking and rate limiting for every LLM call
===========================================================================
One module used by the pipeline (taxonomy, annotation) and the agents.

  record_usage(...)     log tokens, latency, and list-price cost for one call
  RateLimiter           waits BEFORE a call so you stay under tokens/requests
                        per minute, instead of hitting 429 errors and retrying
  retry_after(error)    reads "try again in 5.5s" style hints from 429 errors
  summarize()           cost report per run, component, and model

Every call is appended to logs/llm_usage.jsonl, one JSON object per line.

Report:
    python -m src.utils.llm_usage            # latest run
    python -m src.utils.llm_usage --all      # every run
"""

import argparse
import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml

PRICING_PATH = Path(os.getenv("PRICING_PATH", "config/pricing.yaml"))
USAGE_LOG = Path(os.getenv("LLM_USAGE_LOG", "logs/llm_usage.jsonl"))

# One run id per process, so each pipeline or eval run can be summarized alone
RUN_ID = os.getenv("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")


@lru_cache(maxsize=1)
def _config() -> dict:
    with open(PRICING_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_warned_models: set[str] = set()


def price_for(model: str) -> dict:
    cfg = _config()
    price = cfg.get("models", {}).get(model)
    if price is None:
        if model not in _warned_models:
            print(f"  [COST WARNING] No price for '{model}' in {PRICING_PATH}; "
                  f"using the default price. Add it for accurate cost.")
            _warned_models.add(model)
        return cfg["default"]
    return price


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = price_for(model)
    return (input_tokens * p["input_per_million"] + output_tokens * p["output_per_million"]) / 1_000_000


def extract_usage(response) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from any provider's response.

    Handles:
      - LangChain messages (ChatGroq, ChatGoogleGenerativeAI, ChatOllama, ...)
      - Groq / OpenAI SDK responses        (usage.prompt_tokens, completion_tokens)
      - Google GenAI SDK responses         (usage_metadata.*_token_count)
      - Raw JSON dicts from REST calls     (Gemini usageMetadata, OpenAI usage, Ollama)
    Thinking or reasoning tokens are counted as output, since that is how
    providers bill them.
    """
    # LangChain: one standard format for every provider it wraps
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, dict):
        return meta.get("input_tokens", 0) or 0, meta.get("output_tokens", 0) or 0

    # Google GenAI SDK object
    if meta is not None and hasattr(meta, "prompt_token_count"):
        out = (getattr(meta, "candidates_token_count", 0) or 0) + \
              (getattr(meta, "thoughts_token_count", 0) or 0)
        return getattr(meta, "prompt_token_count", 0) or 0, out

    # Groq / OpenAI SDK object
    usage = getattr(response, "usage", None)
    if usage is not None and hasattr(usage, "prompt_tokens"):
        return usage.prompt_tokens or 0, usage.completion_tokens or 0

    # Raw JSON dicts
    if isinstance(response, dict):
        if "usageMetadata" in response:                     # Gemini REST
            u = response["usageMetadata"]
            return (u.get("promptTokenCount", 0),
                    u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0))
        if "usage" in response:                             # OpenAI-style REST
            u = response["usage"]
            return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        if "prompt_eval_count" in response:                 # Ollama
            return response.get("prompt_eval_count", 0), response.get("eval_count", 0)

    return 0, 0


def record_usage(
    component: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_s: float | None = None,
    **extra,
) -> float:
    """Log one LLM call and return its list-price cost in USD."""
    input_tokens, output_tokens = int(input_tokens or 0), int(output_tokens or 0)
    cost = cost_usd(model, input_tokens, output_tokens)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "component": component,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 8),
        "latency_s": round(latency_s, 3) if latency_s is not None else None,
        **extra,
    }
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return cost


# ---------------------------------------------------------------------------
# Rate limiting: wait before calling, so 429s become rare
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding 60-second window over tokens and requests for ONE model.

    Groq applies limits per model, so each model gets its own window:
    classifier calls on a small model must not use up the main model's budget.
    Limits come from rate_limits[model] if set, else rate_limits[provider]."""

    def __init__(self, provider: str, model: str | None = None):
        # Providers with no entry (e.g. local Ollama models) are never throttled
        limits_cfg = _config().get("rate_limits", {})
        limits = (model and limits_cfg.get(model)) or limits_cfg.get(provider, {})
        self.provider = model or provider
        self.tpm = limits.get("tokens_per_minute")
        self.rpm = limits.get("requests_per_minute")
        self._events: deque[tuple[float, int]] = deque()   # (timestamp, tokens)

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60:
            self._events.popleft()

    def wait(self, estimated_tokens: int) -> float:
        """Block until this call fits in the window. Returns seconds waited."""
        waited = 0.0
        while True:
            now = time.monotonic()
            self._prune(now)
            used = sum(t for _, t in self._events)
            over_tokens = self.tpm and used + estimated_tokens > self.tpm and self._events
            over_requests = self.rpm and len(self._events) >= self.rpm
            if not (over_tokens or over_requests):
                break
            pause = max(60 - (now - self._events[0][0]), 0.5)
            print(f"  [WAIT] {self.provider}: {used:,} tokens / {len(self._events)} requests "
                  f"in the last minute, pausing {pause:.1f}s")
            time.sleep(pause)
            waited += pause
        self._events.append((time.monotonic(), estimated_tokens))
        return waited

    def correct(self, actual_tokens: int) -> None:
        """Replace the last estimate with the real token count once known."""
        if self._events:
            ts, _ = self._events.pop()
            self._events.append((ts, actual_tokens))


def estimate_tokens(text: str, expected_output: int = 500) -> int:
    """Rough count: about 4 characters per token, plus expected output."""
    return len(text) // 4 + expected_output


def retry_after(error: Exception, default: float = 10.0) -> float:
    """Read the provider's suggested wait from a 429 error, if present."""
    msg = str(error)
    m = re.search(r"try again in ([\d.]+)\s*s", msg)          # Groq style
    if m:
        return float(m.group(1)) + 0.5
    m = re.search(r'retryDelay["\']?\s*:\s*["\']?(\d+)s', msg)  # Gemini style
    if m:
        return float(m.group(1)) + 0.5
    return default


def is_rate_limit(error: Exception) -> bool:
    msg = str(error).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg or "resource_exhausted" in msg


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def load_usage() -> list[dict]:
    if not USAGE_LOG.exists():
        return []
    with open(USAGE_LOG, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(run_id: str | None = None) -> str:
    rows = load_usage()
    if not rows:
        return "No LLM usage logged yet."
    if run_id is None:
        run_id = rows[-1]["run_id"]
    if run_id != "all":
        rows = [r for r in rows if r["run_id"] == run_id]

    groups: dict[tuple[str, str], dict] = {}
    for r in rows:
        g = groups.setdefault((r["component"], r["model"]),
                              {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
        g["calls"] += 1
        g["in"] += r["input_tokens"]
        g["out"] += r["output_tokens"]
        g["cost"] += r["cost_usd"]

    lines = [f"LLM usage for run: {run_id}", ""]
    lines.append(f"{'component':<22}{'model':<24}{'calls':>7}{'input tok':>12}{'output tok':>12}{'cost $':>11}")
    lines.append("-" * 88)
    total = {"calls": 0, "in": 0, "out": 0, "cost": 0.0}
    for (comp, model), g in sorted(groups.items()):
        lines.append(f"{comp:<22}{model:<24}{g['calls']:>7}{g['in']:>12,}{g['out']:>12,}{g['cost']:>11.4f}")
        for k in total:
            total[k] += g[k]
    lines.append("-" * 88)
    lines.append(f"{'TOTAL':<46}{total['calls']:>7}{total['in']:>12,}{total['out']:>12,}{total['cost']:>11.4f}")
    if total["calls"]:
        lines.append(f"\nAverage cost per call: ${total['cost'] / total['calls']:.5f}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize LLM usage and cost")
    parser.add_argument("--run", help="run id to summarize (default: latest)")
    parser.add_argument("--all", action="store_true", help="summarize every run together")
    args = parser.parse_args()
    print(summarize("all" if args.all else args.run))
