"""
src/rag/guardrails/pii.py - PII redaction with consistent pseudonyms
====================================================================
Replaces personal names, emails, and phone numbers with placeholders before
text reaches the LLM. Within a conversation each person keeps one placeholder
("Maya Lopez" is always [PERSON_1]), so the LLM can still tell people apart.

Detection: Microsoft Presidio, running locally. A spaCy model recognizes
names from context; built-in patterns catch emails and phone numbers.
Settings live in config/guardrails.yaml.

Privacy rules for this module:
  - Logs record only COUNTS by type, never the redacted values.
  - The name -> placeholder map stays in server-side graph state and is
    never sent to the LLM.
"""

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(os.getenv("GUARDRAILS_CONFIG", "config/guardrails.yaml"))
GUARD_LOG = Path("logs/guardrail_events.jsonl")

# Short labels for placeholders
_LABELS = {"PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE"}
# Our own placeholders must never be re-detected as names
_PLACEHOLDER = re.compile(r"\[?(PERSON|EMAIL|PHONE)_\d+\]?")


@lru_cache(maxsize=1)
def _settings() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["pii"]


@lru_cache(maxsize=1)
def default_allowlist() -> frozenset:
    """Configured allowlist plus department names from schema.yaml."""
    words = {w.lower() for w in _settings().get("allowlist", [])}
    try:
        from src.rag.schema import load_schema
        for dept in load_schema().get("department_rules", {}):
            words.add(dept.lower())
            words.update(p.strip().lower() for p in re.split(r"[/&]", dept))
    except Exception:
        pass
    return frozenset(w for w in words if w)


@lru_cache(maxsize=1)
def _analyzer():
    """Loaded on first use: the spaCy model takes a few seconds to load."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": _settings()["spacy_model"]}],
    })
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


def _detect(text: str, allowlist: frozenset) -> list:
    """Presidio findings, minus allowlisted words, our own placeholders,
    and overlaps (keeps the longer, then higher-scoring, span)."""
    cfg = _settings()
    results = _analyzer().analyze(
        text=text, entities=cfg["entities"], language="en",
        score_threshold=cfg.get("score_threshold", 0.4),
    )
    kept = []
    for r in sorted(results, key=lambda r: (-(r.end - r.start), -r.score)):
        span = text[r.start:r.end]
        if span.strip().lower() in allowlist or _PLACEHOLDER.fullmatch(span.strip()):
            continue
        if any(r.start < k.end and k.start < r.end for k in kept):
            continue
        kept.append(r)
    return sorted(kept, key=lambda r: r.start)


def _placeholder(entity: str, value: str, mapping: dict) -> str:
    key = f"{entity}:{value.strip().lower()}"
    if key not in mapping:
        label = _LABELS.get(entity, entity)
        n = sum(1 for k in mapping if k.startswith(f"{entity}:")) + 1
        mapping[key] = f"[{label}_{n}]"
    return mapping[key]


def redact_text(text: str, mapping: dict, allowlist: frozenset | None = None) -> tuple[str, list[str]]:
    """Replace PII in text with placeholders.

    mapping: name -> placeholder dict, updated in place so placeholders stay
             consistent across texts (pass the same dict for one conversation).
    Returns (redacted_text, list of entity types found).
    """
    if not text or not text.strip():
        return text, []
    allowlist = default_allowlist() if allowlist is None else allowlist
    found = _detect(text, allowlist)
    # Number placeholders in reading order (first person mentioned = [PERSON_1]),
    # THEN replace from the end so earlier character offsets stay valid.
    replacements = [(r, _placeholder(r.entity_type, text[r.start:r.end], mapping)) for r in found]
    for r, placeholder in reversed(replacements):
        text = text[:r.start] + placeholder + text[r.end:]
    return text, [r.entity_type for r in found]


def log_event(location: str, found: list[str], run_id: str = "", session: str = "") -> None:
    """Record WHAT KIND of PII was redacted and where. Never the values."""
    GUARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "session": session,
        "guard": "pii_redaction",
        "location": location,              # tool_results | final_answer
        "counts": dict(Counter(found)),
    }
    with open(GUARD_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
