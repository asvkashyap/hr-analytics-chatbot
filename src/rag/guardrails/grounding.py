"""
src/rag/guardrails/grounding.py - Number grounding check (strict)
=================================================================
Every number in a final answer must come from the tool results for that
question. "Tools compute, the LLM reports": numbers the LLM computed itself
(percentages, sums, differences) fail, even if the arithmetic is right.

What counts as a match:
  - the same value                         442      vs 442
  - a rounded version of an evidence value 44%      vs 44.2%
  - the same value as fraction or percent  0.442    vs 44.2%

What is not checked (not claims about the data):
  - list and rank markers                  "1." / "| 3 |" in a rank column
  - numbers the user put in the question   "top 5"
  - numbers inside quoted message text     already in the evidence

Pure functions, no LLM calls: free, instant, and unit-testable.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

GUARD_LOG = Path("logs/guardrail_events.jsonl")
# Full transcripts of failed checks, for promoting into replay tests
GUARD_CASES = Path("logs/guard_cases.jsonl")

# A number not glued to letters, digits, or a decimal point on either side
# (so "Engineer III", "Q3", "gpt-oss-120b" and "v2.1" are ignored).
_NUMBER = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?:\s?(%))?(?![\w])"
)
# Quoted spans, straight or curly quotes, single-line
_QUOTED = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”")
# List markers at the start of a line: "1." "2)" "- 3."
_LIST_MARKER = re.compile(r"^\s*(?:[-*]\s*)?\d+[.)]\s", re.MULTILINE)


@dataclass
class Number:
    raw: str
    value: float
    decimals: int
    percent: bool


@dataclass
class GroundingReport:
    passed: bool
    checked: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def _parse(match: re.Match) -> Number:
    whole, frac, pct = match.group(1), match.group(2), match.group(3)
    value = float(whole.replace(",", "") + ("." + frac if frac else ""))
    return Number(match.group(0).strip(), value, len(frac or ""), bool(pct))


def extract_numbers(text: str) -> list[Number]:
    return [_parse(m) for m in _NUMBER.finditer(text)]


def _rank_column_cells(text: str) -> set[int]:
    """Character offsets of numbers that sit alone in the FIRST cell of a
    markdown table row, which is where rank columns go."""
    offsets = set()
    for line in re.finditer(r"^\s*\|([^|\n]*)\|", text, re.MULTILINE):
        cell = line.group(1)
        if re.fullmatch(r"\s*\d+\s*", cell):
            offsets.add(line.start(1) + len(cell) - len(cell.lstrip()))
    return offsets


def answer_claims(answer: str) -> list[Number]:
    """Numbers in the answer that are claims about the data."""
    # Blank out quoted text (keeping offsets stable) and list markers
    masked = _QUOTED.sub(lambda m: " " * len(m.group(0)), answer)
    masked = _LIST_MARKER.sub(lambda m: " " * len(m.group(0)), masked)
    ranks = _rank_column_cells(masked)
    return [_parse(m) for m in _NUMBER.finditer(masked) if m.start() not in ranks]


def _matches(claim: Number, evidence: Number) -> bool:
    """Same value, allowing the claim to be rounded at its own precision and
    allowing fraction <-> percent (0.442 <-> 44.2%)."""
    tolerance = 0.5 * 10 ** (-claim.decimals) + 1e-9
    candidates = [evidence.value]
    if claim.percent and not evidence.percent:
        candidates.append(evidence.value * 100)
    if evidence.percent and not claim.percent:
        candidates.append(evidence.value / 100)
    return any(abs(claim.value - c) <= tolerance for c in candidates)


def check_numbers(answer: str, evidence_texts: list[str], question: str = "") -> GroundingReport:
    evidence = [n for t in evidence_texts for n in extract_numbers(t)]
    asked = {n.value for n in extract_numbers(question)}
    report = GroundingReport(passed=True)
    for claim in answer_claims(answer):
        if claim.value in asked:
            continue
        report.checked.append(claim.raw)
        if not any(_matches(claim, e) for e in evidence):
            report.unsupported.append(claim.raw)
    # One entry per distinct unsupported number, in order of appearance
    report.unsupported = list(dict.fromkeys(report.unsupported))
    report.passed = not report.unsupported
    return report


def feedback_message(report: GroundingReport) -> str:
    nums = ", ".join(report.unsupported)
    return (
        f"[Guardrail check] These numbers in your answer do not appear in the tool "
        f"results: {nums}. Report only figures the tools returned, exactly as given. "
        f"Do not compute new percentages, totals, or differences yourself. Remove "
        f"or replace these numbers and answer again."
    )


def warning_note(report: GroundingReport) -> str:
    nums = ", ".join(report.unsupported)
    return (
        f"\n\n---\n**Note:** These figures could not be verified against the data "
        f"and may be inaccurate: {nums}."
    )


def _redact_for_log(text: str) -> str:
    """Users can type names into questions ("Is Maya a low performer?").
    Logs must never store them. If redaction fails, withhold the text:
    for logs, a missing question is better than a leaked name."""
    try:
        from src.rag.guardrails import pii
        return pii.redact_text(text, {})[0]
    except Exception:
        return "[withheld: redaction unavailable]"


def log_event(question: str, report: GroundingReport, action: str, revisions: int,
              run_id: str = "", session: str = "") -> None:
    GUARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "session": session,
        "guard": "number_grounding",
        "action": action,              # passed | revise | warned
        "revisions": revisions,
        "question": _redact_for_log(question),
        "numbers_checked": len(report.checked),
        "unsupported": report.unsupported,
    }
    with open(GUARD_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def capture_case(question: str, evidence_texts: list[str], answer: str,
                 report: GroundingReport, action: str, run_id: str = "") -> None:
    """Save the FULL case when a check fails, so it can be replayed later.
    (The event log keeps only a summary; replay needs complete tool output.)

    To turn a captured case into a permanent test: copy its line into
    tests/fixtures/guard_replay.jsonl and add "expected_pass" and
    "expected_unsupported" with the result you consider correct."""
    GUARD_CASES.parent.mkdir(parents=True, exist_ok=True)
    case = {
        "id": f"{run_id}-{datetime.now(timezone.utc).strftime('%H%M%S%f')}",
        "source": f"captured from run {run_id} ({action})",
        "question": _redact_for_log(question),
        "evidence": evidence_texts,
        "answer": answer,
        "flagged": report.unsupported,
    }
    with open(GUARD_CASES, "a", encoding="utf-8") as f:
        f.write(json.dumps(case) + "\n")
