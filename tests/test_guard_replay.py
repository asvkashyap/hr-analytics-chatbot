"""
Replay tests: real guard cases, checked instantly with no LLM.

Cases live in tests/fixtures/guard_replay.jsonl, one JSON object per line.
When a real run fails a check, the full case is saved to
logs/guard_cases.jsonl. To keep it as a permanent test, copy its line into
the fixture file and add "expected_pass" and "expected_unsupported".

Run: python -m pytest tests/test_guard_replay.py -v
"""

import json
from pathlib import Path

import pytest

from src.rag.guardrails.grounding import check_numbers

FIXTURES = Path(__file__).parent / "fixtures" / "guard_replay.jsonl"


def load_cases():
    with open(FIXTURES, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c["id"])
def test_replay(case):
    report = check_numbers(case["answer"], case["evidence"], case["question"])
    assert report.passed == case["expected_pass"], f"unsupported: {report.unsupported}"
    assert report.unsupported == case["expected_unsupported"]
