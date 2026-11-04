"""
Unit tests for the number grounding guardrail.
Run: python -m pytest tests/test_grounding.py      (or: python -m tests.test_grounding)
"""

from src.rag.guardrails.grounding import check_numbers, answer_claims

EVIDENCE = [
    "=== Category Distribution ===\n"
    "  Organizational & Team Enablement: 442 records (44.2%)\n"
    "  Creative & Brand Impact: 53 records (5.3%)\n"
    "  Total recognitions: 1,999\n"
    "  Engineering: 0.519\n"
]


def test_exact_number_passes():
    assert check_numbers("There are 442 records.", EVIDENCE).passed


def test_rounded_percent_passes():
    assert check_numbers("About 44% of recognitions.", EVIDENCE).passed


def test_fraction_percent_equivalence_passes():
    assert check_numbers("A share of 0.442 overall.", EVIDENCE).passed
    assert check_numbers("Centrality is 51.9%.", EVIDENCE).passed


def test_comma_number_passes():
    assert check_numbers("All 1,999 recognitions.", EVIDENCE).passed


def test_invented_number_fails():
    r = check_numbers("Roughly 70% of Leadership's recognitions.", EVIDENCE)
    assert not r.passed and r.unsupported == ["70%"]


def test_computed_percent_fails_strict():
    # 53 / 442 = 12.0% is correct arithmetic, but the tool never returned it
    assert not check_numbers("Creative is 12% the size of Enablement.", EVIDENCE).passed


def test_wrong_precision_fails():
    # 44.7 does not round from 44.2
    assert not check_numbers("44.7% of recognitions.", EVIDENCE).passed


def test_question_numbers_skipped():
    assert check_numbers("Here are the top 7 roles.", EVIDENCE, question="show the top 7").passed


def test_list_and_rank_markers_skipped():
    answer = "1. First item\n2) Second\n| 8 | Engineering | 442 |\n| 9 | Design | 53 |"
    assert [n.raw for n in answer_claims(answer)] == ["442", "53"]


def test_quoted_text_skipped():
    assert check_numbers('They wrote "we shipped 37 features in 2025" about it.', EVIDENCE).passed


def test_numbers_inside_words_ignored():
    assert check_numbers("Helpdesk Engineer III and Q3 planning on gpt-oss-120b.", EVIDENCE).passed


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
