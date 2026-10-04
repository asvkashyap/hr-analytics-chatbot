"""
Unit tests for PII redaction. Uses the real Presidio + spaCy model
(the first test takes a few seconds while the model loads).

Run: python -m pytest tests/test_pii.py -v
"""

from src.rag.guardrails.pii import redact_text


def test_person_names_are_redacted():
    text, found = redact_text("Thank you Maya Lopez for the launch.", {})
    assert "Maya Lopez" not in text
    assert "[PERSON_1]" in text
    assert "PERSON" in found


def test_emails_are_redacted():
    text, found = redact_text("Send questions to maya.lopez@example.com today.", {})
    assert "maya.lopez@example.com" not in text
    assert "EMAIL_ADDRESS" in found


def test_same_name_gets_same_placeholder():
    mapping = {}
    first, _ = redact_text("Kelly Chen ran the workshop.", mapping)
    second, _ = redact_text("Thanks again, Kelly Chen.", mapping)
    assert first.startswith("[PERSON_1]")
    assert "[PERSON_1]" in second


def test_different_people_get_different_placeholders():
    text, _ = redact_text("Maya Lopez and Kelly Chen both helped.", {})
    assert "[PERSON_1]" in text and "[PERSON_2]" in text


def test_placeholders_are_numbered_in_reading_order():
    # The first person mentioned is [PERSON_1], the second [PERSON_2]
    text, _ = redact_text("Maya Lopez and Kelly Chen both helped.", {})
    assert text.index("[PERSON_1]") < text.index("[PERSON_2]")


def test_allowlisted_words_are_kept():
    text, found = redact_text("Thank you Maya Lopez.", {}, allowlist=frozenset({"maya lopez"}))
    assert "Maya Lopez" in text
    assert found == []


def test_placeholders_are_not_redacted_again():
    text, found = redact_text("[PERSON_1] thanked [PERSON_2].", {})
    assert text == "[PERSON_1] thanked [PERSON_2]."
    assert found == []


def test_departments_and_numbers_are_kept():
    text, _ = redact_text("Engineering gave 179 recognitions to Customer Service.", {})
    assert "Engineering" in text and "179" in text and "Customer Service" in text


def test_empty_text_is_unchanged():
    assert redact_text("", {}) == ("", [])
