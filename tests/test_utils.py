"""
Unit tests for utility modules.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd

from src.utils.exceptions import (
    DataSchemaError,
    DataValidationError,
    LLMResponseError,
)
from src.utils.json_utils import extract_json
from src.utils.data_utils import validate_schema, validate_data_quality, clean_dataframe


# ── JSON Utils ────────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json(self):
        raw = '{"key": "value"}'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_json_in_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_json_in_plain_fence(self):
        raw = '```\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_json_in_prose(self):
        raw = 'Here is your result:\n{"key": "value"}\nHope that helps!'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_json5_trailing_comma(self):
        raw = '{"key": "value",}'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_raises_on_empty(self):
        with pytest.raises(LLMResponseError):
            extract_json("")

    def test_raises_on_invalid(self):
        with pytest.raises(LLMResponseError):
            extract_json("This is not JSON at all.")

    def test_list_response(self):
        raw = '[1, 2, 3]'
        result = extract_json(raw)
        assert result == [1, 2, 3]


# ── Data Utils ────────────────────────────────────────────────────────────────

class TestValidateSchema:
    def test_passes_with_required_cols(self):
        df = pd.DataFrame({"message": ["hello"], "award_reason": ["teamwork"]})
        # Should not raise
        validate_schema(df, source="test")

    def test_raises_on_missing_col(self):
        df = pd.DataFrame({"message": ["hello"]})
        with pytest.raises(DataSchemaError) as exc_info:
            validate_schema(df, source="test")
        assert "award_reason" in exc_info.value.missing_columns


class TestValidateDataQuality:
    def test_returns_report(self):
        df = pd.DataFrame({
            "message": ["hello", "world", "foo"],
            "award_reason": ["a", "b", "c"],
        })
        report = validate_data_quality(df)
        assert "duplicate_messages" in report
        assert "empty_message_pct" in report

    def test_raises_on_too_many_nulls(self):
        df = pd.DataFrame({
            "message": [None] * 60 + ["hello"] * 40,
            "award_reason": ["a"] * 100,
        })
        with pytest.raises(DataValidationError):
            validate_data_quality(df)


class TestCleanDataframe:
    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "message": ["hello", "hello", "world"],
            "award_reason": ["a", "a", "b"],
        })
        cleaned = clean_dataframe(df)
        assert len(cleaned) == 2

    def test_removes_excluded_reasons(self):
        df = pd.DataFrame({
            "message": ["msg1", "msg2"],
            "award_reason": ["Birthday", "Teamwork"],
        })
        cleaned = clean_dataframe(df, exclude_award_reasons=["Birthday"])
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["award_reason"] == "Teamwork"

    def test_removes_null_messages(self):
        df = pd.DataFrame({
            "message": [None, "valid message"],
            "award_reason": ["a", "b"],
        })
        cleaned = clean_dataframe(df)
        assert len(cleaned) == 1