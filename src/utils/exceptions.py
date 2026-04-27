"""
Custom exception hierarchy for the HR Taxonomy Pipeline.

Design principles:
  - Every module raises domain-specific exceptions (not generic ValueError/RuntimeError).
  - Exceptions carry enough context to debug without reading source code.
  - Callers can catch broad base classes OR specific exceptions as needed.

Hierarchy:
    PipelineError                   ← catch-all for the whole pipeline
    ├── ConfigurationError          ← bad config.yaml / missing env vars
    ├── DataError                   ← data loading, validation, schema issues
    │   ├── DataLoadError
    │   ├── DataValidationError
    │   └── DataSchemaError
    ├── LLMError                    ← anything related to LLM API calls
    │   ├── LLMAuthError            ← invalid / missing API key
    │   ├── LLMRateLimitError       ← 429 / quota exceeded
    │   ├── LLMTimeoutError         ← request timed out
    │   ├── LLMResponseError        ← unexpected / unparseable response
    │   └── LLMCostLimitError       ← exceeded budget guard
    ├── TaxonomyError               ← taxonomy creation & compression issues
    │   ├── TaxonomyParseError
    │   └── TaxonomyMergeError
    └── AnnotationError             ← annotation / classification issues
        └── BatchProcessingError
"""


# ── Base ──────────────────────────────────────────────────────────────────────


class PipelineError(Exception):
    """Base exception for all pipeline errors."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} | context: {ctx}"
        return base


# ── Configuration ─────────────────────────────────────────────────────────────


class ConfigurationError(PipelineError):
    """Raised when config.yaml is malformed or required env vars are missing."""


# ── Data ──────────────────────────────────────────────────────────────────────


class DataError(PipelineError):
    """Base class for all data-related errors."""


class DataLoadError(DataError):
    """Raised when a data file cannot be read (missing, corrupt, wrong format)."""


class DataValidationError(DataError):
    """Raised when data fails quality checks (nulls, out-of-range values, etc.)."""


class DataSchemaError(DataError):
    """Raised when a required column is missing from the dataset."""

    def __init__(self, missing_columns: list[str], source: str = "") -> None:
        self.missing_columns = missing_columns
        context = {"missing": missing_columns, "source": source}
        super().__init__(
            f"Required columns not found: {missing_columns}", context=context
        )


# ── LLM ───────────────────────────────────────────────────────────────────────


class LLMError(PipelineError):
    """Base class for all LLM API errors."""


class LLMAuthError(LLMError):
    """Raised when the API key is invalid or missing."""


class LLMRateLimitError(LLMError):
    """Raised when the API rate limit or quota is exceeded."""


class LLMTimeoutError(LLMError):
    """Raised when an API request times out."""

    def __init__(self, timeout_seconds: float, batch_id: int | None = None) -> None:
        context = {"timeout_s": timeout_seconds, "batch_id": batch_id}
        super().__init__(
            f"LLM request timed out after {timeout_seconds}s", context=context
        )


class LLMResponseError(LLMError):
    """Raised when the LLM returns an unexpected or unparseable response."""

    def __init__(self, raw_response: str, batch_id: int | None = None) -> None:
        self.raw_response = raw_response
        preview = raw_response[:200] if raw_response else "<empty>"
        context = {"batch_id": batch_id, "response_preview": preview}
        super().__init__("Failed to parse LLM response as JSON", context=context)


class LLMCostLimitError(LLMError):
    """Raised when the accumulated cost exceeds a configured budget guard."""

    def __init__(self, current_cost: float, limit: float) -> None:
        context = {"current_cost_usd": round(current_cost, 4), "limit_usd": limit}
        super().__init__(
            f"Cost limit exceeded: ${current_cost:.4f} > ${limit:.2f}", context=context
        )


# ── Taxonomy ──────────────────────────────────────────────────────────────────


class TaxonomyError(PipelineError):
    """Base class for taxonomy-related errors."""


class TaxonomyParseError(TaxonomyError):
    """Raised when a taxonomy JSON cannot be parsed or has invalid structure."""


class TaxonomyMergeError(TaxonomyError):
    """Raised when merging two taxonomies fails (e.g., incompatible schemas)."""


# ── Annotation ────────────────────────────────────────────────────────────────


class AnnotationError(PipelineError):
    """Base class for annotation/classification errors."""


class BatchProcessingError(AnnotationError):
    """Raised when a batch of records fails annotation."""

    def __init__(self, batch_id: int, reason: str) -> None:
        context = {"batch_id": batch_id, "reason": reason}
        super().__init__(f"Batch {batch_id} failed: {reason}", context=context)