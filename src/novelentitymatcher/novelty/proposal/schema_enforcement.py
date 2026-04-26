"""
Schema enforcement for LLM proposal outputs.

Provides retry-aware validation of LLM-generated proposals against
Pydantic schemas, with structured error feedback for re-prompting.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationResult:
    """Result of validating raw LLM output against a schema."""

    def __init__(
        self,
        is_valid: bool,
        parsed: BaseModel | None = None,
        errors: list[dict[str, Any]] | None = None,
    ):
        self.is_valid = is_valid
        self.parsed = parsed
        self.errors = errors or []

    def error_summary(self) -> str:
        return "; ".join(f"{e['field']}: {e['message']}" for e in self.errors)


class SchemaEnforcer:
    """Validate and enforce Pydantic schemas on LLM outputs with retry logic.

    Usage::

        enforcer = SchemaEnforcer(max_retries=2, schema_model=LLMProposalSchema)
        result = enforcer.enforce(raw_output, proposer_fn, context)
    """

    def __init__(
        self,
        max_retries: int = 2,
        schema_model: type[BaseModel] | None = None,
    ):
        self.max_retries = max_retries
        self.schema_model = schema_model

    def validate(self, raw_output: dict[str, Any]) -> ValidationResult:
        """Validate raw LLM output against the configured Pydantic schema.

        Args:
            raw_output: Parsed JSON dict from LLM response.

        Returns:
            ValidationResult with validity status and any errors.
        """
        if self.schema_model is None:
            return ValidationResult(is_valid=True, parsed=None, errors=[])

        try:
            parsed = self.schema_model(**raw_output)
            return ValidationResult(is_valid=True, parsed=parsed, errors=[])
        except ValidationError as exc:
            errors = []
            for error in exc.errors():
                loc = error.get("loc", [])
                field_path = ".".join(str(part) for part in loc) if loc else "root"
                errors.append(
                    {
                        "field": field_path,
                        "type": error.get("type", "unknown"),
                        "message": error.get("msg", ""),
                        "input": str(error.get("input", ""))[:100],
                    }
                )
            return ValidationResult(is_valid=False, parsed=None, errors=errors)

    def enforce(
        self,
        raw_output: dict[str, Any],
        proposer_fn: Callable[[str | None], dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate with retry loop. On failure, re-prompt with error feedback.

        Args:
            raw_output: Initial parsed LLM output to validate.
            proposer_fn: Callable that takes an error feedback string and returns
                         a new raw output dict from the LLM.
            context: Optional context for error messages.

        Returns:
            Validated raw output dict (possibly from a retry).
        """
        result = self.validate(raw_output)
        if result.is_valid:
            return raw_output

        for attempt in range(self.max_retries):
            feedback = self._build_feedback(result.errors, attempt + 1)
            logger.info(
                "Schema validation failed (attempt %d/%d), re-prompting: %s",
                attempt + 1,
                self.max_retries,
                result.error_summary(),
            )
            raw_output = proposer_fn(feedback)
            result = self.validate(raw_output)
            if result.is_valid:
                return raw_output

        logger.warning(
            "Schema enforcement exhausted retries (%d). Returning last output.",
            self.max_retries,
        )
        return raw_output

    def _build_feedback(self, errors: list[dict[str, Any]], attempt: int) -> str:
        """Build structured error feedback for re-prompting."""
        error_lines = []
        for err in errors:
            error_lines.append(
                f"  - Field '{err['field']}': {err['message']} (got: {err['input']})"
            )
        return (
            f"VALIDATION ERRORS (retry {attempt}):\n"
            + "\n".join(error_lines)
            + "\nPlease fix the errors above and return valid JSON."
        )
