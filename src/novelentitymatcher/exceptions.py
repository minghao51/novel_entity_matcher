"""Custom exceptions for novel_entity_matcher with helpful context and suggestions."""

import json
import re
from typing import Any


class SemanticMatcherError(Exception):
    """Base exception for all novel_entity_matcher errors."""


class ValidationError(ValueError, SemanticMatcherError):
    """Raised when input validation fails with helpful context.

    Attributes:
        entity: The entity that failed validation (if applicable)
        field: The specific field that failed validation
        suggestion: Helpful suggestion for fixing the error
    """

    def __init__(
        self,
        message: str,
        *,
        entity: dict[str, Any] | None = None,
        field: str | None = None,
        suggestion: str | None = None,
    ):
        self.raw_message = message
        self.entity = entity
        self.field = field
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context."""
        msg = self.raw_message

        if self.field:
            msg += f"\n  Problem field: {self.field}"

        if self.entity:
            # Truncate entity representation to avoid huge error messages
            entity_str = json.dumps(self.entity, indent=2)
            if len(entity_str) > 200:
                entity_str = entity_str[:200] + "..."
            msg += f"\n  Entity:\n{entity_str}"

        if self.suggestion:
            msg += f"\n  💡 Suggestion: {self.suggestion}"

        return msg


class TrainingError(RuntimeError, SemanticMatcherError):
    """Raised when training fails with diagnostic information.

    Attributes:
        training_mode: The mode that was being trained
        details: Additional diagnostic information
    """

    def __init__(
        self,
        message: str,
        *,
        training_mode: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.raw_message = message
        self.training_mode = training_mode
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context."""
        msg = self.raw_message

        if self.training_mode:
            msg += f"\n  Training mode: {self.training_mode}"

        if self.details:
            msg += "\n  Details:"
            for key, value in self.details.items():
                msg += f"\n    {key}: {value}"

        return msg


class MatchingError(RuntimeError, SemanticMatcherError):
    """Raised when matching operations fail."""


class ModeError(ValueError, SemanticMatcherError):
    """Raised when matcher mode configuration is invalid.

    Attributes:
        invalid_mode: The mode that was provided
        valid_modes: List of valid mode options
    """

    def __init__(
        self,
        message: str,
        *,
        invalid_mode: str | None = None,
        valid_modes: list[str] | None = None,
    ):
        self.raw_message = message
        self.invalid_mode = invalid_mode
        self.valid_modes = valid_modes or [
            "zero-shot",
            "head-only",
            "full",
            "hybrid",
            "auto",
        ]
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context."""
        msg = self.raw_message

        if self.invalid_mode:
            msg += f"\n  Invalid mode: '{self.invalid_mode}'"
            msg += f"\n  Valid modes: {', '.join(self.valid_modes)}"

        return msg


_API_KEY_PATTERN = re.compile(
    r"(sk-or-v1-[A-Za-z0-9]+"
    r"|sk-ant-[A-Za-z0-9]+"
    r"|sk-[A-Za-z0-9]{20,}"
    r"|hf_[A-Za-z0-9]+"
    r"|AIza[A-Za-z0-9_\\-]{35}"
    r"|ya29\.[A-Za-z0-9_\\-]+"
    r")"
)


def _redact_api_keys(text: str) -> str:
    return _API_KEY_PATTERN.sub("...REDACTED...", text)


class LLMError(SemanticMatcherError):
    """Raised when LLM operations fail after all retries.

    Attributes:
        last_error: The last exception that caused all models to fail
        attempted_models: List of models that were attempted
    """

    def __init__(
        self,
        message: str,
        *,
        last_error: Exception | None = None,
        attempted_models: list[str] | None = None,
    ):
        self.raw_message = message
        self.last_error = last_error
        self.attempted_models = attempted_models or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context."""
        msg = _redact_api_keys(self.raw_message)

        if self.attempted_models:
            msg += f"\n  Attempted models: {', '.join(self.attempted_models)}"

        if self.last_error:
            last_error_str = _redact_api_keys(str(self.last_error))
            msg += f"\n  Last error: {last_error_str}"

        return msg
