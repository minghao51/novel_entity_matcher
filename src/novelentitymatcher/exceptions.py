"""Custom exceptions for novel_entity_matcher with helpful context and suggestions."""

from typing import Any, Dict, Optional
import json


class SemanticMatcherError(Exception):
    """Base exception for all novel_entity_matcher errors."""

    pass


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
        entity: Optional[Dict[str, Any]] = None,
        field: Optional[str] = None,
        suggestion: Optional[str] = None,
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
        training_mode: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
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

    pass


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
        invalid_mode: Optional[str] = None,
        valid_modes: Optional[list[str]] = None,
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
        last_error: Optional[Exception] = None,
        attempted_models: Optional[list[str]] = None,
    ):
        self.raw_message = message
        self.last_error = last_error
        self.attempted_models = attempted_models or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context."""
        msg = self.raw_message

        if self.attempted_models:
            msg += f"\n  Attempted models: {', '.join(self.attempted_models)}"

        if self.last_error:
            msg += f"\n  Last error: {self.last_error}"

        return msg
