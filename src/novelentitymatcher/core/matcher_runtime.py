from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ..config import (
    resolve_bert_model_alias,
    resolve_model_alias,
    resolve_training_model_alias,
)
from ..exceptions import ModeError
from ..utils.validation import validate_threshold

_VALID_TRAINING_MODES = {"auto", "zero-shot", "head-only", "full", "hybrid", "bert"}


@dataclass
class MatcherRuntimeState:
    """Centralized matcher configuration and mutable runtime state."""

    requested_model: str
    model_name: str
    training_model_name: str
    bert_model_name: str
    threshold: float
    training_mode: str
    detected_mode: Optional[str] = None
    has_training_data: bool = False

    @classmethod
    def create(
        cls,
        *,
        model: str,
        threshold: float,
        mode: Optional[str],
    ) -> "MatcherRuntimeState":
        validated_threshold = validate_threshold(threshold)
        training_mode = cls._coerce_training_mode(mode)
        return cls(
            requested_model=model,
            model_name=resolve_model_alias(model),
            training_model_name=resolve_training_model_alias(model),
            bert_model_name=resolve_bert_model_alias(model),
            threshold=validated_threshold,
            training_mode=training_mode,
        )

    @staticmethod
    def _coerce_training_mode(mode: Optional[str]) -> str:
        if mode is None or mode == "auto":
            return "auto"
        if mode not in _VALID_TRAINING_MODES:
            raise ModeError(f"Invalid mode: {mode}", invalid_mode=mode)
        return mode

    def update_training_mode(self, mode: str) -> str:
        self.training_mode = self._coerce_training_mode(mode)
        return self.training_mode

    def set_detected_mode(self, mode: str) -> str:
        self.detected_mode = mode
        return mode

    def apply_threshold(self, threshold: float, matchers: Iterable[Any]) -> float:
        self.threshold = validate_threshold(threshold)
        for matcher in matchers:
            if matcher is not None:
                matcher.threshold = self.threshold
        return self.threshold
