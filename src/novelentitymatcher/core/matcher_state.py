from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import (
    resolve_bert_model_alias,
    resolve_model_alias,
    resolve_training_model_alias,
)
from ..exceptions import ModeError
from ..utils.validation import validate_threshold

if TYPE_CHECKING:
    from .matcher import Matcher

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
    detected_mode: str | None = None
    has_training_data: bool = False

    @classmethod
    def create(
        cls,
        *,
        model: str,
        threshold: float,
        mode: str | None,
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
    def _coerce_training_mode(mode: str | None) -> str:
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


class MatcherComponentFactory:
    """Lazy matcher-component construction behind the public Matcher facade."""

    def __init__(self, owner: "Matcher") -> None:
        self._owner = owner
        self._embedding_matcher: Any = None
        self._entity_matcher: Any = None
        self._bert_matcher: Any = None
        self._hybrid_matcher: Any = None

    def iter_threshold_targets(self) -> tuple[Any, ...]:
        return (
            self._embedding_matcher,
            self._entity_matcher,
            self._bert_matcher,
        )

    def get_embedding_matcher(self) -> Any:
        if self._embedding_matcher is None:
            from .embedding_matcher import EmbeddingMatcher

            self._embedding_matcher = EmbeddingMatcher(
                entities=self._owner.entities,
                model_name=self._owner.model_name,
                threshold=self._owner.threshold,
                normalize=self._owner.normalize,
            )
        return self._embedding_matcher

    def get_entity_matcher(self) -> Any:
        if self._entity_matcher is None:
            from .matcher_entity import _EntityMatcher

            self._entity_matcher = _EntityMatcher(
                entities=self._owner.entities,
                model_name=self._owner._training_model_name,
                threshold=self._owner.threshold,
                normalize=self._owner.normalize,
                classifier_type="setfit",
            )
        return self._entity_matcher

    def get_bert_matcher(self) -> Any:
        if self._bert_matcher is None:
            from .matcher_entity import _EntityMatcher

            self._bert_matcher = _EntityMatcher(
                entities=self._owner.entities,
                model_name=self._owner._bert_model_name,
                threshold=self._owner.threshold,
                normalize=self._owner.normalize,
                classifier_type="bert",
            )
        return self._bert_matcher

    def get_hybrid_matcher(self) -> Any:
        if self._hybrid_matcher is None:
            from .hybrid import HybridMatcher

            self._hybrid_matcher = HybridMatcher(
                entities=self._owner.entities,
                blocking_strategy=self._owner.blocking_strategy,
                retriever_model=self._owner.model_name,
                reranker_model=self._owner.reranker_model,
                normalize=self._owner.normalize,
            )
        return self._hybrid_matcher
