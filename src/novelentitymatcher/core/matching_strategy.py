"""Matching strategy pattern for Matcher mode selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .embedding_matcher import EmbeddingMatcher
    from .entity_matcher import EntityMatcher
    from .bert_classifier import BERTClassifier
    from .hybrid_matcher import HybridMatcher
    from .matcher_shared import TextInput


@dataclass
class StrategyConfig:
    """Configuration for matching strategies.

    Encapsulates threshold, model settings, and training mode
    that were previously managed in _EntityMatcher.
    """

    threshold: float
    model_name: str
    training_mode: str
    normalize: bool = True


class MatchingStrategy(ABC):
    """Abstract base class for matching strategies."""

    def __init__(self, matcher: "MatcherFacade"):
        self._matcher = matcher

    @abstractmethod
    def match(
        self,
        texts: "TextInput",
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """Execute matching with this strategy."""
        pass

    @abstractmethod
    async def match_async(
        self,
        texts: "TextInput",
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """Execute async matching with this strategy."""
        pass

    @abstractmethod
    def build_index(self) -> None:
        """Build any required index for this strategy."""
        pass

    @abstractmethod
    def get_reference_corpus(self) -> dict:
        """Get reference corpus for this strategy."""
        pass


class ZeroShotStrategy(MatchingStrategy):
    """Strategy for zero-shot (embedding-only) matching."""

    def match(
        self,
        texts: "TextInput",
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        effective_threshold = self._resolve_threshold(threshold_override)
        return self._matcher.embedding_matcher.match(
            texts,
            top_k=top_k,
            threshold_override=effective_threshold,
            **kwargs,
        )

    async def match_async(
        self,
        texts: "TextInput",
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        effective_threshold = self._resolve_threshold(threshold_override)
        return await self._matcher.embedding_matcher.match_async(
            texts,
            top_k=top_k,
            threshold_override=effective_threshold,
            **kwargs,
        )

    def build_index(self) -> None:
        self._matcher.embedding_matcher.build_index()

    def get_reference_corpus(self) -> dict:
        return self._matcher.embedding_matcher.get_reference_corpus()

    def _resolve_threshold(self, override: Optional[float]) -> float:
        return self._matcher._resolve_threshold(override, self._matcher.threshold)


class HeadOnlyFullStrategy(MatchingStrategy):
    """Strategy for head-only and full training modes."""

    def match(
        self,
        texts: "TextInput",
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        effective_threshold = self._resolve_threshold(threshold_override)
        return self._matcher.entity_matcher.match(
            texts,
            candidates=kwargs.get("candidates"),
            top_k=top_k,
            threshold_override=effective_threshold,
        )

    async def match_async(
        self,
        texts: "TextInput",
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        effective_threshold = self._resolve_threshold(threshold_override)
        return await self._matcher.entity_matcher.match_async(
            texts,
            candidates=kwargs.get("candidates"),
            top_k=top_k,
            threshold_override=effective_threshold,
        )

    def build_index(self) -> None:
        pass

    def get_reference_corpus(self) -> dict:
        return self._matcher.entity_matcher.get_reference_corpus()

    def _resolve_threshold(self, override: Optional[float]) -> float:
        return self._matcher._resolve_threshold(override, self._matcher.threshold)


class BertStrategy(MatchingStrategy):
    """Strategy for BERT-based matching."""

    def match(
        self,
        texts: "TextInput",
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        effective_threshold = self._resolve_threshold(threshold_override)
        return self._matcher.bert_matcher.match(
            texts,
            candidates=kwargs.get("candidates"),
            top_k=top_k,
            threshold_override=effective_threshold,
        )

    async def match_async(
        self,
        texts: "TextInput",
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        effective_threshold = self._resolve_threshold(threshold_override)
        return await self._matcher.bert_matcher.match_async(
            texts,
            candidates=kwargs.get("candidates"),
            top_k=top_k,
            threshold_override=effective_threshold,
        )

    def build_index(self) -> None:
        encoder = self._matcher.embedding_matcher
        if encoder.model is None:
            encoder.build_index()

    def get_reference_corpus(self) -> dict:
        encoder_matcher = self._matcher.embedding_matcher
        if encoder_matcher.model is None:
            encoder_matcher.build_index()
        return self._matcher.bert_matcher.get_reference_corpus(
            encoder=encoder_matcher.model
        )

    def _resolve_threshold(self, override: Optional[float]) -> float:
        return self._matcher._resolve_threshold(override, self._matcher.threshold)


class HybridStrategy(MatchingStrategy):
    """Strategy for hybrid blocking + retrieval matching."""

    def match(
        self,
        texts: "TextInput",
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        effective_threshold = self._resolve_threshold(threshold_override)
        return self._matcher._match_hybrid(
            texts,
            top_k=top_k,
            threshold_override=effective_threshold,
            **kwargs,
        )

    async def match_async(
        self,
        texts: "TextInput",
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        effective_threshold = self._resolve_threshold(threshold_override)
        return await self._matcher._match_hybrid_async(
            texts,
            top_k=top_k,
            threshold_override=effective_threshold,
            **kwargs,
        )

    def build_index(self) -> None:
        pass

    def get_reference_corpus(self) -> dict:
        return self._matcher.embedding_matcher.get_reference_corpus()

    def _resolve_threshold(self, override: Optional[float]) -> float:
        return self._matcher._resolve_threshold(override, self._matcher.threshold)


_STRATEGY_MAP = {
    "zero-shot": ZeroShotStrategy,
    "head-only": HeadOnlyFullStrategy,
    "full": HeadOnlyFullStrategy,
    "bert": BertStrategy,
    "hybrid": HybridStrategy,
    "auto": ZeroShotStrategy,
}


def get_strategy(mode: str) -> type[MatchingStrategy]:
    """Get strategy class for the given mode."""
    if mode not in _STRATEGY_MAP:
        from ..exceptions import ModeError

        raise ModeError(f"Unknown mode: {mode}", invalid_mode=mode)
    return _STRATEGY_MAP[mode]


class MatcherFacade:
    """Facade providing access to all matcher components for strategies."""

    def __init__(
        self,
        embedding_matcher: "EmbeddingMatcher",
        entity_matcher: "EntityMatcher",
        bert_matcher: "BERTClassifier",
        hybrid_matcher: "HybridMatcher",
        config: StrategyConfig,
    ):
        self.embedding_matcher = embedding_matcher
        self.entity_matcher = entity_matcher
        self.bert_matcher = bert_matcher
        self.hybrid_matcher = hybrid_matcher
        self.threshold = config.threshold
        self.model_name = config.model_name
        self._training_mode = config.training_mode
        self._config = config

    @staticmethod
    def _resolve_threshold(
        threshold_override: Optional[float], default: float
    ) -> float:
        """Resolve threshold from override or default.

        Moved from _EntityMatcher to MatcherFacade for better encapsulation.
        """
        from .matcher_shared import resolve_threshold

        return resolve_threshold(threshold_override, default)

    def get_strategy(self, mode: Optional[str] = None) -> MatchingStrategy:
        """Get strategy instance for the given or current mode."""
        effective_mode = mode or self._training_mode
        strategy_cls = get_strategy(effective_mode)
        return strategy_cls(self)
