from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .embedding_matcher import EmbeddingMatcher
from .matcher_entity import _EntityMatcher

if TYPE_CHECKING:
    from .matcher import Matcher


class MatcherComponentFactory:
    """Lazy matcher-component construction behind the public Matcher facade."""

    def __init__(self, owner: Matcher):
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
            self._embedding_matcher = EmbeddingMatcher(
                entities=self._owner.entities,
                model_name=self._owner.model_name,
                threshold=self._owner.threshold,
                normalize=self._owner.normalize,
            )
        return self._embedding_matcher

    def get_entity_matcher(self) -> Any:
        if self._entity_matcher is None:
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
