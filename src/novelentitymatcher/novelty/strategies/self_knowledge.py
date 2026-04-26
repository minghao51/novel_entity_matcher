"""
Self-knowledge detection strategy wrapper.

Wraps SelfKnowledgeDetector to implement NoveltyStrategy protocol.
"""

from typing import Any, Dict, List, Set

import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import SelfKnowledgeConfig
from .self_knowledge_impl import SelfKnowledgeDetector


@StrategyRegistry.register
class SelfKnowledgeStrategy(NoveltyStrategy):
    """
    Self-knowledge strategy for novelty detection.

    Uses a sparse autoencoder to learn representations of known
    samples and flags high reconstruction error as novel.
    """

    strategy_id = "self_knowledge"
    maturity = "internal"

    def __init__(self):
        self._config: SelfKnowledgeConfig = None
        self._detector: SelfKnowledgeDetector = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: SelfKnowledgeConfig,
    ) -> None:
        self._config = config or SelfKnowledgeConfig()

        self._detector = SelfKnowledgeDetector(
            hidden_dim=self._config.hidden_dim,
            knowledge_threshold=self._config.threshold,
        )
        self._detector.fit(reference_embeddings, verbose=False)

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        flags: set[int] = set()
        metrics: dict[int, dict[str, Any]] = {}

        if self._detector is None or not self._detector._is_fitted:
            return flags, metrics

        novelty_scores = self._detector.compute_novelty_scores(embeddings)

        for idx, score in enumerate(novelty_scores):
            is_novel = score >= self._config.threshold

            if is_novel:
                flags.add(idx)

            metrics[idx] = {
                "self_knowledge_reconstruction_error": float(score),
                "self_knowledge_novelty_score": float(score),
                "self_knowledge_is_novel": bool(is_novel),
            }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        return SelfKnowledgeConfig

    def get_weight(self) -> float:
        return 0.15
