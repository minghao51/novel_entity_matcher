"""
Self-knowledge detection strategy wrapper.

Wraps SelfKnowledgeDetector to implement NoveltyStrategy protocol.
"""

from typing import Any

import numpy as np

from ..config.strategies import SelfKnowledgeConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy, SignalInfo
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
    score_keys = ("self_knowledge_novelty_score",)
    signal_info = SignalInfo(
        score_key="self_knowledge_novelty_score",
        flag_key="self_knowledge_is_novel",
        weight_name="self_knowledge",
        kind="flag",
    )
    default_weight = 0.15

    def __init__(self):
        self._config: SelfKnowledgeConfig | None = None
        self._detector: SelfKnowledgeDetector | None = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: SelfKnowledgeConfig,
    ) -> None:
        self._config = config or SelfKnowledgeConfig()

        self._detector = SelfKnowledgeDetector(
            hidden_dim=self._config.hidden_dim,
            knowledge_threshold=self._config.threshold,
        )
        self._detector.fit(reference_embeddings, verbose=False)

    def _detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        flags: set[int] = set()
        metrics: dict[int, dict[str, Any]] = {}

        if len(embeddings) == 0:
            return flags, metrics

        if self._detector is None or not self._detector.is_fitted:
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
