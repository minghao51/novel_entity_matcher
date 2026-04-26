"""
Pattern-based novelty detection strategy wrapper.

Wraps PatternScorer to implement NoveltyStrategy protocol.
"""

from typing import Any

import numpy as np

from ..config.strategies import PatternConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy
from .pattern_impl import PatternScorer


@StrategyRegistry.register
class PatternStrategy(NoveltyStrategy):
    strategy_id = "pattern"
    maturity = "internal"

    def __init__(self):
        self._config: PatternConfig = None
        self._pattern_scorer: PatternScorer = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: PatternConfig,
    ) -> None:
        self._config = config or PatternConfig()
        self._pattern_scorer = PatternScorer(known_entities=reference_labels)

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        flags = set()
        metrics = {}

        for idx, text in enumerate(texts):
            novelty_score = self._pattern_scorer.score_novelty(text)
            is_novel = novelty_score >= self._config.threshold

            if is_novel:
                flags.add(idx)

            metrics[idx] = {
                "pattern_novelty_score": novelty_score,
                "pattern_is_novel": is_novel,
                "pattern_text": text,
            }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        return PatternConfig

    def get_weight(self) -> float:
        return 0.2
