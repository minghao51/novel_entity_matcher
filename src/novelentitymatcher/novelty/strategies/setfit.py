"""
SetFit contrastive novelty detection strategy wrapper.

Wraps SetFitDetector to implement NoveltyStrategy protocol.
"""

from typing import Any

import numpy as np

from ..config.strategies import SetFitConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy
from .setfit_impl import SetFitDetector


@StrategyRegistry.register
class SetFitStrategy(NoveltyStrategy):
    strategy_id = "setfit"
    maturity = "internal"

    def __init__(self):
        self._config: SetFitConfig = None
        self._detector: SetFitDetector = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: SetFitConfig,
    ) -> None:
        self._config = config or SetFitConfig()

        self._detector = SetFitDetector(
            known_entities=reference_labels,
            model_name=self._config.model_name,
            margin=self._config.margin,
            num_epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            learning_rate=self._config.learning_rate,
        )
        self._detector.train(show_progress=False)

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        flags: set[int] = set()
        metrics: dict[int, dict[str, Any]] = {}

        if self._detector is None or not self._detector.is_trained:
            return flags, metrics

        results = self._detector.score_batch(texts)

        for idx, (is_novel, confidence) in enumerate(results):
            if is_novel:
                flags.add(idx)

            metrics[idx] = {
                "setfit_is_novel": is_novel,
                "setfit_confidence": confidence,
                "setfit_novelty_score": 1.0 - confidence,
            }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        return SetFitConfig

    def get_weight(self) -> float:
        return 0.1
