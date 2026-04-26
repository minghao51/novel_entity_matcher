"""
One-Class SVM novelty detection strategy wrapper.

Wraps OneClassSVMDetector to implement NoveltyStrategy protocol.
"""

from typing import Any

import numpy as np

from ..config.strategies import OneClassConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy
from .oneclass_impl import OneClassSVMDetector


@StrategyRegistry.register
class OneClassStrategy(NoveltyStrategy):
    strategy_id = "oneclass"
    maturity = "experimental"

    def __init__(self):
        self._config: OneClassConfig = None
        self._detector: OneClassSVMDetector = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: OneClassConfig,
    ) -> None:
        self._config = config or OneClassConfig()

        self._detector = OneClassSVMDetector(
            model_name=self._config.model_name,
            nu=self._config.nu,
            kernel=self._config.kernel,
            gamma=self._config.gamma,
        )
        self._detector.train(reference_labels, show_progress=False)

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        if not self._detector.is_trained:
            return set(), {}

        flags = set()
        metrics = {}

        results = self._detector.score_batch(texts)

        for idx, (is_novel, confidence) in enumerate(results):
            if is_novel:
                flags.add(idx)

            metrics[idx] = {
                "oneclass_is_novel": is_novel,
                "oneclass_confidence": confidence,
                "oneclass_novelty_score": confidence,
            }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        return OneClassConfig

    def get_weight(self) -> float:
        return 0.1
