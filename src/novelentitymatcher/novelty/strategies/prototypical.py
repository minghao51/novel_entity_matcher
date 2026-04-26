"""
Prototypical network novelty detection strategy wrapper.

Wraps PrototypicalDetector to implement NoveltyStrategy protocol.
"""

from typing import Dict, List, Set, Any
import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import PrototypicalConfig
from .prototypical_impl import PrototypicalDetector


@StrategyRegistry.register
class PrototypicalStrategy(NoveltyStrategy):
    strategy_id = "prototypical"
    maturity = "experimental"

    def __init__(self):
        self._config: PrototypicalConfig = None
        self._detector: PrototypicalDetector = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: PrototypicalConfig,
    ) -> None:
        self._config = config or PrototypicalConfig()

        self._detector = PrototypicalDetector(
            distance_threshold=self._config.distance_threshold,
            model_name=self._config.model_name,
        )

        training_data = [{"text": label, "label": label} for label in reference_labels]
        self._detector.train(training_data, show_progress=False)

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

        if self._detector is None or not self._detector.is_trained:
            return flags, metrics

        results = self._detector.score_batch(texts)

        for idx, (is_novel, distance, nearest_label) in enumerate(results):
            if is_novel:
                flags.add(idx)

            metrics[idx] = {
                "prototypical_is_novel": is_novel,
                "prototypical_distance": distance,
                "prototypical_nearest_label": nearest_label,
                "prototypical_novelty_score": min(
                    distance / self._config.distance_threshold, 1.0
                )
                if distance
                else 0.0,
            }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        return PrototypicalConfig

    def get_weight(self) -> float:
        return 0.1
