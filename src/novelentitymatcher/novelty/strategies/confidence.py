"""
Confidence threshold-based novelty detection strategy.

Flags samples with prediction confidence below a threshold as novel.
"""

from typing import Dict, List, Set, Any
import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import ConfidenceConfig


@StrategyRegistry.register
class ConfidenceStrategy(NoveltyStrategy):
    """
    Confidence threshold strategy for novelty detection.

    Flags samples as novel if their prediction confidence falls
    below a configured threshold.
    """

    strategy_id = "confidence"
    maturity = "production"

    def __init__(self):
        self._config: ConfidenceConfig = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: ConfidenceConfig,
    ) -> None:
        """
        Initialize the confidence strategy.

        Args:
            reference_embeddings: Embeddings of known samples (not used)
            reference_labels: Labels of known samples (not used)
            config: ConfidenceConfig with threshold parameter
        """
        self._config = config or ConfidenceConfig()

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using confidence threshold.

        Args:
            texts: Input texts
            embeddings: Text embeddings (not used)
            predicted_classes: Predicted classes (not used)
            confidences: Prediction confidence scores
            **kwargs: Additional parameters

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
        flags = set()
        metrics = {}

        for idx, confidence in enumerate(confidences):
            is_novel = confidence < self._config.threshold

            if is_novel:
                flags.add(idx)

            metrics[idx] = {
                "confidence_score": float(confidence),
                "confidence_is_novel": is_novel,
            }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        """Return ConfidenceConfig as the config schema."""
        return ConfidenceConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        # Confidence is a foundational signal, give it moderate weight
        return 0.35
