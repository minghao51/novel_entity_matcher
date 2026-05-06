"""ReAct-style feature trimming utility for OOD strategies.

ReAct (Sun & Li, 2021) trims extreme activations before scoring to improve
OOD detection. This module provides a reusable wrapper that can be applied
to any strategy that operates on embeddings.
"""

from typing import Any

import numpy as np

from ...utils.logging_config import get_logger
from ..config.strategies import ReActConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy
from .energy import EnergyOODStrategy

logger = get_logger(__name__)


def trim_activations(embeddings: np.ndarray, percentile: float) -> np.ndarray:
    """Trim activations above a percentile threshold.

    Args:
        embeddings: Input embeddings (n_samples, dim).
        percentile: Threshold percentile (0-1). Values above this percentile
            are clamped to the threshold value.

    Returns:
        Trimmed embeddings with the same shape as input.
    """
    threshold = np.percentile(embeddings, percentile * 100)
    trimmed = embeddings.copy()
    trimmed[trimmed > threshold] = threshold
    return trimmed


@StrategyRegistry.register
class ReActEnergyStrategy(NoveltyStrategy):
    """ReAct wrapper around EnergyOODStrategy.

    Trims top-percentile activations from embeddings before passing them
    to an inner energy strategy for scoring.
    """

    strategy_id = "react_energy"
    maturity = "experimental"

    def __init__(self):
        self._config: ReActConfig = None
        self._inner: EnergyOODStrategy | None = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: ReActConfig,
    ) -> None:
        """Initialize ReAct wrapper and underlying energy strategy."""
        self._config = config or ReActConfig()
        # Initialize inner strategy with trimmed reference embeddings
        trimmed = trim_activations(reference_embeddings, self._config.trim_percentile)
        self._inner = EnergyOODStrategy()
        from ..config.strategies import EnergyConfig

        inner_config = EnergyConfig()
        self._inner.initialize(trimmed, reference_labels, inner_config)
        logger.info(
            "ReActEnergyStrategy initialized: trim_percentile=%.2f, inner=%s",
            self._config.trim_percentile,
            self._inner.strategy_id,
        )

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        """Detect novel samples with ReAct trimming before energy scoring."""
        trimmed = trim_activations(embeddings, self._config.trim_percentile)
        flags, metrics = self._inner.detect(
            texts, trimmed, predicted_classes, confidences, **kwargs
        )
        # Annotate metrics with ReAct info
        for idx in metrics:
            metrics[idx]["react_trim_percentile"] = self._config.trim_percentile
            metrics[idx]["react_energy_is_novel"] = idx in flags
        return flags, metrics

    @property
    def config_schema(self) -> type:
        return ReActConfig

    def get_weight(self) -> float:
        return 0.30
