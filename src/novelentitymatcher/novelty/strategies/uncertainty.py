"""
Uncertainty-based novelty detection strategy.

Flags samples based on prediction uncertainty using margin and entropy.
"""

from typing import Dict, List, Set, Any, Optional
import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import UncertaintyConfig


@StrategyRegistry.register
class UncertaintyStrategy(NoveltyStrategy):
    """
    Uncertainty-based strategy for novelty detection.

    Flags samples as novel if their prediction uncertainty
    exceeds configured thresholds (margin or entropy).
    """

    strategy_id = "uncertainty"
    maturity = "internal"

    def __init__(self):
        self._config: UncertaintyConfig = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: UncertaintyConfig,
    ) -> None:
        """
        Initialize the uncertainty strategy.

        Args:
            reference_embeddings: Embeddings of known samples (not used)
            reference_labels: Labels of known samples (not used)
            config: UncertaintyConfig with thresholds
        """
        self._config = config or UncertaintyConfig()

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using uncertainty metrics.

        Args:
            texts: Input texts
            embeddings: Text embeddings (not used)
            predicted_classes: Predicted classes (not used)
            confidences: Prediction confidence scores
            **kwargs: Additional parameters, may include 'all_probs' for full distribution

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
        flags = set()
        metrics = {}

        # Check if we have full probability distributions
        all_probs = kwargs.get("all_probs", None)

        for idx, confidence in enumerate(confidences):
            metric = self._compute_uncertainty_metrics(
                idx,
                confidence,
                all_probs[idx] if all_probs is not None else None,
            )
            metrics[idx] = metric

            # Check if uncertainty exceeds thresholds
            is_novel = (
                metric["margin_score"] < self._config.margin_threshold
                or metric["entropy_score"] > self._config.entropy_threshold
            )

            if is_novel:
                flags.add(idx)

        return flags, metrics

    def _compute_uncertainty_metrics(
        self,
        idx: int,
        confidence: float,
        probs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute uncertainty metrics for a single sample.

        Args:
            idx: Sample index
            confidence: Prediction confidence
            probs: Full probability distribution (optional)

        Returns:
            Dictionary with uncertainty metrics
        """
        # Margin score: difference between top-1 and top-2 predictions
        # If we only have confidence (top-1), estimate margin
        if probs is not None and len(probs) >= 2:
            sorted_probs = np.sort(probs)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1])
        else:
            # Estimate margin from confidence
            margin = confidence * 0.2  # Placeholder

        # Entropy score
        if probs is not None:
            eps = 1e-10
            p = np.clip(probs, eps, 1.0 - eps)
            entropy = float(-np.sum(p * np.log(p)))
        else:
            # Binary entropy approximation from confidence
            eps = 1e-10
            p = np.clip(confidence, eps, 1.0 - eps)
            entropy = float(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))

        # Overall uncertainty score (normalized)
        uncertainty_score = 1.0 - margin  # Lower margin = higher uncertainty

        return {
            "uncertainty_score": uncertainty_score,
            "margin_score": margin,
            "entropy_score": entropy,
            "confidence": float(confidence),
        }

    @property
    def config_schema(self) -> type:
        """Return UncertaintyConfig as the config schema."""
        return UncertaintyConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        # Uncertainty is a strong signal
        return 0.35
