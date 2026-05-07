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
        self._calibrator: Any = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: ReActConfig,
    ) -> None:
        """Initialize ReAct wrapper and underlying energy strategy.

        When ``calibration_mode="conformal"``, trains a calibrator on
        trimmed-embedding energies.
        """
        self._config = config or ReActConfig()
        self._calibrator = None
        trimmed = trim_activations(reference_embeddings, self._config.trim_percentile)
        self._inner = EnergyOODStrategy()
        from ..config.strategies import EnergyConfig

        inner_config = EnergyConfig()
        self._inner.initialize(trimmed, reference_labels, inner_config)

        if self._config.calibration_mode == "conformal":
            self._initialize_conformal(trimmed, reference_labels)

        logger.info(
            "ReActEnergyStrategy initialized: trim_percentile=%.2f, inner=%s",
            self._config.trim_percentile,
            self._inner.strategy_id,
        )

    def _initialize_conformal(
        self,
        trimmed_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        """Train conformal calibrator on trimmed-embedding energies."""
        from .conformal import ConformalCalibrator

        n = len(trimmed_embeddings)
        frac = self._config.calibration_set_fraction
        n_cal = max(1, int(n * frac))
        if n_cal >= n:
            logger.warning(
                "ReAct conformal calibration disabled: split too large (n=%d, n_cal=%d)",
                n,
                n_cal,
            )
            return

        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        cal_indices = indices[:n_cal]

        cal_embeddings = trimmed_embeddings[cal_indices]
        cal_labels = [reference_labels[i] for i in cal_indices]

        logits = self._inner._compute_logits(cal_embeddings)
        energies = self._inner._compute_energy(logits)

        self._calibrator = ConformalCalibrator(
            alpha=self._config.calibration_alpha,
            method=self._config.calibration_method,
        )
        self._calibrator.calibrate(energies, np.array(cal_labels))

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        """Detect novel samples with ReAct trimming before energy scoring.

        When ``calibration_mode="conformal"``, uses own calibrator trained
        on trimmed-embedding energies.
        """
        trimmed = trim_activations(embeddings, self._config.trim_percentile)

        if (
            self._config.calibration_mode == "conformal"
            and self._calibrator is not None
            and self._calibrator.is_calibrated
        ):
            logits = self._inner._compute_logits(trimmed)
            energies = self._inner._compute_energy(logits)

            if self._config.calibration_method == "mondrian":
                p_values = self._calibrator.predict_pvalues_for_class(
                    energies, predicted_classes
                )
            else:
                p_values = self._calibrator.predict_pvalues(energies)

            flags = set()
            metrics = {}
            for idx in range(len(embeddings)):
                energy = float(energies[idx])
                metrics[idx] = {
                    "energy_score": energy,
                    "energy_threshold": self._inner._threshold,
                    "predicted_class": predicted_classes[idx],
                    "p_value": float(p_values[idx]),
                    "calibration_mode": "conformal",
                    "react_trim_percentile": self._config.trim_percentile,
                    "react_energy_is_novel": p_values[idx]
                    < self._config.calibration_alpha,
                }
                if p_values[idx] < self._config.calibration_alpha:
                    flags.add(idx)
            return flags, metrics

        flags, metrics = self._inner.detect(
            texts, trimmed, predicted_classes, confidences, **kwargs
        )
        for idx in metrics:
            metrics[idx]["react_trim_percentile"] = self._config.trim_percentile
            metrics[idx]["react_energy_is_novel"] = idx in flags
        return flags, metrics

    @property
    def config_schema(self) -> type:
        return ReActConfig

    def get_weight(self) -> float:
        return 0.30
