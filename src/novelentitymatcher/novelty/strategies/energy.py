"""Energy-based OOD detection strategy.

Flags samples based on energy scores computed from class-conditional centroid
logits. Lower energy = more in-distribution. Provably better aligned with
input density than raw distance heuristics (Liu et al., NeurIPS 2020).
"""

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ...utils.logging_config import get_logger
from ..config.strategies import EnergyConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy

logger = get_logger(__name__)


@StrategyRegistry.register
class EnergyOODStrategy(NoveltyStrategy):
    """Energy score strategy for novelty detection.

    Computes energy as ``E(x) = -T * log(sum_i exp(logit_i(x) / T))``
    where logits are derived from cosine similarity to class centroids.
    Samples with energy below the learned threshold are flagged as novel.
    """

    strategy_id = "energy_ood"
    maturity = "experimental"

    def __init__(self):
        self._config: EnergyConfig = None
        self._centroids: dict[str, np.ndarray] = {}
        self._centroid_matrix: np.ndarray | None = None
        self._centroid_labels: list[str] = []
        self._threshold: float = 0.0
        self._temperature: float = 1.0
        self._scale: float = 25.0
        self._calibrator: Any = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: EnergyConfig,
    ) -> None:
        """Initialize energy strategy with reference data.

        Computes per-class centroids, derives reference energies, and sets
        the threshold as ``mean(energies) - 2 * std(energies)``.

        When ``calibration_mode="conformal"``, splits reference data into
        core and calibration sets, then wraps raw energies with p-values.
        """
        self._config = config or EnergyConfig()
        self._temperature = self._config.temperature
        self._scale = self._config.scale
        self._centroids = {}
        self._calibrator = None

        if self._config.calibration_mode == "conformal":
            self._initialize_with_calibration(reference_embeddings, reference_labels)
        else:
            self._initialize_core(reference_embeddings, reference_labels)

    def _initialize_with_calibration(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        """Initialize with conformal calibration, splitting reference data."""
        from .conformal import ConformalCalibrator

        n = len(reference_embeddings)
        frac = self._config.calibration_set_fraction
        n_cal = max(1, int(n * frac))
        if n_cal >= n:
            logger.warning(
                "Energy conformal calibration disabled: calibration split "
                "would leave no core reference samples (n=%d, n_cal=%d)",
                n,
                n_cal,
            )
            self._initialize_core(reference_embeddings, reference_labels)
            return

        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        cal_indices = indices[:n_cal]
        core_indices = indices[n_cal:]

        core_embeddings = reference_embeddings[core_indices]
        core_labels = [reference_labels[i] for i in core_indices]

        self._initialize_core(core_embeddings, core_labels)

        cal_embeddings = reference_embeddings[cal_indices]
        cal_labels = [reference_labels[i] for i in cal_indices]
        cal_logits = self._compute_logits(cal_embeddings)
        cal_energies = self._compute_energy(cal_logits)

        self._calibrator = ConformalCalibrator(
            alpha=self._config.calibration_alpha,
            method=self._config.calibration_method,
        )
        self._calibrator.calibrate(cal_energies, np.array(cal_labels))
        logger.info(
            "Energy strategy initialized with conformal calibration: "
            "n_core=%d, n_cal=%d, method=%s",
            len(core_embeddings),
            n_cal,
            self._config.calibration_method,
        )

    def _initialize_core(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        """Core initialization: compute centroids and threshold."""
        for label in set(reference_labels):
            mask = np.array(reference_labels) == label
            self._centroids[label] = reference_embeddings[mask].mean(axis=0)

        self._centroid_labels = list(self._centroids.keys())
        self._centroid_matrix = np.array(
            [self._centroids[label] for label in self._centroid_labels]
        )

        ref_logits = self._compute_logits(reference_embeddings)
        ref_energies = self._compute_energy(ref_logits)
        self._threshold = float(
            np.mean(ref_energies)
            + self._config.threshold_std_multiplier * np.std(ref_energies)
        )

        logger.info(
            "EnergyOODStrategy initialized: n_classes=%d, T=%.2f, scale=%.2f, threshold=%.4f",
            len(self._centroids),
            self._temperature,
            self._scale,
            self._threshold,
        )

    def _compute_logits(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute logits as scaled cosine similarity to centroids."""
        sims = cosine_similarity(embeddings, self._centroid_matrix)
        return sims * self._scale

    def _compute_energy(self, logits: np.ndarray) -> np.ndarray:
        """Compute energy scores from logits."""
        scaled = logits / self._temperature
        max_per_row = np.max(scaled, axis=1, keepdims=True)
        stable_sum = np.sum(np.exp(scaled - max_per_row), axis=1)
        logsumexp = np.log(stable_sum) + max_per_row.squeeze(axis=1)
        return -self._temperature * logsumexp

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        """Detect novel samples using energy scoring.

        When ``calibration_mode="conformal"``, flagging uses p-values
        instead of raw energy thresholds.

        Returns:
            (flags, metrics) where flags contains indices with energy > threshold
            (or p_value < alpha in conformal mode).
        """
        flags = set()
        metrics = {}

        if len(embeddings) == 0:
            return flags, metrics

        logits = self._compute_logits(embeddings)
        energies = self._compute_energy(logits)

        if (
            self._config.calibration_mode == "conformal"
            and self._calibrator is not None
            and self._calibrator.is_calibrated
        ):
            if self._config.calibration_method == "mondrian":
                p_values = self._calibrator.predict_pvalues_for_class(
                    energies, predicted_classes
                )
            else:
                p_values = self._calibrator.predict_pvalues(energies)

            for idx in range(len(embeddings)):
                energy = float(energies[idx])
                metrics[idx] = {
                    "energy_score": energy,
                    "energy_threshold": self._threshold,
                    "predicted_class": predicted_classes[idx],
                    "p_value": float(p_values[idx]),
                    "calibration_mode": "conformal",
                    "energy_is_novel": p_values[idx] < self._config.calibration_alpha,
                }
                if p_values[idx] < self._config.calibration_alpha:
                    flags.add(idx)
        else:
            for idx in range(len(embeddings)):
                energy = float(energies[idx])
                metrics[idx] = {
                    "energy_score": energy,
                    "energy_threshold": self._threshold,
                    "predicted_class": predicted_classes[idx],
                    "energy_is_novel": energy > self._threshold,
                }
                if energy > self._threshold:
                    flags.add(idx)

        return flags, metrics

    @property
    def config_schema(self) -> type:
        return EnergyConfig

    def get_weight(self) -> float:
        return 0.30
