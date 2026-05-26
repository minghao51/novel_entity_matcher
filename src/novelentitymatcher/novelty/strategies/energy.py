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
from .base import NoveltyStrategy, SignalInfo
from .conformal_mixin import ConformalMixin

logger = get_logger(__name__)


@StrategyRegistry.register
class EnergyOODStrategy(ConformalMixin, NoveltyStrategy):
    strategy_id = "energy_ood"
    maturity = "experimental"
    score_keys = ("energy_score",)
    signal_info = SignalInfo(
        score_key="energy_score",
        flag_key="energy_is_novel",
        weight_name="energy_ood",
        kind="special",
    )
    default_weight = 0.3

    def __init__(self):
        self._config: EnergyConfig | None = None
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
        self._run_conformal_calibration(
            reference_embeddings,
            reference_labels,
            lambda embs, _labels: self._compute_energy(self._compute_logits(embs)),
        )

    def _initialize_core(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        """Core initialization: compute centroids and threshold."""
        self._centroids = self.compute_class_means(
            reference_embeddings, reference_labels
        )

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

    def _detect(
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

        if self._is_conformal_active():
            return self._conformal_detect_loop(
                energies,
                predicted_classes,
                lambda idx, energy, pv: {
                    "energy_score": energy,
                    "energy_threshold": self._threshold,
                    "predicted_class": predicted_classes[idx],
                    "energy_is_novel": pv < self._config.calibration_alpha,
                },
            )

        for idx in range(len(embeddings)):
            energy = float(energies[idx])
            is_novel = energy > self._threshold
            metrics[idx] = {
                "energy_score": energy,
                "energy_threshold": self._threshold,
                "predicted_class": predicted_classes[idx],
                "energy_is_novel": is_novel,
            }
            if is_novel:
                flags.add(idx)

        return flags, metrics

    @property
    def config_schema(self) -> type:
        return EnergyConfig
