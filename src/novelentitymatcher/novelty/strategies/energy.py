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

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: EnergyConfig,
    ) -> None:
        """Initialize energy strategy with reference data.

        Computes per-class centroids, derives reference energies, and sets
        the threshold as ``mean(energies) - 2 * std(energies)``.
        """
        self._config = config or EnergyConfig()
        self._temperature = self._config.temperature
        self._scale = self._config.scale
        self._centroids = {}

        for label in set(reference_labels):
            mask = np.array(reference_labels) == label
            self._centroids[label] = reference_embeddings[mask].mean(axis=0)

        self._centroid_labels = list(self._centroids.keys())
        self._centroid_matrix = np.array(
            [self._centroids[label] for label in self._centroid_labels]
        )

        ref_logits = self._compute_logits(reference_embeddings)
        ref_energies = self._compute_energy(ref_logits)
        # Higher energy (closer to 0) = OOD; lower energy (more negative) = in-dist
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
        return -self._temperature * np.log(
            np.sum(np.exp(logits / self._temperature), axis=1)
        )

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        """Detect novel samples using energy scoring.

        Returns:
            (flags, metrics) where flags contains indices with energy > threshold.
            Higher energy (closer to 0) indicates OOD; lower energy (more negative)
            indicates in-distribution.
        """
        flags = set()
        metrics = {}

        if len(embeddings) == 0:
            return flags, metrics

        logits = self._compute_logits(embeddings)
        energies = self._compute_energy(logits)

        for idx in range(len(embeddings)):
            energy = float(energies[idx])
            metrics[idx] = {
                "energy_score": energy,
                "energy_threshold": self._threshold,
                "predicted_class": predicted_classes[idx],
            }
            if energy > self._threshold:
                flags.add(idx)

        return flags, metrics

    @property
    def config_schema(self) -> type:
        return EnergyConfig

    def get_weight(self) -> float:
        return 0.30
