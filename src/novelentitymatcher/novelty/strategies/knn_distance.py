"""
kNN distance-based novelty detection strategy.

Flags samples based on their distance to k-nearest neighbors in the
reference set.
"""

from typing import Any

import numpy as np

from ..config.strategies import KNNConfig
from ..core.strategies import StrategyRegistry
from ..storage.index import ANNIndex
from .base import NoveltyStrategy, SignalInfo


@StrategyRegistry.register
class KNNDistanceStrategy(NoveltyStrategy):
    """
    kNN distance strategy for novelty detection.

    Flags samples as novel if their average distance to k-nearest
    neighbors in the reference set exceeds a threshold.
    """

    strategy_id = "knn_distance"
    maturity = "production"
    score_keys = ("knn_novelty_score",)
    signal_info = SignalInfo(
        score_key="knn_novelty_score",
        flag_key="knn_is_novel",
        weight_name="knn",
        kind="score",
    )
    default_weight = 0.45

    def __init__(self):
        self._config: KNNConfig | None = None
        self._ann_index: ANNIndex | None = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: KNNConfig,
    ) -> None:
        """
        Initialize the kNN strategy with reference data.

        Args:
            reference_embeddings: Embeddings of known samples
            reference_labels: Labels of known samples
            config: KNNConfig with k, thresholds, and metric
        """
        self._config = config or KNNConfig()

        # Initialize ANN index
        self._ann_index = ANNIndex(
            dim=reference_embeddings.shape[1],
            max_elements=len(reference_labels),
        )
        self._ann_index.add_vectors(reference_embeddings, reference_labels)

    def _detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs: Any,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        if self._ann_index is None:
            return set(), {}
        k = min(self._config.k, self._ann_index.n_elements)

        # Query kNN
        similarities, neighbor_indices = self._ann_index.knn_query(embeddings, k=k)

        flags = set()
        metrics = {}

        for idx in range(len(embeddings)):
            metric = self._compute_knn_metrics(
                idx,
                similarities[idx],
                neighbor_indices[idx],
                predicted_classes[idx],
            )
            metrics[idx] = metric

            # Check if novelty score exceeds threshold
            if metric["knn_novelty_score"] >= self._config.distance_threshold:
                flags.add(idx)

        return flags, metrics

    def _compute_knn_metrics(
        self,
        idx: int,
        similarities: np.ndarray,
        neighbor_indices: np.ndarray,
        predicted_class: str,
    ) -> dict[str, Any]:
        """
        Compute kNN-based metrics for a single sample.

        Args:
            idx: Sample index
            similarities: Similarities to k-nearest neighbors
            neighbor_indices: Indices of k-nearest neighbors
            predicted_class: Predicted class for this sample

        Returns:
            Dictionary with kNN metrics
        """
        # Convert similarities to distances (cosine distance = 1 - similarity)
        distances = 1.0 - similarities

        mean_distance = float(np.mean(distances))

        max_distance = float(np.max(distances))

        novelty_score = mean_distance

        return {
            "knn_mean_distance": mean_distance,
            "knn_max_distance": max_distance,
            "knn_novelty_score": novelty_score,
            "knn_predicted_class": predicted_class,
        }

    @property
    def config_schema(self) -> type:
        """Return KNNConfig as the config schema."""
        return KNNConfig
