"""
Clustering-based novelty detection strategy.

Flags samples that form small, isolated clusters or don't fit
well into any existing cluster.
"""

from typing import Any

import numpy as np

from ..clustering.scalable import ScalableClusterer
from ..clustering.validation import ClusterValidator
from ..config.strategies import ClusteringConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy


@StrategyRegistry.register
class ClusteringStrategy(NoveltyStrategy):
    """
    Clustering-based strategy for novelty detection.

    Uses HDBSCAN to cluster samples and identifies novel samples
    as those that are in small or low-cohesion clusters.
    """

    strategy_id = "clustering"
    maturity = "experimental"

    def __init__(self):
        self._config: ClusteringConfig = None
        self._clusterer: ScalableClusterer = None
        self._validator: ClusterValidator = None
        self._reference_embeddings: np.ndarray = None
        self._reference_labels: list[str] = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: ClusteringConfig,
    ) -> None:
        """
        Initialize the clustering strategy.

        Args:
            reference_embeddings: Embeddings of known samples
            reference_labels: Labels of known samples
            config: ClusteringConfig with thresholds
        """
        self._config = config or ClusteringConfig()
        self._reference_embeddings = reference_embeddings
        self._reference_labels = reference_labels

        # Initialize clusterer
        self._clusterer = ScalableClusterer(
            min_cluster_size=self._config.hdbscan_min_cluster_size,
            min_samples=self._config.hdbscan_min_samples,
            cluster_selection_epsilon=self._config.cluster_selection_epsilon,
        )

        # Initialize validator
        self._validator = ClusterValidator(
            min_cohesion_threshold=self._config.cohesion_threshold,
            min_persistence_threshold=self._config.persistence_threshold,
        )

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        """
        Detect novel samples using clustering.

        Args:
            texts: Input texts
            embeddings: Text embeddings
            predicted_classes: Predicted classes
            confidences: Prediction confidences
            **kwargs: Additional parameters

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
        # Combine reference and query embeddings for clustering
        all_embeddings = np.vstack([self._reference_embeddings, embeddings])

        # Fit clusterer on all embeddings
        self._clusterer.fit(all_embeddings)

        # Get cluster labels
        labels = self._clusterer.labels

        # Separate query labels (reference samples come first)
        query_labels = labels[len(self._reference_embeddings) :]

        flags = set()
        metrics = {}

        # Validate clusters and identify novel samples
        unique_labels = np.unique(query_labels)

        for label in unique_labels:
            if label == -1:  # Noise points
                # All noise points are novel
                mask = query_labels == label
                indices = np.where(mask)[0]
                for idx in indices:
                    flags.add(idx)
                    metrics[idx] = {
                        "cluster_label": -1,
                        "cluster_support_score": 0.0,
                        "cluster_is_novel": True,
                        "cluster_size": 1,
                    }
            else:
                # Check if cluster is valid
                # Get all embeddings with this label (including reference)
                all_mask = labels == label
                _cluster_embeddings = all_embeddings[all_mask]

                is_valid = self._validator.is_valid_cluster(
                    all_embeddings,
                    labels,
                    label,
                    min_size=self._config.min_cluster_size,
                )

                # Compute support score (1 - cohesion)
                cohesion = self._validator.compute_cohesion(
                    all_embeddings, labels, label
                )
                support_score = 1.0 - cohesion

                # Get query indices for this cluster
                query_mask = query_labels == label
                query_indices = np.where(query_mask)[0]

                for idx in query_indices:
                    # Novel if cluster is invalid or support score is low
                    is_novel = not is_valid or support_score < (
                        1.0 - self._config.cohesion_threshold
                    )

                    if is_novel:
                        flags.add(idx)

                    metrics[idx] = {
                        "cluster_label": int(label),
                        "cluster_support_score": support_score,
                        "cluster_is_novel": is_novel,
                        "cluster_size": int(np.sum(all_mask)),
                        "cluster_cohesion": cohesion,
                    }

        return flags, metrics

    @property
    def config_schema(self) -> type:
        """Return ClusteringConfig as the config schema."""
        return ClusteringConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        # Clustering provides complementary signal
        return 0.2
