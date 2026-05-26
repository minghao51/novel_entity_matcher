"""
Clustering-based novelty detection strategy.

Flags samples that form small, isolated clusters or don't fit
well into any existing cluster.
"""

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from ..clustering.scalable import ScalableClusterer
from ..clustering.validation import ClusterValidator
from ..config.strategies import ClusteringConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy, SignalInfo


@StrategyRegistry.register
class ClusteringStrategy(NoveltyStrategy):
    """
    Clustering-based strategy for novelty detection.

    Uses HDBSCAN to cluster samples and identifies novel samples
    as those that are in small or low-cohesion clusters.
    """

    strategy_id = "clustering"
    maturity = "experimental"
    score_keys = ("cluster_support_score",)
    signal_info = SignalInfo(
        score_key="cluster_support_score",
        flag_key="cluster_is_novel",
        weight_name="cluster",
        kind="score",
    )
    default_weight = 0.2

    def __init__(self):
        self._config: ClusteringConfig = None
        self._clusterer: ScalableClusterer = None
        self._validator: ClusterValidator = None
        self._reference_embeddings: np.ndarray = None
        self._reference_labels: list[str] = None
        self._ref_cluster_labels: np.ndarray = None
        self._ref_centroids: dict[int, np.ndarray] = None
        self._ref_cluster_valid: dict[int, bool] = None
        self._ref_cluster_cohesion: dict[int, float] = None

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

        ref_clusterer = ScalableClusterer(
            min_cluster_size=self._config.hdbscan_min_cluster_size,
            min_samples=self._config.hdbscan_min_samples,
            cluster_selection_epsilon=self._config.cluster_selection_epsilon,
        )
        ref_clusterer.fit(reference_embeddings)
        self._ref_cluster_labels = ref_clusterer.labels
        self._ref_centroids = {}
        self._ref_cluster_valid = {}
        self._ref_cluster_cohesion = {}
        for lbl in np.unique(self._ref_cluster_labels):
            if lbl == -1:
                continue
            mask = self._ref_cluster_labels == lbl
            self._ref_centroids[int(lbl)] = np.mean(reference_embeddings[mask], axis=0)
            is_valid = self._validator.is_valid_cluster(
                reference_embeddings,
                self._ref_cluster_labels,
                lbl,
                min_size=self._config.min_cluster_size,
            )
            cohesion = self._validator.compute_cohesion(
                reference_embeddings, self._ref_cluster_labels, lbl
            )
            self._ref_cluster_valid[int(lbl)] = is_valid
            self._ref_cluster_cohesion[int(lbl)] = cohesion

    def _detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs: Any,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        if self._reference_embeddings is None:
            return set(), {}
        if not self._ref_centroids:
            all_embeddings = np.vstack([self._reference_embeddings, embeddings])
            self._clusterer.fit(all_embeddings)
            labels = self._clusterer.labels
            query_labels = labels[len(self._reference_embeddings) :]
        else:
            centroid_labels = list(self._ref_centroids.keys())
            centroid_matrix = np.array(
                [self._ref_centroids[cl] for cl in centroid_labels]
            )
            dists = cosine_distances(embeddings, centroid_matrix)
            nearest_idx = np.argmin(dists, axis=1)
            query_labels = np.array([centroid_labels[i] for i in nearest_idx])
            min_dists = dists[np.arange(len(nearest_idx)), nearest_idx]
            noise_percentile = self._config.noise_percentile
            min_dists = dists[np.arange(len(nearest_idx)), nearest_idx]
            if len(min_dists) > 1:
                noise_mask = min_dists > np.percentile(min_dists, noise_percentile)
            else:
                noise_mask = np.array([False])
            query_labels[noise_mask] = -1
            labels = np.concatenate([self._ref_cluster_labels, query_labels])
            all_embeddings = np.vstack([self._reference_embeddings, embeddings])

        flags = set()
        metrics = {}

        unique_labels = np.unique(query_labels)

        for label in unique_labels:
            if label == -1:
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
                int_label = int(label)
                all_mask = labels == label
                _cluster_embeddings = all_embeddings[all_mask]

                if int_label in self._ref_cluster_valid:
                    is_valid = self._ref_cluster_valid[int_label]
                    cohesion = self._ref_cluster_cohesion[int_label]
                else:
                    is_valid = self._validator.is_valid_cluster(
                        all_embeddings,
                        labels,
                        label,
                        min_size=self._config.min_cluster_size,
                    )
                    cohesion = self._validator.compute_cohesion(
                        all_embeddings, labels, label
                    )

                support_score = 1.0 - cohesion

                query_mask = query_labels == label
                query_indices = np.where(query_mask)[0]

                for idx in query_indices:
                    is_novel = not is_valid or support_score < (
                        1.0 - self._config.cohesion_threshold
                    )

                    if is_novel:
                        flags.add(idx)

                    metrics[idx] = {
                        "cluster_label": int_label,
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
