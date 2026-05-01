"""
Scalable density-based clustering for novelty detection.

Supports HDBSCAN, sOPTICS (accelerated), and UMAP-preprocessed clustering
for handling up to 1M scale with subquadratic runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...utils.logging_config import get_logger
from .backends import ClusteringBackendRegistry

logger = get_logger(__name__)


class ScalableClusterer:
    """
    Wrapper for scalable density-based clustering.

    Supports:
    - HDBSCAN: Standard hierarchical DBSCAN (best for <100K points)
    - sOPTICS: LSH-accelerated OPTICS (for 100K-1M points)
    - UMAP+HDBSCAN: UMAP dimensionality reduction before HDBSCAN
    - Auto: Automatic backend selection based on dataset size
    """

    BACKEND_HDBSCAN = "hdbscan"
    BACKEND_SOPTICS = "soptics"
    BACKEND_UMAP_HDBSCAN = "umap_hdbscan"
    BACKEND_AUTO = "auto"

    def __init__(
        self,
        backend: str = "auto",
        min_cluster_size: int = 5,
        min_samples: int = 5,
        cluster_selection_epsilon: float = 0.0,
        n_neighbors: int = 15,
        umap_dim: int = 10,
        umap_metric: str = "cosine",
        prediction_data: bool = True,
    ):
        """
        Initialize scalable clusterer.

        Args:
            backend: Clustering backend ('hdbscan', 'soptics', 'umap_hdbscan', 'auto')
            min_cluster_size: Minimum points to form a cluster.
            min_samples: Min samples for core distance (OPTICS).
            cluster_selection_epsilon: Distance threshold for cluster selection.
            n_neighbors: Neighbors for UMAP (if used).
            umap_dim: Target dimensionality for UMAP preprocessing.
            umap_metric: Metric for UMAP.
            prediction_data: Whether to compute prediction_data for HDBSCAN.
        """
        self.backend = backend
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.n_neighbors = n_neighbors
        self.umap_dim = umap_dim
        self.umap_metric = umap_metric
        self.prediction_data = prediction_data

        self._backend_instance: Any | None = None
        self._labels: np.ndarray | None = None
        self._probabilities: np.ndarray | None = None
        self._n_points: int = 0

    def _auto_backend(self, n_points: int) -> str:
        """Select backend based on dataset size."""
        if n_points < 100_000:
            return self.BACKEND_HDBSCAN
        elif n_points < 1_000_000:
            return self.BACKEND_SOPTICS
        else:
            return self.BACKEND_UMAP_HDBSCAN

    def _create_backend(self, backend_name: str) -> Any:
        """Create a backend instance from the registry."""
        kwargs: dict[str, Any] = {
            "min_samples": self.min_samples,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "prediction_data": self.prediction_data,
        }
        if backend_name == self.BACKEND_UMAP_HDBSCAN:
            kwargs.update(
                {
                    "n_neighbors": self.n_neighbors,
                    "umap_dim": self.umap_dim,
                    "umap_metric": self.umap_metric,
                }
            )
        return ClusteringBackendRegistry.create(backend_name, **kwargs)

    def fit_predict(
        self,
        embeddings: np.ndarray,
        metric: str = "cosine",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Fit clusterer and predict labels.

        Args:
            embeddings: Input embeddings (n_samples, dim)
            metric: Distance metric ('cosine', 'euclidean', 'precomputed')

        Returns:
            Tuple of (cluster_labels, probabilities, validation_info)
        """
        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        self._n_points = X.shape[0]

        backend_name = self.backend
        if backend_name == self.BACKEND_AUTO:
            backend_name = self._auto_backend(self._n_points)
            logger.info(
                f"Auto-selected backend: {backend_name} for {self._n_points} points"
            )

        self._backend_instance = self._create_backend(backend_name)

        labels, probabilities, backend_info = self._backend_instance.fit_predict(
            X, min_cluster_size=self.min_cluster_size, metric=metric
        )

        self._labels = labels
        self._probabilities = probabilities

        unique_clusters = sorted({int(label) for label in labels if int(label) >= 0})
        validation_info: dict[str, Any] = {
            "backend": backend_name,
            "n_points": self._n_points,
            "n_clusters": len(unique_clusters),
            "n_noise": int(np.sum(labels == -1)),
            "persistences": backend_info.get("persistences", []),
            "unique_clusters": unique_clusters,
        }

        logger.info(
            f"Clustering complete: {validation_info['n_clusters']} clusters, "
            f"{validation_info['n_noise']} noise points"
        )

        return labels, probabilities, validation_info

    def fit(
        self,
        embeddings: np.ndarray,
        metric: str = "cosine",
    ) -> ScalableClusterer:
        """Fit the clusterer (alias for compatibility)."""
        self.fit_predict(embeddings, metric=metric)
        return self

    @property
    def labels(self) -> np.ndarray | None:
        """Get cluster labels."""
        return self._labels

    @property
    def probabilities(self) -> np.ndarray | None:
        """Get cluster membership probabilities."""
        return self._probabilities

    def get_cluster_members(
        self,
        cluster_id: int,
    ) -> np.ndarray:
        """Get indices of members in a specific cluster."""
        if self._labels is None:
            raise RuntimeError("Clusterer must be fitted first")
        return np.where(self._labels == cluster_id)[0]

    def get_noise_points(self) -> np.ndarray:
        """Get indices of noise points (label = -1)."""
        if self._labels is None:
            raise RuntimeError("Clusterer must be fitted first")
        return np.where(self._labels == -1)[0]


def compute_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    known_embeddings: np.ndarray | None = None,
    metric: str = "cosine",
) -> dict[str, float]:
    """
    Compute quality metrics for discovered clusters.

    Args:
        embeddings: Cluster member embeddings (n_cluster, dim)
        labels: Cluster labels for all points (n_total,)
        known_embeddings: Optional known entity embeddings for ratio calculation
        metric: Distance metric

    Returns:
        Dictionary with quality metrics:
        - cohesion: avg pairwise distance within clusters (lower = better)
        - separation: avg distance between cluster centroids
        - silhouette: standard silhouette score
        - known_ratio: fraction of cluster close to known entities
    """
    unique_labels = sorted({int(label) for label in labels if int(label) >= 0})
    n_clusters = len(unique_labels)

    if n_clusters == 0:
        return {
            "cohesion": 0.0,
            "separation": 0.0,
            "silhouette": 0.0,
            "known_ratio": 0.0,
        }

    from sklearn.metrics import pairwise_distances

    cohesion_scores = []
    for cluster_id in unique_labels:
        member_indices = np.where(labels == cluster_id)[0]
        if len(member_indices) > 1:
            cluster_embeddings = embeddings[member_indices]
            pairwise_dists = pairwise_distances(cluster_embeddings, metric=metric)
            upper_tri = pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)]
            cohesion_scores.append(float(np.mean(upper_tri)))

    cohesion = float(np.mean(cohesion_scores)) if cohesion_scores else 0.0

    centroids: list[Any] = []
    for cluster_id in unique_labels:
        member_indices = np.where(labels == cluster_id)[0]
        centroid = np.mean(embeddings[member_indices], axis=0)
        centroids.append(centroid)
    centroids_array = np.array(centroids)

    if len(centroids_array) > 1:
        centroid_distances = pairwise_distances(centroids_array, metric=metric)
        upper_tri = centroid_distances[np.triu_indices_from(centroid_distances, k=1)]
        separation = float(np.mean(upper_tri))
    else:
        separation = 0.0

    if len(unique_labels) > 1 and len(embeddings) > len(unique_labels):
        try:
            from sklearn.metrics import silhouette_score

            silhouette = float(silhouette_score(embeddings, labels, metric=metric))
        except (ValueError, TypeError, RuntimeError):
            silhouette = 0.0
    else:
        silhouette = 0.0

    known_ratio = 0.0
    if known_embeddings is not None and len(known_embeddings) > 0:
        known_dists = pairwise_distances(embeddings, known_embeddings, metric=metric)
        min_known_dists = np.min(known_dists, axis=1)
        threshold = np.percentile(min_known_dists, 25)
        known_ratio = float(np.mean(min_known_dists < threshold))

    return {
        "cohesion": cohesion,
        "separation": separation,
        "silhouette": silhouette,
        "known_ratio": known_ratio,
    }


def validate_novel_cluster(
    cluster_embeddings: np.ndarray,
    known_embeddings: np.ndarray,
    cohesion_threshold: float = 0.45,
    known_ratio_threshold: float = 0.4,
    min_cluster_size: int = 5,
    metric: str = "cosine",
) -> tuple[bool, float]:
    """
    Validate that a cluster represents truly novel entities.

    Args:
        cluster_embeddings: Embeddings of cluster members
        known_embeddings: Embeddings of known entities
        cohesion_threshold: Max avg pairwise distance within cluster
        known_ratio_threshold: Max fraction that should be close to known
        min_cluster_size: Minimum required members
        metric: Distance metric

    Returns:
        Tuple of (is_valid_novel, validation_score)
    """
    from sklearn.metrics import pairwise_distances

    n_members = len(cluster_embeddings)

    if n_members < min_cluster_size:
        return False, 0.0

    if len(cluster_embeddings) > 1:
        pairwise_dists = pairwise_distances(cluster_embeddings, metric=metric)
        upper_tri = pairwise_dists[np.triu_indices_from(pairwise_dists, k=1)]
        cohesion = float(np.mean(upper_tri)) if upper_tri.size > 0 else 0.0
    else:
        cohesion = 0.0

    cohesion_valid = cohesion <= cohesion_threshold

    if known_embeddings is not None and len(known_embeddings) > 0:
        known_dists = pairwise_distances(
            cluster_embeddings, known_embeddings, metric=metric
        )
        min_known_dists = np.min(known_dists, axis=1)
        known_ratio = float(np.mean(min_known_dists < cohesion_threshold))
    else:
        known_ratio = 0.0

    known_valid = known_ratio <= known_ratio_threshold

    is_valid = bool(cohesion_valid and known_valid)

    score = float(
        np.mean(
            [
                1.0 - min(cohesion / cohesion_threshold, 1.0)
                if cohesion_threshold > 0
                else 1.0,
                1.0 - min(known_ratio / known_ratio_threshold, 1.0)
                if known_ratio_threshold > 0
                else 1.0,
            ]
        )
    )

    return is_valid, score
