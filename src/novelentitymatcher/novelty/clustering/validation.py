"""
Cluster validation logic for novelty detection.

This module provides utilities for validating clustering results
and assessing cluster quality for novelty detection.
"""

import numpy as np


class ClusterValidator:
    """
    Validates clustering results for novelty detection.

    Provides metrics and validation methods to assess cluster
    quality and determine if samples represent novel clusters.
    """

    def __init__(
        self,
        min_cohesion_threshold: float = 0.45,
        min_persistence_threshold: float = 0.1,
    ):
        """
        Initialize the cluster validator.

        Args:
            min_cohesion_threshold: Minimum cohesion for valid clusters
            min_persistence_threshold: Minimum persistence for valid clusters
        """
        self.min_cohesion_threshold = min_cohesion_threshold
        self.min_persistence_threshold = min_persistence_threshold

    def compute_cohesion(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        cluster_id: int,
    ) -> float:
        """
        Compute cluster cohesion (compactness).

        Cohesion is the average pairwise similarity within a cluster.

        Args:
            embeddings: All embeddings
            labels: Cluster labels for each embedding
            cluster_id: Cluster to compute cohesion for

        Returns:
            Cohesion score (0-1, higher = more compact)
        """
        mask = labels == cluster_id
        if mask.sum() < 2:
            return 0.0

        cluster_embeddings = embeddings[mask]

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(cluster_embeddings, axis=1)
        normalized = cluster_embeddings / norms[:, np.newaxis]

        # Average pairwise similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        # Exclude diagonal
        np.fill_diagonal(similarity_matrix, 0)

        cohesion = similarity_matrix.sum() / (
            similarity_matrix.size - len(cluster_embeddings)
        )

        return float(cohesion)

    def compute_separation(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        cluster_id: int,
    ) -> float:
        """
        Compute cluster separation (distinctiveness from other clusters).

        Separation is the minimum average distance to another cluster.

        Args:
            embeddings: All embeddings
            labels: Cluster labels for each embedding
            cluster_id: Cluster to compute separation for

        Returns:
            Separation score (0-1, higher = more separated)
        """
        mask = labels == cluster_id
        if mask.sum() == 0:
            return 0.0

        cluster_embeddings = embeddings[mask]
        cluster_center = cluster_embeddings.mean(axis=0)

        unique_clusters = np.unique(labels)
        min_distance = float("inf")

        for other_id in unique_clusters:
            if other_id == cluster_id or other_id == -1:
                continue

            other_mask = labels == other_id
            other_embeddings = embeddings[other_mask]
            other_center = other_embeddings.mean(axis=0)

            # Cosine distance
            distance = 1.0 - np.dot(cluster_center, other_center) / (
                np.linalg.norm(cluster_center) * np.linalg.norm(other_center)
            )

            min_distance = min(min_distance, distance)

        return float(min_distance if min_distance != float("inf") else 0.0)

    def is_valid_cluster(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        cluster_id: int,
        min_size: int = 5,
    ) -> bool:
        """
        Determine if a cluster is valid (stable and meaningful).

        Args:
            embeddings: All embeddings
            labels: Cluster labels
            cluster_id: Cluster to validate
            min_size: Minimum number of samples for valid cluster

        Returns:
            True if cluster is valid
        """
        # Check size
        mask = labels == cluster_id
        if mask.sum() < min_size:
            return False

        # Check cohesion
        cohesion = self.compute_cohesion(embeddings, labels, cluster_id)
        if cohesion < self.min_cohesion_threshold:
            return False

        return True

    def get_cluster_statistics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> dict[int, dict[str, float]]:
        """
        Compute statistics for all clusters.

        Args:
            embeddings: All embeddings
            labels: Cluster labels

        Returns:
            Dict mapping cluster_id to statistics dict
        """
        unique_clusters = np.unique(labels)
        stats = {}

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise points
                continue

            mask = labels == cluster_id
            size = mask.sum()

            stats[cluster_id] = {
                "size": int(size),
                "cohesion": self.compute_cohesion(embeddings, labels, cluster_id),
                "separation": self.compute_separation(embeddings, labels, cluster_id),
                "is_valid": self.is_valid_cluster(embeddings, labels, cluster_id),
            }

        return stats
