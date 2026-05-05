"""Abstract contract for clustering backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ClusteringBackend(ABC):
    """Abstract contract for clustering backends."""

    name: str

    @abstractmethod
    def fit_predict(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Fit and predict cluster labels.

        Args:
            embeddings: Input embeddings (n_samples, dim).
            min_cluster_size: Minimum points to form a cluster.
            **kwargs: Backend-specific parameters.

        Returns:
            (labels, probabilities, info_dict)
            - labels: array of cluster assignments (-1 for noise)
            - probabilities: array of cluster membership probabilities
            - info: dict with backend-specific metadata
        """
        ...

    def get_condensed_tree(self) -> dict[str, Any]:
        """Return the cluster hierarchy condensed tree for multi-resolution analysis."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support condensed tree"
        )

    def extract_clusters_at_stability(
        self, min_persistence: float = 0.1
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Re-extract clusters at a different stability threshold."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support cluster re-extraction"
        )
