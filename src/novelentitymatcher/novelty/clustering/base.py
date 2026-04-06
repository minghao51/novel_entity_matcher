"""Abstract contract for clustering backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

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
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
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
