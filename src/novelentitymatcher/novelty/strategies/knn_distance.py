"""
kNN distance-based novelty detection strategy.

Flags samples based on their distance to k-nearest neighbors in the
reference set.
"""

from typing import Dict, List, Set, Any, Optional
import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import KNNConfig
from ..storage.index import ANNIndex


@StrategyRegistry.register
class KNNDistanceStrategy(NoveltyStrategy):
    """
    kNN distance strategy for novelty detection.

    Flags samples as novel if their average distance to k-nearest
    neighbors in the reference set exceeds a threshold.
    """

    strategy_id = "knn_distance"
    maturity = "production"

    def __init__(self):
        self._config: KNNConfig = None
        self._ann_index: Optional[ANNIndex] = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
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

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using kNN distance.

        Args:
            texts: Input texts
            embeddings: Text embeddings
            predicted_classes: Predicted classes
            confidences: Prediction confidences
            **kwargs: Additional parameters

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
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
    ) -> Dict[str, Any]:
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

        # Average distance to k-nearest neighbors
        mean_distance = float(np.mean(distances))

        # Maximum distance (to the farthest neighbor)
        max_distance = float(np.max(distances))

        # Check if predicted class matches neighbors
        # (This would require label info, placeholder for now)
        class_match_ratio = 1.0  # Placeholder

        # Compute novelty score
        # Higher distance = more novel
        novelty_score = mean_distance

        return {
            "knn_mean_distance": mean_distance,
            "knn_max_distance": max_distance,
            "knn_novelty_score": novelty_score,
            "knn_class_match_ratio": class_match_ratio,
            "knn_predicted_class": predicted_class,
        }

    @property
    def config_schema(self) -> type:
        """Return KNNConfig as the config schema."""
        return KNNConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        # kNN is a strong signal, give it high weight
        return 0.45
