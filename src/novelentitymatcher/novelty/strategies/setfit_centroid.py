"""
SetFit centroid distance novelty detection strategy.

Computes minimum cosine distance from each query to known class centroids
in the SetFit fine-tuned embedding space. Produces continuous novelty scores.

This is the recommended strategy when SetFit full training is used for Phase 1,
as contrastive learning creates tight, well-separated class clusters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import SetFitCentroidConfig


@StrategyRegistry.register
class SetFitCentroidStrategy(NoveltyStrategy):
    """
    Centroid distance strategy using SetFit fine-tuned embeddings.

    For each known class, computes a centroid in the SetFit embedding space.
    Novelty score = minimum cosine distance from query to any centroid.
    """

    strategy_id = "setfit_centroid"

    def __init__(self) -> None:
        self._config: SetFitCentroidConfig | None = None
        self._centroids: Optional[np.ndarray] = None
        self._class_labels: Optional[List[str]] = None
        self._threshold: Optional[float] = None
        self._setfit_model: Optional[Any] = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: SetFitCentroidConfig,
    ) -> None:
        """
        Initialize centroids from reference embeddings.

        Args:
            reference_embeddings: Embeddings of known samples (already from SetFit model)
            reference_labels: Class labels for known samples
            config: SetFitCentroidConfig with threshold
        """
        self._config = config or SetFitCentroidConfig()
        self._class_labels = list(set(reference_labels))

        # Compute per-class centroids
        centroids = {}
        for label in self._class_labels:
            mask = np.array(reference_labels) == label
            class_embeddings = reference_embeddings[mask]
            if len(class_embeddings) > 0:
                centroids[label] = np.mean(class_embeddings, axis=0)

        # Sort centroids by class label for consistent indexing
        self._centroids = np.array(
            [centroids[label] for label in sorted(centroids.keys())]
        )
        self._class_labels = sorted(centroids.keys())

        # Calibrate threshold from reference set if not explicitly set
        if self._config.threshold is None:
            self._threshold = self._calibrate_threshold(
                reference_embeddings, reference_labels
            )
        else:
            self._threshold = self._config.threshold

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using centroid distance.

        Args:
            texts: Input texts (unused, embeddings are pre-computed)
            embeddings: Query embeddings
            predicted_classes: Predicted class for each sample
            confidences: Prediction confidence scores

        Returns:
            (flags, metrics) - flagged indices and per-sample metrics
        """
        if self._centroids is None or self._threshold is None:
            return set(), {}

        flags: Set[int] = set()
        metrics: Dict[int, Dict[str, Any]] = {}

        # Normalize embeddings for cosine distance
        query_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_norms = np.where(query_norms == 0, 1, query_norms)
        query_normalized = embeddings / query_norms

        centroid_norms = np.linalg.norm(self._centroids, axis=1, keepdims=True)
        centroid_norms = np.where(centroid_norms == 0, 1, centroid_norms)
        centroids_normalized = self._centroids / centroid_norms

        # Compute cosine similarity matrix (queries x centroids)
        similarity_matrix = query_normalized @ centroids_normalized.T

        # Convert to cosine distance
        distance_matrix = 1.0 - similarity_matrix

        for idx in range(len(embeddings)):
            distances = distance_matrix[idx]
            min_distance = float(np.min(distances))
            nearest_centroid_idx = int(np.argmin(distances))
            nearest_class = self._class_labels[nearest_centroid_idx]

            # Continuous novelty score (normalized to [0, 1] via sigmoid)
            novelty_score = self._distance_to_score(min_distance)

            is_novel = min_distance > self._threshold

            if is_novel:
                flags.add(idx)

            metrics[idx] = {
                "setfit_centroid_min_distance": min_distance,
                "setfit_centroid_nearest_class": nearest_class,
                "setfit_centroid_novelty_score": novelty_score,
                "setfit_centroid_is_novel": is_novel,
                "setfit_centroid_predicted_class": predicted_classes[idx],
                "setfit_centroid_confidence": float(confidences[idx]),
            }

        return flags, metrics

    def _distance_to_score(self, distance: float) -> float:
        """
        Convert raw cosine distance to a [0, 1] novelty score.

        Uses sigmoid scaling centered at the threshold so that:
        - distance << threshold → score ≈ 0
        - distance == threshold → score ≈ 0.5
        - distance >> threshold → score ≈ 1
        """
        if self._threshold is None or self._threshold == 0:
            return min(distance, 1.0)

        steepness = 10.0
        score = 1.0 / (1.0 + np.exp(-steepness * (distance - self._threshold)))
        return float(np.clip(score, 0.0, 1.0))

    def _calibrate_threshold(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
    ) -> float:
        """
        Calibrate threshold using the reference set.

        Uses the 95th percentile of distances from each sample to its own
        class centroid, ensuring most known samples fall below threshold.
        """
        unique_labels = list(set(reference_labels))
        centroids = {}

        for label in unique_labels:
            mask = np.array(reference_labels) == label
            class_embeddings = reference_embeddings[mask]
            if len(class_embeddings) > 0:
                centroids[label] = np.mean(class_embeddings, axis=0)

        distances = []
        for i, label in enumerate(reference_labels):
            if label in centroids:
                centroid = centroids[label]
                # Cosine distance
                query_norm = np.linalg.norm(reference_embeddings[i])
                centroid_norm = np.linalg.norm(centroid)
                if query_norm > 0 and centroid_norm > 0:
                    sim = np.dot(reference_embeddings[i], centroid) / (
                        query_norm * centroid_norm
                    )
                    distances.append(1.0 - sim)

        if not distances:
            return 0.5

        return float(np.percentile(distances, 95))

    @property
    def config_schema(self) -> type:
        return SetFitCentroidConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        return 0.45
