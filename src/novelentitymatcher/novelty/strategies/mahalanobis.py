"""
Mahalanobis distance-based novelty detection strategy.

Flags samples based on their Mahalanobis distance to the class-conditional
distribution of their predicted class.
"""

from typing import Dict, List, Set, Any, Optional
import numpy as np

from .base import NoveltyStrategy
from ..core.strategies import StrategyRegistry
from ..config.strategies import MahalanobisConfig


@StrategyRegistry.register
class MahalanobisDistanceStrategy(NoveltyStrategy):
    """
    Mahalanobis distance strategy for novelty detection.

    Computes the Mahalanobis distance from each sample to the class-conditional
    distribution (mean + shared covariance) of its predicted class. Samples
    whose distance exceeds a configurable threshold are flagged as novel.
    """

    strategy_id = "mahalanobis"

    def __init__(self):
        self._config: MahalanobisConfig = None
        self._class_means: Dict[str, np.ndarray] = {}
        self._cov_inv: Optional[np.ndarray] = None
        self._dim: int = 0

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: MahalanobisConfig,
    ) -> None:
        """
        Initialize the Mahalanobis strategy with reference data.

        Computes per-class mean vectors and a shared (pooled) covariance matrix
        with regularization for numerical stability.

        Args:
            reference_embeddings: Embeddings of known samples (n_samples, dim)
            reference_labels: Class labels for known samples
            config: MahalanobisConfig with threshold, regularization, etc.
        """
        self._config = config or MahalanobisConfig()
        self._dim = reference_embeddings.shape[1]

        # Compute per-class means
        unique_labels = set(reference_labels)
        for label in unique_labels:
            mask = np.array([ref_label == label for ref_label in reference_labels])
            class_embeddings = reference_embeddings[mask]
            self._class_means[label] = np.mean(class_embeddings, axis=0)

        # Compute shared (pooled) covariance matrix
        if self._config.use_class_conditional:
            cov = self._compute_pooled_covariance(
                reference_embeddings, reference_labels
            )
        else:
            cov = np.cov(reference_embeddings, rowvar=False)

        # Add regularization for numerical stability
        cov += self._config.regularization * np.eye(self._dim)

        # Compute inverse of covariance matrix
        self._cov_inv = np.linalg.inv(cov)

    def _compute_pooled_covariance(
        self,
        embeddings: np.ndarray,
        labels: List[str],
    ) -> np.ndarray:
        """
        Compute the pooled (within-class) covariance matrix.

        Args:
            embeddings: All reference embeddings
            labels: Corresponding class labels

        Returns:
            Pooled covariance matrix (dim, dim)
        """
        unique_labels = set(labels)
        n_samples, dim = embeddings.shape
        pooled_cov = np.zeros((dim, dim))

        for label in unique_labels:
            mask = np.array([sample_label == label for sample_label in labels])
            class_embeddings = embeddings[mask]
            class_mean = self._class_means[label]
            diff = class_embeddings - class_mean
            pooled_cov += diff.T @ diff

        # Normalize by total number of samples minus number of classes
        n_classes = len(unique_labels)
        pooled_cov /= max(n_samples - n_classes, 1)

        return pooled_cov

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using Mahalanobis distance.

        Args:
            texts: Input texts
            embeddings: Text embeddings
            predicted_classes: Predicted classes
            confidences: Prediction confidences
            **kwargs: Additional parameters

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
        flags = set()
        metrics = {}

        for idx in range(len(embeddings)):
            metric = self._compute_mahalanobis_metrics(
                idx,
                embeddings[idx],
                predicted_classes[idx],
            )
            metrics[idx] = metric

            if metric["mahalanobis_distance"] >= self._config.threshold:
                flags.add(idx)

        return flags, metrics

    def _compute_mahalanobis_metrics(
        self,
        idx: int,
        embedding: np.ndarray,
        predicted_class: str,
    ) -> Dict[str, Any]:
        """
        Compute Mahalanobis distance metrics for a single sample.

        Args:
            idx: Sample index
            embedding: The embedding vector
            predicted_class: Predicted class for this sample

        Returns:
            Dictionary with Mahalanobis metrics
        """
        # Get the mean for the predicted class (fallback to global mean if unknown)
        if predicted_class in self._class_means:
            class_mean = self._class_means[predicted_class]
        else:
            # Fallback: use average of all class means
            class_mean = np.mean(list(self._class_means.values()), axis=0)

        # Compute Mahalanobis distance: sqrt((x - mu)^T * Sigma^{-1} * (x - mu))
        diff = embedding - class_mean
        left = diff @ self._cov_inv
        mahalanobis_dist = float(np.sqrt(np.abs(left @ diff)))

        # Euclidean distance to class mean (for interpretability)
        euclidean_dist = float(np.linalg.norm(diff))

        # Novelty score: normalized Mahalanobis distance (0-1 range via sigmoid-like transform)
        novelty_score = 1.0 - np.exp(-mahalanobis_dist / self._config.threshold)

        return {
            "mahalanobis_distance": mahalanobis_dist,
            "mahalanobis_novelty_score": float(novelty_score),
            "predicted_class_mean_distance": euclidean_dist,
            "predicted_class": predicted_class,
        }

    @property
    def config_schema(self) -> type:
        """Return MahalanobisConfig as the config schema."""
        return MahalanobisConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        # Mahalanobis is a strong parametric signal, slightly less than kNN
        return 0.35
