"""
Mahalanobis distance-based novelty detection strategy.

Flags samples based on their Mahalanobis distance to the class-conditional
distribution of their predicted class. Supports optional conformal calibration
for statistically grounded p-value based novelty routing.
"""

from typing import Any

import numpy as np

from ...utils.logging_config import get_logger
from ..config.strategies import MahalanobisConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy, SignalInfo
from .conformal_mixin import ConformalMixin

logger = get_logger(__name__)


@StrategyRegistry.register
class MahalanobisDistanceStrategy(ConformalMixin, NoveltyStrategy):
    strategy_id = "mahalanobis"
    maturity = "production"
    score_keys = ("mahalanobis_novelty_score",)
    signal_info = SignalInfo(
        score_key="mahalanobis_novelty_score",
        flag_key="mahalanobis_is_novel",
        weight_name="mahalanobis",
        kind="score",
    )
    default_weight = 0.35

    def __init__(self):
        self._config: MahalanobisConfig | None = None
        self._class_means: dict[str, np.ndarray] = {}
        self._cov_inv: np.ndarray | None = None
        self._dim: int = 0
        self._calibrator: Any = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
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
        self._class_means = {}
        self._cov_inv = None
        self._calibrator = None

        if self._config.calibration_mode == "conformal":
            self._initialize_with_calibration(reference_embeddings, reference_labels)
        else:
            self._initialize_core(reference_embeddings, reference_labels)

    def _initialize_with_calibration(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        self._run_conformal_calibration(
            reference_embeddings,
            reference_labels,
            self._compute_all_distances,
        )

    def _initialize_core(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        """Core initialization: compute class means and pooled covariance."""
        self._class_means = self.compute_class_means(
            reference_embeddings, reference_labels
        )

        if self._config.use_class_conditional:
            cov = self._compute_pooled_covariance(
                reference_embeddings, reference_labels
            )
        else:
            cov = np.cov(reference_embeddings, rowvar=False)

        cov += self._config.regularization * np.eye(self._dim)
        self._cov_inv = np.linalg.inv(cov)

    def _compute_all_distances(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> np.ndarray:
        """Compute Mahalanobis distances for a batch of samples."""
        distances = np.empty(len(embeddings))
        for i in range(len(embeddings)):
            pred_class = labels[i]
            global_mean = np.mean(list(self._class_means.values()), axis=0)
            class_mean = self._resolve_class_mean(
                pred_class, self._class_means, global_mean
            )
            diff = embeddings[i] - class_mean
            quad = float(diff @ self._cov_inv @ diff)
            if quad < 0:
                logger.warning(
                    "Negative quadratic form in Mahalanobis distance: %.6f", quad
                )
            distances[i] = float(np.sqrt(max(quad, 0.0)))
        return distances

    def _compute_pooled_covariance(
        self,
        embeddings: np.ndarray,
        labels: list[str],
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

        n_classes = len(unique_labels)
        pooled_cov /= max(n_samples - n_classes, 1)

        return pooled_cov

    def _detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs: Any,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        flags = set()
        metrics = {}

        if self._is_conformal_active():
            raw_distances = self._compute_all_distances(embeddings, predicted_classes)
            global_mean = np.mean(list(self._class_means.values()), axis=0)
            return self._conformal_detect_loop(
                raw_distances,
                predicted_classes,
                lambda idx, _score, _pv: {
                    "mahalanobis_distance": float(raw_distances[idx]),
                    "euclidean_distance": float(
                        np.linalg.norm(
                            embeddings[idx]
                            - self._resolve_class_mean(
                                predicted_classes[idx], self._class_means, global_mean
                            )
                        )
                    ),
                    "mahalanobis_is_novel": _pv < self._config.calibration_alpha,
                },
            )

        for idx in range(len(embeddings)):
            metric = self._compute_mahalanobis_metrics(
                idx, embeddings[idx], predicted_classes[idx]
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
    ) -> dict[str, Any]:
        """
        Compute Mahalanobis distance metrics for a single sample.

        Args:
            idx: Sample index
            embedding: The embedding vector
            predicted_class: Predicted class for this sample

        Returns:
            Dictionary with Mahalanobis metrics
        """
        global_mean = np.mean(list(self._class_means.values()), axis=0)
        class_mean = self._resolve_class_mean(
            predicted_class, self._class_means, global_mean
        )

        diff = embedding - class_mean
        quad = float((diff @ self._cov_inv) @ diff)
        if quad < 0:
            logger.warning(
                "Negative quadratic form in Mahalanobis distance: %.6f", quad
            )
        mahalanobis_dist = float(np.sqrt(max(quad, 0.0)))

        euclidean_dist = float(np.linalg.norm(diff))

        novelty_score = 1.0 - np.exp(-mahalanobis_dist / self._config.threshold)

        is_novel = mahalanobis_dist >= self._config.threshold

        return {
            "mahalanobis_distance": mahalanobis_dist,
            "mahalanobis_novelty_score": float(novelty_score),
            "mahalanobis_is_novel": is_novel,
            "predicted_class_mean_distance": euclidean_dist,
            "predicted_class": predicted_class,
        }

    @property
    def config_schema(self) -> type:
        """Return MahalanobisConfig as the config schema."""
        return MahalanobisConfig
