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
from .base import NoveltyStrategy

logger = get_logger(__name__)


@StrategyRegistry.register
class MahalanobisDistanceStrategy(NoveltyStrategy):
    """
    Mahalanobis distance strategy for novelty detection.

    Computes the Mahalanobis distance from each sample to the class-conditional
    distribution (mean + shared covariance) of its predicted class. Samples
    whose distance exceeds a configurable threshold are flagged as novel.

    When ``calibration_mode="conformal"``, raw distances are wrapped with
    conformal p-values for statistically grounded routing. This is backward-
    compatible: ``calibration_mode="none"`` produces identical results to the
    original threshold-only behavior.
    """

    strategy_id = "mahalanobis"
    maturity = "production"

    def __init__(self):
        self._config: MahalanobisConfig = None
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
        """Initialize with conformal calibration, splitting reference data."""
        from .conformal import ConformalCalibrator

        n = len(reference_embeddings)
        frac = self._config.calibration_set_fraction
        n_cal = max(1, int(n * frac))
        if n_cal >= n:
            logger.warning(
                "Mahalanobis conformal calibration disabled because the calibration "
                "split would leave no core reference samples (n=%d, n_cal=%d)",
                n,
                n_cal,
            )
            self._initialize_core(reference_embeddings, reference_labels)
            return

        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        cal_indices = indices[:n_cal]
        core_indices = indices[n_cal:]

        core_embeddings = reference_embeddings[core_indices]
        core_labels = [reference_labels[i] for i in core_indices]

        self._initialize_core(core_embeddings, core_labels)

        cal_embeddings = reference_embeddings[cal_indices]
        cal_labels = [reference_labels[i] for i in cal_indices]
        cal_distances = self._compute_all_distances(cal_embeddings, cal_labels)

        self._calibrator = ConformalCalibrator(
            alpha=self._config.calibration_alpha,
            method=self._config.calibration_method,
        )
        self._calibrator.calibrate(cal_distances, np.array(cal_labels))
        logger.info(
            "Mahalanobis strategy initialized with conformal calibration: "
            "n_core=%d, n_cal=%d, method=%s",
            len(core_embeddings),
            n_cal,
            self._config.calibration_method,
        )

    def _initialize_core(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        """Core initialization: compute class means and pooled covariance."""
        unique_labels = set(reference_labels)
        for label in unique_labels:
            mask = np.array([ref_label == label for ref_label in reference_labels])
            class_embeddings = reference_embeddings[mask]
            self._class_means[label] = np.mean(class_embeddings, axis=0)

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
            if pred_class in self._class_means:
                class_mean = self._class_means[pred_class]
            else:
                class_mean = np.mean(list(self._class_means.values()), axis=0)
            diff = embeddings[i] - class_mean
            distances[i] = float(np.sqrt(np.abs(diff @ self._cov_inv @ diff)))
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

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        """
        Detect novel samples using Mahalanobis distance.

        When ``calibration_mode="conformal"``, flagging uses p-values
        instead of raw distance thresholds. A sample is flagged if
        ``p_value < calibration_alpha``.

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

        if (
            self._config.calibration_mode == "conformal"
            and self._calibrator is not None
            and self._calibrator.is_calibrated
        ):
            raw_distances = self._compute_all_distances(embeddings, predicted_classes)
            if self._config.calibration_method == "mondrian":
                p_values = self._calibrator.predict_pvalues_for_class(
                    raw_distances, predicted_classes
                )
            else:
                p_values = self._calibrator.predict_pvalues(raw_distances)

            for idx in range(len(embeddings)):
                metric = self._compute_mahalanobis_metrics(
                    idx,
                    embeddings[idx],
                    predicted_classes[idx],
                )
                metric["p_value"] = float(p_values[idx])
                metric["calibration_mode"] = "conformal"
                metrics[idx] = metric

                if p_values[idx] < self._config.calibration_alpha:
                    flags.add(idx)
        else:
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
        if predicted_class in self._class_means:
            class_mean = self._class_means[predicted_class]
        else:
            class_mean = np.mean(list(self._class_means.values()), axis=0)

        diff = embedding - class_mean
        left = diff @ self._cov_inv
        mahalanobis_dist = float(np.sqrt(np.abs(left @ diff)))

        euclidean_dist = float(np.linalg.norm(diff))

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
        return 0.35
