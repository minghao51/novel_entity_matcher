"""Drift scoring utilities.

Compare current embeddings against a baseline DistributionSnapshot to
produce a drift report with global and per-class scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DriftReport:
    """Result of a drift comparison."""

    global_drift_score: float
    per_class_drift: dict[str, float]
    drift_detected: bool
    method: str
    recommendation: str


class DriftScorer:
    """Compare current distribution against a baseline snapshot.

    Supports multiple methods:
    - ``mmd_linear``: Maximum Mean Discrepancy with linear kernel (fast).
    - ``kl_gaussian``: KL divergence between fitted Gaussians (accurate).
    - ``cosine_centroid``: Cosine distance between global centroids.
    """

    def __init__(self, method: str = "mmd_linear", threshold: float = 0.05):
        self.method = method
        self.threshold = threshold

    def score(
        self,
        current: np.ndarray,
        baseline_mean: np.ndarray,
        baseline_cov: np.ndarray | None = None,
        per_class_baselines: dict[str, dict[str, np.ndarray]] | None = None,
        current_labels: list[str] | None = None,
    ) -> DriftReport:
        """Score drift between current embeddings and a baseline.

        Args:
            current: Current embedding batch (n_samples, dim).
            baseline_mean: Mean vector from baseline snapshot.
            baseline_cov: Covariance matrix from baseline snapshot (optional).
            per_class_baselines: Mapping class -> {"mean": ..., "cov": ...}.
            current_labels: Predicted labels for current samples (optional).

        Returns:
            DriftReport with scores and recommendation.
        """
        if self.method == "mmd_linear":
            global_score = self._mmd_linear(current, baseline_mean)
        elif self.method == "kl_gaussian":
            if baseline_cov is None:
                raise ValueError("kl_gaussian requires baseline_cov")
            global_score = self._kl_gaussian(current, baseline_mean, baseline_cov)
        elif self.method == "cosine_centroid":
            global_score = self._cosine_centroid(current, baseline_mean)
        else:
            raise ValueError(f"Unknown drift method: {self.method}")

        per_class: dict[str, float] = {}
        if per_class_baselines and current_labels is not None:
            current_labels_arr = np.array(current_labels)
            for label, stats in per_class_baselines.items():
                mask = current_labels_arr == label
                if mask.sum() == 0:
                    continue
                class_current = current[mask]
                class_mean = stats["mean"]
                sim = cosine_similarity(
                    class_current.mean(axis=0).reshape(1, -1),
                    class_mean.reshape(1, -1),
                )[0, 0]
                per_class[label] = float(1.0 - sim)

        detected = global_score > self.threshold
        recommendation = self._recommendation(detected, global_score)

        return DriftReport(
            global_drift_score=round(global_score, 6),
            per_class_drift=per_class,
            drift_detected=detected,
            method=self.method,
            recommendation=recommendation,
        )

    def _mmd_linear(self, current: np.ndarray, baseline_mean: np.ndarray) -> float:
        """MMD with linear kernel = squared norm of mean difference."""
        mean_diff = current.mean(axis=0) - baseline_mean
        return float(mean_diff @ mean_diff)

    def _kl_gaussian(
        self,
        current: np.ndarray,
        baseline_mean: np.ndarray,
        baseline_cov: np.ndarray,
    ) -> float:
        """KL divergence between two Gaussian approximations."""
        mu0 = baseline_mean
        mu1 = current.mean(axis=0)
        sigma0 = baseline_cov + np.eye(baseline_cov.shape[0]) * 1e-6
        sigma1 = np.cov(current, rowvar=False) + np.eye(mu0.shape[0]) * 1e-6
        if sigma1.ndim < 2:
            sigma1 = np.array([[sigma1]])

        sigma0_inv = np.linalg.inv(sigma0)
        dim = mu0.shape[0]

        trace_term = float(np.trace(sigma0_inv @ sigma1))
        mean_diff = mu1 - mu0
        quad_term = float(mean_diff @ sigma0_inv @ mean_diff)
        logdet_term = float(np.linalg.slogdet(sigma0)[1] - np.linalg.slogdet(sigma1)[1])
        kl = 0.5 * (trace_term + quad_term - dim + logdet_term)
        return float(np.clip(kl, 0.0, None))

    def _cosine_centroid(self, current: np.ndarray, baseline_mean: np.ndarray) -> float:
        """1 - cosine similarity between global centroids."""
        sim = cosine_similarity(
            current.mean(axis=0).reshape(1, -1),
            baseline_mean.reshape(1, -1),
        )[0, 0]
        return float(1.0 - sim)

    def _recommendation(self, detected: bool, score: float) -> str:
        if not detected:
            return "monitor"
        if score > self.threshold * 3:
            return "retrain"
        return "add_entities"
