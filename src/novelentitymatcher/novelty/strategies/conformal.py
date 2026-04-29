"""
Conformal prediction-based calibration for OOD detection strategies.

Wraps raw strategy scores with statistically grounded p-values,
enabling rigorous routing of out-of-distribution inputs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class ConformalCalibrator:
    """Calibrate raw OOD scores into conformal p-values.

    Supports two methods:
    - ``"split"``: Holds out a fraction of reference data for calibration.
    - ``"mondrian"``: Uses class-conditional (Mondrian) conformal calibration
      with per-class nonconformity distributions.

    Usage::

        cal = ConformalCalibrator(alpha=0.1, method="split")
        cal.calibrate(raw_scores, labels)
        pvals = cal.predict_pvalues(test_scores)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        method: Literal["mondrian", "split"] = "split",
    ):
        self.alpha = alpha
        self.method = method
        self._nonconformity_scores: np.ndarray | None = None
        self._class_scores: dict[str, np.ndarray] = {}
        self._n_calibration: int = 0
        self._is_calibrated: bool = False

    def calibrate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> ConformalCalibrator:
        """Compute nonconformity scores from calibration data.

        Args:
            scores: Raw OOD scores for calibration samples, shape (n_samples,).
                    Higher scores indicate more anomalous / novel.
            labels: Class labels for calibration samples, shape (n_samples,).

        Returns:
            Self for fluent chaining.
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels)

        if self.method == "mondrian":
            self._calibrate_mondrian(scores, labels)
        else:
            self._nonconformity_scores = np.sort(scores)
            self._n_calibration = len(scores)

        self._is_calibrated = True
        logger.info(
            "Conformal calibration complete: method=%s, n=%d, alpha=%.3f",
            self.method,
            self._n_calibration,
            self.alpha,
        )
        return self

    def predict_pvalues(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw OOD scores to calibrated p-values.

        Args:
            scores: Raw scores for test samples, shape (n_samples,).

        Returns:
            p-values, shape (n_samples,). Lower p-value = more likely OOD.
        """
        if not self._is_calibrated:
            raise RuntimeError(
                "Calibrator has not been calibrated. Call calibrate() first."
            )

        scores = np.asarray(scores, dtype=np.float64)

        if self.method == "mondrian":
            return self._predict_mondrian_pvalues(scores)

        return self._compute_pvalues(scores, self._nonconformity_scores)

    def _compute_pvalues(
        self,
        test_scores: np.ndarray,
        calibration_scores: np.ndarray,
    ) -> np.ndarray:
        """Compute p-values as fraction of calibration scores >= test score.

        For OOD detection, higher raw scores mean more anomalous.
        p-value = (count(calibration >= test) + 1) / (n_calibration + 1)
        """
        n_cal = len(calibration_scores)
        pvalues = np.empty(len(test_scores))
        for i, score in enumerate(test_scores):
            first_ge = np.searchsorted(calibration_scores, score, side="left")
            pvalues[i] = (n_cal - first_ge + 1) / (n_cal + 1)
        return pvalues

    def _calibrate_mondrian(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Per-class (Mondrian) conformal calibration."""
        self._class_scores = {}
        total = 0
        for label in np.unique(labels):
            mask = labels == label
            class_scores = np.sort(scores[mask])
            self._class_scores[str(label)] = class_scores
            total += len(class_scores)
        self._n_calibration = total

    def _predict_mondrian_pvalues(self, scores: np.ndarray) -> np.ndarray:
        """Predict p-values using Mondrian (class-conditional) approach.

        Falls back to global calibration if no class information available.
        """
        if not self._class_scores:
            if self._nonconformity_scores is not None:
                return self._compute_pvalues(scores, self._nonconformity_scores)
            return np.ones(len(scores))

        all_cal = np.concatenate(list(self._class_scores.values()))
        return self._compute_pvalues(scores, np.sort(all_cal))

    def predict_pvalues_for_class(
        self,
        scores: np.ndarray,
        predicted_classes: list[str],
    ) -> np.ndarray:
        """Compute class-conditional p-values when predicted classes are known.

        Args:
            scores: Raw OOD scores for test samples.
            predicted_classes: Predicted class for each sample.

        Returns:
            p-values, shape (n_samples,).
        """
        if not self._is_calibrated:
            raise RuntimeError(
                "Calibrator has not been calibrated. Call calibrate() first."
            )

        scores = np.asarray(scores, dtype=np.float64)
        pvalues = np.empty(len(scores))

        all_cal = (
            np.sort(np.concatenate(list(self._class_scores.values())))
            if self._class_scores
            else self._nonconformity_scores
        )

        for i, (score, pred_class) in enumerate(
            zip(scores, predicted_classes, strict=False)
        ):
            class_cal = self._class_scores.get(str(pred_class))
            if class_cal is not None and len(class_cal) > 0:
                pvalues[i] = self._compute_pvalues(np.array([score]), class_cal)[0]
            elif all_cal is not None:
                pvalues[i] = self._compute_pvalues(np.array([score]), all_cal)[0]
            else:
                pvalues[i] = 1.0

        return pvalues

    @property
    def calibration_metadata(self) -> dict:
        """Return calibration metadata for reproducibility."""
        return {
            "alpha": self.alpha,
            "method": self.method,
            "n_calibration": self._n_calibration,
            "is_calibrated": self._is_calibrated,
            "n_classes": len(self._class_scores),
        }

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated
