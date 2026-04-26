"""
Unified novelty detection evaluator.

Supports both benchmark and research evaluation modes with
comprehensive metrics and reporting.
"""

from datetime import datetime
from typing import Literal

import numpy as np

from ..schemas.reports import EvaluationReport
from .metrics import (
    compute_auprc,
    compute_auroc,
    compute_confusion_matrix,
    compute_detection_rates,
    compute_precision_recall_f1,
)


class NoveltyEvaluator:
    """
    Unified evaluator for novelty detection.

    Supports two modes:
    - benchmark: Quick evaluation on OOD splits with core metrics
    - research: Comprehensive evaluation with confusion matrices and threshold sweeping

    Metrics computed:
    - AUROC, AUPRC
    - Detection rates at 1%, 5%, 10% FPR
    - Precision, Recall, F1 at optimal threshold
    """

    def __init__(
        self,
        mode: Literal["benchmark", "research"] = "benchmark",
        metrics: list[str] | None = None,
    ):
        """
        Initialize the evaluator.

        Args:
            mode: Evaluation mode ('benchmark' or 'research')
            metrics: List of metrics to compute (None for default based on mode)
        """
        self.mode = mode
        self.metrics = metrics or self._default_metrics_for_mode(mode)

    def _default_metrics_for_mode(self, mode: str) -> list[str]:
        """Get default metrics for evaluation mode."""
        if mode == "benchmark":
            return ["auroc", "auprc", "detection_rate_5"]
        else:  # research
            return [
                "auroc",
                "auprc",
                "detection_rate_1",
                "detection_rate_5",
                "detection_rate_10",
                "precision",
                "recall",
                "f1",
            ]

    def evaluate(
        self,
        novelty_scores: np.ndarray,
        is_novel_true: np.ndarray,
        threshold: float | None = None,
    ) -> dict[str, float]:
        """
        Evaluate novelty detection performance.

        Args:
            novelty_scores: Predicted novelty scores (higher = more novel)
            is_novel_true: Ground truth novelty labels (True = novel)
            threshold: Optional threshold for discrete predictions

        Returns:
            Dictionary of metric name -> value
        """
        scores = np.asarray(novelty_scores)
        labels = np.asarray(is_novel_true, dtype=bool)

        results = {}

        # AUROC and AUPRC
        if "auroc" in self.metrics:
            results["auroc"] = compute_auroc(scores, labels)

        if "auprc" in self.metrics:
            results["auprc"] = compute_auprc(scores, labels)

        # Detection rates at various FPR thresholds
        if any(m.startswith("detection_rate_") for m in self.metrics):
            dr_metrics = [m for m in self.metrics if m.startswith("detection_rate_")]
            fpr_thresholds = []
            for m in dr_metrics:
                if m == "detection_rate_1":
                    fpr_thresholds.append(0.01)
                elif m == "detection_rate_5":
                    fpr_thresholds.append(0.05)
                elif m == "detection_rate_10":
                    fpr_thresholds.append(0.10)

            if fpr_thresholds:
                detection_rates = compute_detection_rates(
                    scores, labels, tuple(fpr_thresholds)
                )
                results.update(detection_rates)

        # Precision, Recall, F1
        if any(m in ["precision", "recall", "f1"] for m in self.metrics):
            prf_results = compute_precision_recall_f1(scores, labels, threshold)
            if "precision" in self.metrics:
                results["precision"] = prf_results["precision"]
            if "recall" in self.metrics:
                results["recall"] = prf_results["recall"]
            if "f1" in self.metrics:
                results["f1"] = prf_results["f1"]
            results["optimal_threshold"] = prf_results["threshold"]

        return results

    def create_report(
        self,
        novelty_scores: np.ndarray,
        is_novel_true: np.ndarray,
        threshold: float | None = None,
    ) -> EvaluationReport:
        """
        Create a comprehensive evaluation report.

        Args:
            novelty_scores: Predicted novelty scores (higher = more novel)
            is_novel_true: Ground truth novelty labels (True = novel)
            threshold: Optional threshold for discrete predictions

        Returns:
            EvaluationReport with all metrics
        """
        scores = np.asarray(novelty_scores)
        labels = np.asarray(is_novel_true, dtype=bool)

        # Compute all metrics
        auroc = compute_auroc(scores, labels)
        auprc = compute_auprc(scores, labels)

        detection_rates = compute_detection_rates(scores, labels)
        dr_at_1 = detection_rates.get("detection_rate_1", 0.0)
        dr_at_5 = detection_rates.get("detection_rate_5", 0.0)
        dr_at_10 = detection_rates.get("detection_rate_10", 0.0)

        prf_results = compute_precision_recall_f1(scores, labels, threshold)
        optimal_threshold = prf_results["threshold"]

        # Confusion matrix
        cm = compute_confusion_matrix(scores, labels, optimal_threshold)

        return EvaluationReport(
            auroc=auroc,
            auprc=auprc,
            detection_rate_at_1=dr_at_1,
            detection_rate_at_5=dr_at_5,
            detection_rate_at_10=dr_at_10,
            precision=prf_results["precision"],
            recall=prf_results["recall"],
            f1=prf_results["f1"],
            optimal_threshold=optimal_threshold,
            confusion_matrix=cm,
            num_samples=len(scores),
            num_novel=int(np.sum(labels)),
            timestamp=datetime.now().isoformat(),
        )

    def sweep_thresholds(
        self,
        novelty_scores: np.ndarray,
        is_novel_true: np.ndarray,
        num_thresholds: int = 100,
    ) -> dict[str, np.ndarray]:
        """
        Sweep across thresholds and compute metrics at each.

        Args:
            novelty_scores: Predicted novelty scores (higher = more novel)
            is_novel_true: Ground truth novelty labels (True = novel)
            num_thresholds: Number of thresholds to evaluate

        Returns:
            Dict with arrays for thresholds and metrics
        """
        from .metrics import sweep_thresholds

        thresholds = np.linspace(0, 1, num_thresholds)
        return sweep_thresholds(novelty_scores, is_novel_true, thresholds)

    def compare_thresholds(
        self,
        novelty_scores: np.ndarray,
        is_novel_true: np.ndarray,
        thresholds: list[float],
    ) -> list[dict[str, float]]:
        """
        Compare metrics at specific thresholds.

        Args:
            novelty_scores: Predicted novelty scores (higher = more novel)
            is_novel_true: Ground truth novelty labels (True = novel)
            thresholds: List of thresholds to evaluate

        Returns:
            List of dicts with metrics at each threshold
        """
        results = []
        for thresh in thresholds:
            metrics = self.evaluate(novelty_scores, is_novel_true, threshold=thresh)
            metrics["threshold"] = thresh
            results.append(metrics)
        return results
