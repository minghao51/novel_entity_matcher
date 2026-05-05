"""Pipeline stages for drift detection and other advanced checks."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ...novelty.drift.scorer import DriftScorer
from ...novelty.drift.snapshot import DistributionSnapshot
from ..contracts import PipelineStage, StageContext, StageResult

logger = logging.getLogger(__name__)


class DriftCheckStage(PipelineStage):
    """Optional pipeline stage that compares query embeddings against a
    stored baseline distribution and emits a drift report.
    """

    name = "drift_check"

    def __init__(
        self,
        baseline_path: str | Path | None = None,
        method: str = "mmd_linear",
        threshold: float = 0.05,
        enabled: bool = True,
    ):
        self.baseline_path = baseline_path
        self.method = method
        self.threshold = threshold
        self.enabled = enabled

    def run(self, context: StageContext) -> StageResult:
        if not self.enabled or self.baseline_path is None:
            return StageResult(
                stage_name=self.name,
                artifacts={"drift_report": None},
                metadata={"enabled": self.enabled, "skipped": True},
            )

        baseline = DistributionSnapshot.load(self.baseline_path)
        current_embeddings = self._extract_embeddings(context)

        if current_embeddings is None or len(current_embeddings) == 0:
            return StageResult(
                stage_name=self.name,
                artifacts={"drift_report": None},
                metadata={"error": "no_query_embeddings"},
            )

        scorer = DriftScorer(method=self.method, threshold=self.threshold)
        report = scorer.score(
            current=current_embeddings,
            baseline_mean=baseline.mean,
            baseline_cov=baseline.covariance,
            per_class_baselines=baseline.per_class_stats,
            current_labels=self._extract_labels(context),
        )

        if report.drift_detected:
            logger.warning(
                "Drift detected: score=%.4f method=%s recommendation=%s",
                report.global_drift_score,
                report.method,
                report.recommendation,
            )

        return StageResult(
            stage_name=self.name,
            artifacts={"drift_report": report},
            metadata={
                "drift_detected": report.drift_detected,
                "global_drift_score": report.global_drift_score,
                "method": report.method,
                "recommendation": report.recommendation,
            },
            stage_config_snapshot={
                "method": self.method,
                "threshold": self.threshold,
                "enabled": self.enabled,
            },
        )

    def _extract_embeddings(self, context: StageContext) -> np.ndarray | None:
        """Try to pull query embeddings from upstream artifacts."""
        match_result = context.artifacts.get("match_result")
        if match_result is not None and hasattr(match_result, "embeddings"):
            return match_result.embeddings
        return context.artifacts.get("query_embeddings")

    def _extract_labels(self, context: StageContext) -> list[str] | None:
        """Try to pull predicted labels from upstream artifacts."""
        match_result = context.artifacts.get("match_result")
        if match_result is not None and hasattr(match_result, "predictions"):
            return match_result.predictions
        return None
