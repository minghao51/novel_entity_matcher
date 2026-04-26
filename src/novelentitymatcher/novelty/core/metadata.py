"""
Metadata builder for novelty detection reports.

This module creates rich reports with sample metadata, strategy outputs,
and novelty scores.
"""

from typing import Any

import numpy as np

from ..config.base import DetectionConfig
from ..schemas import NovelSampleMetadata, NovelSampleReport


class MetadataBuilder:
    """
    Builds comprehensive reports for novelty detection results.

    Aggregates information from all strategies and creates detailed
    reports with per-sample metrics and explanations.
    """

    def __init__(self):
        """Initialize the metadata builder."""

    def build_report(
        self,
        texts: list[str],
        confidences: np.ndarray,
        predicted_classes: list[str],
        novel_indices: set[int],
        novelty_scores: dict[int, float],
        all_metrics: dict[int, dict[str, Any]],
        strategy_outputs: dict[str, tuple[set[int], dict]],
        config: DetectionConfig,
    ) -> NovelSampleReport:
        """
        Build a comprehensive novelty detection report.

        Args:
            texts: Input texts
            confidences: Prediction confidence scores
            predicted_classes: Predicted class for each sample
            novel_indices: Indices flagged as novel
            novelty_scores: Final novelty scores
            all_metrics: All per-sample metrics
            strategy_outputs: Per-strategy outputs
            config: Detection configuration

        Returns:
            NovelSampleReport with all detection results
        """
        signal_counts: dict[str, int] = {}
        novel_samples: list[NovelSampleMetadata] = []

        for strategy_id, (flags, _) in strategy_outputs.items():
            signal_counts[strategy_id] = len(flags)

        for idx in sorted(novel_indices):
            metrics = all_metrics.get(idx, {})
            signals = {
                strategy_id: idx in flags
                for strategy_id, (flags, _) in strategy_outputs.items()
            }
            novel_samples.append(
                NovelSampleMetadata(
                    text=texts[idx],
                    index=idx,
                    confidence=float(confidences[idx]),
                    predicted_class=predicted_classes[idx],
                    novelty_score=float(novelty_scores.get(idx, 0.0)),
                    margin_score=metrics.get("margin_score"),
                    entropy_score=metrics.get("entropy_score"),
                    uncertainty_score=metrics.get("uncertainty_score"),
                    knn_novelty_score=metrics.get("knn_novelty_score"),
                    knn_mean_distance=metrics.get("knn_mean_distance"),
                    knn_max_distance=metrics.get("knn_max_distance"),
                    cluster_id=metrics.get("cluster_label"),
                    cluster_support_score=metrics.get("cluster_support_score"),
                    signals=signals,
                    metrics=metrics,
                )
            )

        return NovelSampleReport(
            novel_samples=novel_samples,
            detection_strategies=list(strategy_outputs.keys()),
            config=config.model_dump() if hasattr(config, "model_dump") else {},
            signal_counts=signal_counts,
        )

    def build_summary(self, report: NovelSampleReport) -> dict[str, Any]:
        """
        Build a summary of the detection report.

        Args:
            report: NovelSampleReport to summarize

        Returns:
            Summary dictionary with key statistics
        """
        total_samples = len(report.novel_samples)
        return {
            "total_samples": total_samples,
            "novel_samples": len(report.novel_samples),
            "novel_ratio": len(report.novel_samples) / total_samples
            if total_samples
            else 0.0,
            "avg_novelty_score": np.mean(
                [
                    sample.novelty_score
                    for sample in report.novel_samples
                    if sample.novelty_score is not None
                ]
            )
            if report.novel_samples
            else 0.0,
            "strategies_used": report.detection_strategies,
            "strategy_counts": report.signal_counts,
        }
