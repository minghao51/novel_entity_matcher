"""
Result dataclasses for novelty detection.

Contains data structures for detection results, metrics, and reports.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyMetrics:
    """
    Metrics from a single strategy.

    Contains the flags and per-sample metrics produced by a strategy.
    """

    strategy_id: str
    """Identifier for the strategy."""

    flags: set[int]
    """Indices flagged as novel by this strategy."""

    metrics: dict[int, dict[str, Any]]
    """Per-sample metrics from this strategy."""


@dataclass
class SampleMetrics:
    """
    Aggregated metrics for a single sample.

    Contains metrics from all strategies for a specific sample.
    """

    index: int
    """Sample index in the input batch."""

    text: str
    """The input text."""

    predicted_class: str
    """Predicted class for this sample."""

    confidence: float
    """Prediction confidence score."""

    is_novel: bool
    """Whether this sample was flagged as novel."""

    novelty_score: float
    """Final combined novelty score."""

    strategy_flags: dict[str, bool]
    """Which strategies flagged this sample."""

    raw_metrics: dict[str, Any]
    """Raw metrics from each strategy."""


@dataclass
class NovelSampleReport:
    """
    Comprehensive report from novelty detection.

    Contains all results from running novelty detection on a batch
    of samples.
    """

    novel_indices: list[int]
    """Indices of samples flagged as novel."""

    novel_scores: dict[int, float]
    """Novelty scores for all flagged samples."""

    num_novel: int
    """Number of samples flagged as novel."""

    num_total: int
    """Total number of samples processed."""

    novel_ratio: float
    """Ratio of novel samples (num_novel / num_total)."""

    sample_metadata: list[dict[str, Any]]
    """Per-sample metadata including text, class, confidence, metrics."""

    strategy_flags: dict[str, dict[str, Any]]
    """Strategy-level statistics (num_flagged, flagged_indices)."""

    config_used: dict[str, Any]
    """Configuration used for detection."""

    def get_novel_samples(self) -> list[dict[str, Any]]:
        """
        Get metadata for only the novel samples.

        Returns:
            List of metadata dicts for novel samples
        """
        return [m for m in self.sample_metadata if m["is_novel"]]

    def get_strategy_novel_count(self, strategy_id: str) -> int:
        """
        Get number of samples flagged by a specific strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Number of samples flagged by the strategy
        """
        return self.strategy_flags.get(strategy_id, {}).get("num_flagged", 0)

    def get_sample_at_index(self, index: int) -> dict[str, Any] | None:
        """
        Get metadata for a specific sample by index.

        Args:
            index: Sample index

        Returns:
            Metadata dict if index is valid, None otherwise
        """
        if 0 <= index < len(self.sample_metadata):
            return self.sample_metadata[index]
        return None


@dataclass
class DetectionReport:
    """
    Report from a complete detection run.

    Contains the NovelSampleReport plus additional metadata about
    the detection run (timing, strategy performance, etc.).
    """

    novelty_report: NovelSampleReport
    """The core novelty detection report."""

    strategies_used: list[str]
    """List of strategies that were used."""

    runtime_seconds: float
    """Time taken for detection in seconds."""

    timestamp: str
    """ISO timestamp of when detection was run."""

    additional_info: dict[str, Any] = field(default_factory=dict)
    """Any additional information to include in the report."""


@dataclass
class EvaluationReport:
    """
    Report from evaluating novelty detection.

    Contains metrics from evaluating on a labeled dataset.
    """

    auroc: float
    """Area under ROC curve."""

    auprc: float
    """Area under Precision-Recall curve."""

    detection_rate_at_1: float
    """Detection rate at 1% false positive rate."""

    detection_rate_at_5: float
    """Detection rate at 5% false positive rate."""

    detection_rate_at_10: float
    """Detection rate at 10% false positive rate."""

    precision: float
    """Precision at optimal threshold."""

    recall: float
    """Recall at optimal threshold."""

    f1: float
    """F1 score at optimal threshold."""

    optimal_threshold: float
    """Threshold that maximizes F1 score."""

    confusion_matrix: dict[str, int] | None = None
    """Confusion matrix at optimal threshold."""

    per_class_metrics: dict[str, dict[str, float]] | None = None
    """Per-class metrics if available."""

    num_samples: int = 0
    """Total number of samples evaluated."""

    num_novel: int = 0
    """Number of actually novel samples."""

    timestamp: str = ""
    """ISO timestamp of when evaluation was run."""
