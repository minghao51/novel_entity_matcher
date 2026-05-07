"""
Result dataclasses for novelty detection.

Contains data structures for detection results, metrics, and reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import NovelSampleReport


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
class DetectionReport:
    """
    Report from a complete detection run.

    Contains the novel sample report plus additional metadata about
    the detection run (timing, strategy performance, etc.).
    """

    novelty_report: NovelSampleReport
    strategies_used: list[str]
    runtime_seconds: float
    timestamp: str
    additional_info: dict[str, Any] = field(default_factory=dict)


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
