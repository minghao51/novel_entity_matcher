"""Metrics schema and utilities for library-centric observability.

Provides MetricEvent dataclass and helper functions for emitting
metrics through user-provided callbacks. Works without external dependencies
and is suitable for library usage patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class MetricEvent:
    """Standardized metric event structure.

    Can be emitted from any component (matcher, stages, detectors)
    and sent to user-provided callbacks for custom handling.

    Attributes:
        name: Metric name (e.g., "match_latency", "novelty_rate")
        value: Numeric value of the metric
        unit: Unit of measurement (e.g., "seconds", "count", "ratio")
        labels: Key-value pairs for categorization (e.g., {"stage": "ood"})
        timestamp: When the metric was recorded
    """

    name: str
    value: float
    unit: str
    labels: dict[str, str]
    timestamp: datetime


# Standard metric names
METRIC_MATCH_LATENCY = "match_latency"
METRIC_NOVELTY_RATE = "novelty_rate"
METRIC_CLUSTERING_DURATION = "clustering_duration"
METRIC_OOD_DETECTION_LATENCY = "ood_detection_latency"
METRIC_EVIDENCE_LATENCY = "evidence_latency"
METRIC_PROPOSAL_LATENCY = "proposal_latency"
METRIC_CLUSTERS_FOUND = "clusters_found"
METRIC_NOVEL_SAMPLES_COUNT = "novel_samples_count"
METRIC_CACHE_HITS = "cache_hits"
METRIC_EMBEDDINGS_GENERATED = "embeddings_generated"
METRIC_MATCH_BATCH_SIZE = "match_batch_size"

# Standard label keys
LABEL_STAGE = "stage"
LABEL_STRATEGY = "strategy"
LABEL_BACKEND = "backend"
LABEL_MODEL = "model"
LABEL_MODE = "mode"


def create_metric(
    name: str,
    value: float,
    unit: str,
    labels: dict[str, str] | None = None,
) -> MetricEvent:
    """Helper to create MetricEvent with current timestamp.

    Args:
        name: Metric name
        value: Numeric value
        unit: Unit of measurement
        labels: Optional labels dictionary

    Returns:
        MetricEvent with current timestamp
    """
    return MetricEvent(
        name=name,
        value=value,
        unit=unit,
        labels=labels or {},
        timestamp=datetime.now(),
    )


def get_metric_summary(events: list[MetricEvent]) -> dict[str, dict[str, float]]:
    """Summarize metric events by name.

    Args:
        events: List of MetricEvent instances

    Returns:
        Dictionary mapping metric names to summary statistics
        {metric_name: {"count": N, "sum": X, "mean": Y, ...}}
    """
    from collections import defaultdict

    grouped: dict[str, list[float]] = defaultdict(list)

    for event in events:
        grouped[event.name].append(event.value)

    summary: dict[str, dict[str, float]] = {}
    for name, values in grouped.items():
        summary[name] = {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    return summary
