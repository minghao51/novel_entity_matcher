"""Unit tests for monitoring.metrics module."""

from datetime import datetime

import pytest

from novelentitymatcher.monitoring.metrics import (
    LABEL_BACKEND,
    LABEL_MODE,
    LABEL_MODEL,
    LABEL_STAGE,
    LABEL_STRATEGY,
    METRIC_CACHE_HITS,
    METRIC_CLUSTERING_DURATION,
    METRIC_CLUSTERS_FOUND,
    METRIC_EMBEDDINGS_GENERATED,
    METRIC_EVIDENCE_LATENCY,
    METRIC_MATCH_BATCH_SIZE,
    METRIC_MATCH_LATENCY,
    METRIC_NOVEL_SAMPLES_COUNT,
    METRIC_NOVELTY_RATE,
    METRIC_OOD_DETECTION_LATENCY,
    METRIC_PROPOSAL_LATENCY,
    MetricEvent,
    create_metric,
    get_metric_summary,
)


class TestMetricEvent:
    def test_metric_event_creation(self):
        event = MetricEvent(
            name=METRIC_MATCH_LATENCY,
            value=0.123,
            unit="seconds",
            labels={LABEL_STAGE: "match"},
            timestamp=datetime.now(),
        )
        assert event.name == METRIC_MATCH_LATENCY
        assert event.value == pytest.approx(0.123)
        assert event.unit == "seconds"
        assert event.labels[LABEL_STAGE] == "match"
        assert isinstance(event.timestamp, datetime)

    def test_metric_event_default_labels(self):
        event = MetricEvent(
            name=METRIC_NOVELTY_RATE,
            value=0.05,
            unit="ratio",
            labels={},
            timestamp=datetime.now(),
        )
        assert event.labels == {}


class TestCreateMetric:
    def test_create_metric_with_defaults(self):
        event = create_metric(METRIC_MATCH_LATENCY, 0.5, "seconds")
        assert event.name == METRIC_MATCH_LATENCY
        assert event.value == pytest.approx(0.5)
        assert event.unit == "seconds"
        assert event.labels == {}
        assert isinstance(event.timestamp, datetime)

    def test_create_metric_with_labels(self):
        labels = {LABEL_STAGE: "ood", LABEL_STRATEGY: "confidence"}
        event = create_metric(METRIC_OOD_DETECTION_LATENCY, 0.1, "seconds", labels)
        assert event.labels == labels

    def test_create_metric_timestamp_is_current(self):
        before = datetime.now()
        event = create_metric(METRIC_CLUSTERING_DURATION, 1.0, "seconds")
        after = datetime.now()
        assert before <= event.timestamp <= after


class TestGetMetricSummary:
    def test_empty_list_returns_empty_dict(self):
        assert get_metric_summary([]) == {}

    def test_single_metric_summary(self):
        events = [
            create_metric(METRIC_MATCH_LATENCY, 0.1, "seconds"),
        ]
        summary = get_metric_summary(events)
        assert METRIC_MATCH_LATENCY in summary
        stats = summary[METRIC_MATCH_LATENCY]
        assert stats["count"] == 1
        assert stats["sum"] == pytest.approx(0.1)
        assert stats["mean"] == pytest.approx(0.1)
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.1)

    def test_multiple_events_same_metric(self):
        events = [
            create_metric(METRIC_MATCH_LATENCY, 0.1, "seconds"),
            create_metric(METRIC_MATCH_LATENCY, 0.2, "seconds"),
            create_metric(METRIC_MATCH_LATENCY, 0.3, "seconds"),
        ]
        summary = get_metric_summary(events)
        stats = summary[METRIC_MATCH_LATENCY]
        assert stats["count"] == 3
        assert stats["sum"] == pytest.approx(0.6)
        assert stats["mean"] == pytest.approx(0.2)
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.3)

    def test_multiple_metrics_grouped(self):
        events = [
            create_metric(METRIC_MATCH_LATENCY, 0.1, "seconds"),
            create_metric(METRIC_NOVELTY_RATE, 0.05, "ratio"),
            create_metric(METRIC_MATCH_LATENCY, 0.2, "seconds"),
            create_metric(METRIC_NOVELTY_RATE, 0.1, "ratio"),
        ]
        summary = get_metric_summary(events)
        assert set(summary.keys()) == {METRIC_MATCH_LATENCY, METRIC_NOVELTY_RATE}
        assert summary[METRIC_MATCH_LATENCY]["count"] == 2
        assert summary[METRIC_NOVELTY_RATE]["count"] == 2

    def test_metric_summary_with_negative_values(self):
        events = [
            create_metric(METRIC_MATCH_LATENCY, -0.1, "seconds"),
            create_metric(METRIC_MATCH_LATENCY, 0.2, "seconds"),
        ]
        summary = get_metric_summary(events)
        stats = summary[METRIC_MATCH_LATENCY]
        assert stats["min"] == pytest.approx(-0.1)
        assert stats["max"] == pytest.approx(0.2)


class TestMetricConstants:
    def test_all_constants_are_strings(self):
        constants = [
            METRIC_MATCH_LATENCY,
            METRIC_NOVELTY_RATE,
            METRIC_CLUSTERING_DURATION,
            METRIC_OOD_DETECTION_LATENCY,
            METRIC_EVIDENCE_LATENCY,
            METRIC_PROPOSAL_LATENCY,
            METRIC_CLUSTERS_FOUND,
            METRIC_NOVEL_SAMPLES_COUNT,
            METRIC_CACHE_HITS,
            METRIC_EMBEDDINGS_GENERATED,
            METRIC_MATCH_BATCH_SIZE,
        ]
        for c in constants:
            assert isinstance(c, str)
            assert len(c) > 0

    def test_all_label_constants_are_strings(self):
        labels = [LABEL_STAGE, LABEL_STRATEGY, LABEL_BACKEND, LABEL_MODEL, LABEL_MODE]
        for label in labels:
            assert isinstance(label, str)
            assert len(label) > 0
