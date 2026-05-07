"""Tests for StabilityFilterStage."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from novelentitymatcher.pipeline.contracts import StageContext
from novelentitymatcher.pipeline.stages.stability_filter import StabilityFilterStage


@dataclass
class FakeSample:
    index: int
    text: str
    confidence: float
    predicted_class: str
    novelty_score: float | None = None
    cluster_id: int | None = None


@dataclass
class FakeCluster:
    cluster_id: int
    sample_indices: list[int]
    sample_count: int
    example_texts: list[str]
    mean_novelty_score: float | None = None
    mean_confidence: float | None = None
    metadata: dict | None = None


@dataclass
class FakeReport:
    novel_samples: list[Any]


@dataclass
class FakeMatchResult:
    embeddings: np.ndarray


def _hdbscan_available() -> bool:
    try:
        import hdbscan  # noqa: F401

        return True
    except ImportError:
        return False


class TestStabilityFilterStage:
    def test_disabled_passes_all_clusters(self):
        clusters = [
            FakeCluster(0, [0, 1], 2, ["a", "b"]),
            FakeCluster(1, [2, 3], 2, ["c", "d"]),
        ]
        context = StageContext(
            inputs=["a", "b", "c", "d"],
            artifacts={
                "discovery_clusters": clusters,
                "novel_sample_report": FakeReport([]),
                "match_result": FakeMatchResult(np.zeros((4, 4))),
            },
        )
        stage = StabilityFilterStage(enabled=False)
        result = stage.run(context)
        assert len(result.artifacts["discovery_clusters"]) == 2

    def test_empty_clusters_passes_through(self):
        context = StageContext(
            inputs=["a"],
            artifacts={
                "discovery_clusters": [],
                "novel_sample_report": FakeReport([]),
                "match_result": FakeMatchResult(np.zeros((1, 4))),
            },
        )
        stage = StabilityFilterStage(enabled=True)
        result = stage.run(context)
        assert result.artifacts["discovery_clusters"] == []

    def test_metadata_reports_filter_stats(self):
        clusters = [
            FakeCluster(0, [0, 1], 2, ["a", "b"]),
        ]
        context = StageContext(
            inputs=["a", "b"],
            artifacts={
                "discovery_clusters": clusters,
                "novel_sample_report": FakeReport([]),
                "match_result": FakeMatchResult(np.zeros((2, 4))),
            },
        )
        stage = StabilityFilterStage(enabled=True)
        result = stage.run(context)
        assert result.metadata["enabled"] is True
        assert "skip_reason" in result.metadata


@pytest.mark.skipif(not _hdbscan_available(), reason="hdbscan not installed")
class TestStabilityFilterStageWithRealData:
    def test_filters_unstable_clusters(self):
        rng = np.random.RandomState(42)
        stable_a = rng.randn(30, 4) + np.array([20, 0, 0, 0])
        stable_b = rng.randn(30, 4) + np.array([0, 20, 0, 0])
        noise = rng.randn(10, 4)
        embs = np.vstack([stable_a, stable_b, noise]).astype(np.float32)

        c0 = FakeCluster(0, list(range(30)), 30, [f"a{i}" for i in range(30)])
        c1 = FakeCluster(1, list(range(30, 60)), 30, [f"b{i}" for i in range(30)])
        c2 = FakeCluster(2, list(range(60, 70)), 10, [f"n{i}" for i in range(10)])

        samples = []
        for i in range(70):
            samples.append(FakeSample(i, f"s{i}", 0.5, "x"))

        context = StageContext(
            inputs=[f"s{i}" for i in range(70)],
            artifacts={
                "discovery_clusters": [c0, c1, c2],
                "novel_sample_report": FakeReport(samples),
                "match_result": FakeMatchResult(embs),
            },
        )
        stage = StabilityFilterStage(
            enabled=True,
            stability_threshold=0.3,
            n_bootstrap=3,
            sample_fraction=0.8,
        )
        result = stage.run(context)
        filtered = result.artifacts["discovery_clusters"]
        assert len(filtered) >= 1
        assert result.metadata["num_clusters_before"] == 3
