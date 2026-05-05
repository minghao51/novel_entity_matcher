"""Tests for DriftCheckStage pipeline integration."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from novelentitymatcher.novelty.drift.snapshot import DistributionSnapshot
from novelentitymatcher.pipeline.contracts import StageContext
from novelentitymatcher.pipeline.stages.drift_hook import DriftCheckStage


class TestDriftCheckStage:
    @pytest.fixture
    def baseline_snapshot(self):
        embs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
        labels = ["a", "a", "b", "b"]
        return DistributionSnapshot.from_embeddings(embs, labels)

    @pytest.fixture
    def baseline_path(self, baseline_snapshot):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline"
            baseline_snapshot.save(path)
            yield str(path)

    def test_disabled_stage_returns_none_report(self):
        stage = DriftCheckStage(enabled=False)
        context = StageContext(inputs=["test"])
        result = stage.run(context)
        assert result.artifacts["drift_report"] is None
        assert result.metadata["skipped"] is True

    def test_missing_baseline_returns_none_report(self):
        stage = DriftCheckStage(baseline_path=None, enabled=True)
        context = StageContext(inputs=["test"])
        result = stage.run(context)
        assert result.artifacts["drift_report"] is None

    def test_no_embeddings_returns_error_metadata(self, baseline_path):
        stage = DriftCheckStage(baseline_path=baseline_path, enabled=True)
        context = StageContext(inputs=["test"])
        result = stage.run(context)
        assert result.artifacts["drift_report"] is None
        assert result.metadata.get("error") == "no_query_embeddings"

    def test_drift_detected_with_shifted_embeddings(self, baseline_path):
        stage = DriftCheckStage(
            baseline_path=baseline_path,
            enabled=True,
            method="mmd_linear",
            threshold=0.05,
        )
        shifted = np.random.randn(20, 3).astype(np.float32) + np.array(
            [5.0, 0.0, 0.0], dtype=np.float32
        )

        class FakeMatchResult:
            embeddings = shifted
            predictions = ["a"] * 20

        context = StageContext(inputs=["x"] * 20)
        context.artifacts["match_result"] = FakeMatchResult()

        result = stage.run(context)
        report = result.artifacts["drift_report"]
        assert report is not None
        assert report.drift_detected
        assert report.global_drift_score > 0.05
        assert result.metadata["recommendation"] == "retrain"

    def test_no_drift_with_similar_embeddings(self, baseline_path):
        stage = DriftCheckStage(
            baseline_path=baseline_path,
            enabled=True,
            method="mmd_linear",
            threshold=0.05,
        )
        similar = np.random.randn(20, 3).astype(np.float32) * 0.01 + np.array(
            [0.5, 0.5, 0.0], dtype=np.float32
        )

        class FakeMatchResult:
            embeddings = similar
            predictions = ["a"] * 10 + ["b"] * 10

        context = StageContext(inputs=["x"] * 20)
        context.artifacts["match_result"] = FakeMatchResult()

        result = stage.run(context)
        report = result.artifacts["drift_report"]
        assert report is not None
        assert not report.drift_detected
        assert result.metadata["recommendation"] == "monitor"
