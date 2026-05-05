"""Tests for DriftScorer."""

import numpy as np
import pytest

from novelentitymatcher.novelty.drift.scorer import DriftScorer


class TestDriftScorer:
    @pytest.fixture
    def baseline_mean(self):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    @pytest.fixture
    def baseline_cov(self):
        return np.eye(3, dtype=np.float32) * 0.01

    def test_identical_distribution_score_near_zero(self, baseline_mean, baseline_cov):
        scorer = DriftScorer(method="mmd_linear", threshold=0.05)
        current = np.random.randn(100, 3).astype(np.float32) * 0.01 + baseline_mean
        report = scorer.score(current, baseline_mean, baseline_cov)
        assert report.global_drift_score < 0.05
        assert not report.drift_detected
        assert report.method == "mmd_linear"

    def test_shifted_distribution_detected(self, baseline_mean, baseline_cov):
        scorer = DriftScorer(method="mmd_linear", threshold=0.05)
        shifted = np.random.randn(100, 3).astype(np.float32) * 0.01 + np.array(
            [5.0, 0.0, 0.0], dtype=np.float32
        )
        report = scorer.score(shifted, baseline_mean, baseline_cov)
        assert report.drift_detected
        assert report.global_drift_score > 0.05

    def test_mmd_symmetry(self):
        """MMD with linear kernel is symmetric."""
        a = np.random.randn(50, 4).astype(np.float32)
        b = np.random.randn(50, 4).astype(np.float32)
        mean_a = a.mean(axis=0)
        mean_b = b.mean(axis=0)
        scorer = DriftScorer(method="mmd_linear")
        report_ab = scorer.score(b, mean_a)
        report_ba = scorer.score(a, mean_b)
        assert report_ab.global_drift_score == pytest.approx(
            report_ba.global_drift_score, abs=1e-6
        )

    def test_cosine_centroid_range(self, baseline_mean):
        scorer = DriftScorer(method="cosine_centroid", threshold=0.05)
        current = np.random.randn(50, 3).astype(np.float32)
        report = scorer.score(current, baseline_mean)
        assert 0.0 <= report.global_drift_score <= 2.0

    def test_kl_gaussian_positive(self, baseline_mean, baseline_cov):
        scorer = DriftScorer(method="kl_gaussian", threshold=0.05)
        current = np.random.randn(50, 3).astype(np.float32) * 0.02 + baseline_mean
        report = scorer.score(current, baseline_mean, baseline_cov)
        assert report.global_drift_score >= 0.0

    def test_per_class_drift(self, baseline_mean, baseline_cov):
        scorer = DriftScorer(method="mmd_linear", threshold=0.05)
        current = np.random.randn(50, 3).astype(np.float32) * 0.01 + baseline_mean
        per_class = {
            "a": {"mean": baseline_mean, "cov": baseline_cov},
            "b": {"mean": baseline_mean, "cov": baseline_cov},
        }
        labels = ["a"] * 25 + ["b"] * 25
        report = scorer.score(
            current,
            baseline_mean,
            baseline_cov,
            per_class_baselines=per_class,
            current_labels=labels,
        )
        assert "a" in report.per_class_drift
        assert "b" in report.per_class_drift
        assert all(0.0 <= v <= 2.0 for v in report.per_class_drift.values())

    def test_recommendation_monitor_when_no_drift(self, baseline_mean):
        scorer = DriftScorer(method="mmd_linear", threshold=0.05)
        report = scorer.score(baseline_mean.reshape(1, -1), baseline_mean)
        assert report.recommendation == "monitor"

    def test_recommendation_retrain_for_severe_drift(self, baseline_mean):
        scorer = DriftScorer(method="mmd_linear", threshold=0.01)
        shifted = np.random.randn(100, 3).astype(np.float32) + np.array(
            [10.0, 0.0, 0.0], dtype=np.float32
        )
        report = scorer.score(shifted, baseline_mean)
        assert report.recommendation == "retrain"

    def test_unknown_method_raises(self):
        scorer = DriftScorer(method="unknown")
        with pytest.raises(ValueError, match="Unknown drift method"):
            scorer.score(np.zeros((5, 3)), np.zeros(3))

    def test_kl_requires_cov(self):
        scorer = DriftScorer(method="kl_gaussian")
        with pytest.raises(ValueError, match="requires baseline_cov"):
            scorer.score(np.zeros((5, 3)), np.zeros(3))
