"""Tests for OODScoreCalibrator."""

import numpy as np
import pytest

from novelentitymatcher.novelty.core.score_calibrator import OODScoreCalibrator


class TestOODScoreCalibrator:
    def test_minmax_output_in_zero_one_range(self):
        """Transformed scores should fall within [0, 1]."""
        calibrator = OODScoreCalibrator(method="minmax")
        raw = np.array([0.1, 0.5, 0.9, 1.2, 2.0])
        calibrator.fit({"knn": raw})
        transformed = calibrator.transform("knn", raw)
        assert np.all(transformed >= 0.0)
        assert np.all(transformed <= 1.0)

    def test_constant_scores_return_zeros(self):
        """Constant scores have zero range, so transform returns zeros."""
        calibrator = OODScoreCalibrator(method="minmax")
        raw = np.array([0.5, 0.5, 0.5])
        calibrator.fit({"knn": raw})
        transformed = calibrator.transform("knn", raw)
        assert np.allclose(transformed, 0.0)

    def test_multiple_strategies_tracked_independently(self):
        """Each strategy gets its own stats."""
        calibrator = OODScoreCalibrator(method="minmax")
        calibrator.fit(
            {
                "knn": np.array([0.0, 1.0]),
                "confidence": np.array([0.2, 0.8]),
            }
        )
        assert calibrator.is_fitted
        assert "knn" in calibrator._stats
        assert "confidence" in calibrator._stats

    def test_transform_unknown_strategy_returns_unchanged(self):
        """Transforming a strategy that was never fitted returns raw scores."""
        calibrator = OODScoreCalibrator(method="minmax")
        raw = np.array([0.1, 0.5])
        calibrator.fit({"knn": raw})
        transformed = calibrator.transform("unknown", np.array([1.0, 2.0]))
        assert np.allclose(transformed, [1.0, 2.0])

    def test_reset_clears_stats(self):
        """Reset should clear fitted stats."""
        calibrator = OODScoreCalibrator(method="minmax")
        calibrator.fit({"knn": np.array([0.0, 1.0])})
        assert calibrator.is_fitted
        calibrator.reset()
        assert not calibrator.is_fitted
        assert calibrator._stats == {}

    def test_percentile_clipping_for_large_samples(self):
        """With >=20 samples, p5/p95 are used instead of min/max."""
        calibrator = OODScoreCalibrator(method="minmax")
        raw = np.arange(100, dtype=float)
        calibrator.fit({"knn": raw})
        stats = calibrator._stats["knn"]
        assert stats["p5"] == pytest.approx(4.95)
        assert stats["p95"] == pytest.approx(94.05)
        # Values outside [p5, p95] should be clipped
        transformed = calibrator.transform("knn", np.array([-10.0, 50.0, 200.0]))
        assert transformed[0] == pytest.approx(0.0)
        assert 0.0 < transformed[1] < 1.0
        assert transformed[2] == pytest.approx(1.0)

    def test_minmax_fallback_for_small_samples(self):
        """With <20 samples, min/max are used as p5/p95 fallback."""
        calibrator = OODScoreCalibrator(method="minmax")
        raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        calibrator.fit({"knn": raw})
        stats = calibrator._stats["knn"]
        assert stats["p5"] == pytest.approx(1.0)
        assert stats["p95"] == pytest.approx(5.0)

    def test_unknown_method_raises(self):
        """An unsupported calibration method should raise ValueError."""
        calibrator = OODScoreCalibrator(method="zscore")
        calibrator.fit({"knn": np.array([0.0, 1.0])})
        with pytest.raises(ValueError, match="Unknown calibration method"):
            calibrator.transform("knn", np.array([0.5]))

    def test_empty_fit_is_noop(self):
        """Fitting with empty dict leaves calibrator unfitted."""
        calibrator = OODScoreCalibrator(method="minmax")
        calibrator.fit({})
        assert not calibrator.is_fitted
