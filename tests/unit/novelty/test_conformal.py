"""Tests for ConformalCalibrator."""

import numpy as np
import pytest

from novelentitymatcher.novelty.config.strategies import MahalanobisConfig
from novelentitymatcher.novelty.strategies.mahalanobis import (
    MahalanobisDistanceStrategy,
)
from novelentitymatcher.novelty.strategies.conformal import ConformalCalibrator


class TestConformalCalibrator:
    """Tests for split and mondrian conformal calibration."""

    @pytest.fixture
    def calibration_scores(self):
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    @pytest.fixture
    def calibration_labels(self):
        return np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])

    @pytest.fixture
    def test_scores(self):
        return np.array([0.15, 0.55, 1.5])

    @pytest.fixture
    def test_predictions(self):
        return np.array(["A", "B", "C"])

    class TestSplitMethod:
        """Tests for split (global) conformal calibration."""

        def test_calibrate_split(self, calibration_scores):
            calibrator = ConformalCalibrator(alpha=0.1, method="split")
            calibrator.calibrate(calibration_scores, np.array(["label"] * len(calibration_scores)))

            assert calibrator.is_calibrated
            assert calibrator._n_calibration == len(calibration_scores)

        def test_predict_pvalues_split(self, calibration_scores):
            calibrator = ConformalCalibrator(alpha=0.1, method="split")
            calibrator.calibrate(calibration_scores, np.array(["label"] * len(calibration_scores)))

            test = np.array([0.15, 0.5, 1.5])
            pvals = calibrator.predict_pvalues(test)

            assert len(pvals) == 3
            assert 0.0 <= pvals[0] <= 1.0
            assert pvals[0] > pvals[1]
            assert pvals[1] > pvals[2]
            assert all(pvals >= 0.0)

        def test_low_score_high_pvalue(self, calibration_scores):
            calibrator = ConformalCalibrator(alpha=0.1, method="split")
            calibrator.calibrate(calibration_scores, np.array(["label"] * len(calibration_scores)))

            low_score = np.array([0.05])
            pval = calibrator.predict_pvalues(low_score)[0]
            assert pval > 0.5

        def test_high_score_low_pvalue(self, calibration_scores):
            calibrator = ConformalCalibrator(alpha=0.1, method="split")
            calibrator.calibrate(calibration_scores, np.array(["label"] * len(calibration_scores)))

            high_score = np.array([2.0])
            pval = calibrator.predict_pvalues(high_score)[0]
            assert pval < 0.1

        def test_predict_pvalues_split_counts_ties_as_nonconforming(self):
            calibrator = ConformalCalibrator(alpha=0.1, method="split")
            calibrator.calibrate(
                np.array([0.1, 0.2, 0.2]),
                np.array(["label", "label", "label"]),
            )

            pval = calibrator.predict_pvalues(np.array([0.2]))[0]

            assert pval == pytest.approx(0.75)

    class TestMondrianMethod:
        """Tests for Mondrian (class-conditional) conformal calibration."""

        def test_calibrate_mondrian(self, calibration_scores, calibration_labels):
            calibrator = ConformalCalibrator(alpha=0.1, method="mondrian")
            calibrator.calibrate(calibration_scores, calibration_labels)

            assert calibrator.is_calibrated
            assert len(calibrator._class_scores) == 2
            assert "A" in calibrator._class_scores
            assert "B" in calibrator._class_scores
            assert len(calibrator._class_scores["A"]) == 5
            assert len(calibrator._class_scores["B"]) == 5

        def test_predict_pvalues_for_class_known_class(self, calibration_scores, calibration_labels):
            calibrator = ConformalCalibrator(alpha=0.1, method="mondrian")
            calibrator.calibrate(calibration_scores, calibration_labels)

            scores = np.array([0.15, 0.55])
            preds = np.array(["A", "B"])
            pvals = calibrator.predict_pvalues_for_class(scores, preds)

            assert len(pvals) == 2
            assert 0.0 <= pvals[0] <= 1.0
            assert 0.0 <= pvals[1] <= 1.0

        def test_predict_pvalues_for_class_unseen_class_fallback(
            self, calibration_scores, calibration_labels
        ):
            calibrator = ConformalCalibrator(alpha=0.1, method="mondrian")
            calibrator.calibrate(calibration_scores, calibration_labels)

            scores = np.array([0.5])
            preds = np.array(["C"])
            pvals = calibrator.predict_pvalues_for_class(scores, preds)

            assert len(pvals) == 1
            assert 0.0 <= pvals[0] <= 1.0

        def test_mondrian_pvalue_lower_for_same_relative_rank_in_class(
            self, calibration_scores, calibration_labels
        ):
            calibrator = ConformalCalibrator(alpha=0.1, method="mondrian")
            calibrator.calibrate(calibration_scores, calibration_labels)

            scores_a = np.array([0.05, 0.25, 0.95])
            preds_a = np.array(["A", "A", "A"])
            pvals_a = calibrator.predict_pvalues_for_class(scores_a, preds_a)

            assert pvals_a[0] > pvals_a[1]
            assert pvals_a[1] > pvals_a[2]

        def test_split_vs_mondrian_pvalues_differ(
            self, calibration_scores, calibration_labels
        ):
            split_calibrator = ConformalCalibrator(alpha=0.1, method="split")
            split_calibrator.calibrate(calibration_scores, calibration_labels)

            mondrian_calibrator = ConformalCalibrator(alpha=0.1, method="mondrian")
            mondrian_calibrator.calibrate(calibration_scores, calibration_labels)

            scores = np.array([0.25, 0.55])
            preds = np.array(["A", "B"])

            split_pvals = split_calibrator.predict_pvalues(scores)
            mondrian_pvals = mondrian_calibrator.predict_pvalues_for_class(scores, preds)

            assert not np.allclose(split_pvals, mondrian_pvals)

    class TestCalibrationMetadata:
        """Tests for calibration metadata and properties."""

        def test_not_calibrated_before_calibrate(self):
            calibrator = ConformalCalibrator()
            assert not calibrator.is_calibrated

        def test_calibration_metadata_after_calibrate(self, calibration_scores, calibration_labels):
            calibrator = ConformalCalibrator(alpha=0.05, method="mondrian")
            calibrator.calibrate(calibration_scores, calibration_labels)

            meta = calibrator.calibration_metadata
            assert meta["alpha"] == 0.05
            assert meta["method"] == "mondrian"
            assert meta["n_calibration"] == 10
            assert meta["is_calibrated"] is True
            assert meta["n_classes"] == 2

        def test_uncalibrated_raises_on_predict(self):
            calibrator = ConformalCalibrator()
            with pytest.raises(RuntimeError, match="not been calibrated"):
                calibrator.predict_pvalues(np.array([0.5]))

    class TestEdgeCases:
        """Tests for edge cases and error handling."""

        def test_empty_calibration_scores(self):
            calibrator = ConformalCalibrator(alpha=0.1, method="split")
            calibrator.calibrate(np.array([]), np.array([]))

            assert calibrator.is_calibrated
            pvals = calibrator.predict_pvalues(np.array([0.5]))
            assert len(pvals) == 1

        def test_single_class_mondrian(self):
            calibrator = ConformalCalibrator(alpha=0.1, method="mondrian")
            calibrator.calibrate(
                np.array([0.1, 0.2, 0.3]),
                np.array(["A", "A", "A"]),
            )

            assert len(calibrator._class_scores) == 1
            pvals = calibrator.predict_pvalues_for_class(
                np.array([0.15]),
                np.array(["A"]),
            )
            assert pvals[0] > 0.0

        def test_calibrator_chain(self):
            calibrator = ConformalCalibrator(alpha=0.1, method="split")
            result = calibrator.calibrate(np.array([0.1, 0.2, 0.3]), np.array(["x", "x", "x"]))

            assert result is calibrator
            assert calibrator.is_calibrated

        def test_mahalanobis_conformal_falls_back_when_core_split_would_be_empty(self):
            strategy = MahalanobisDistanceStrategy()
            strategy.initialize(
                np.array([[1.0, 0.0]], dtype=float),
                ["A"],
                MahalanobisConfig(calibration_mode="conformal"),
            )

            flags, metrics = strategy.detect(
                ["sample"],
                np.array([[1.0, 0.0]], dtype=float),
                ["A"],
                np.array([0.9]),
            )

            assert strategy._calibrator is None
            assert flags == set()
            assert np.isfinite(metrics[0]["mahalanobis_distance"])
            assert np.isfinite(metrics[0]["mahalanobis_novelty_score"])
