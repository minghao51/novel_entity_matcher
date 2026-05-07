"""Tests for MahalanobisDistanceStrategy."""

import numpy as np
import pytest

from novelentitymatcher.novelty.config.strategies import MahalanobisConfig
from novelentitymatcher.novelty.strategies.mahalanobis import (
    MahalanobisDistanceStrategy,
)


class TestMahalanobisDistanceStrategy:
    @pytest.fixture
    def reference_embeddings(self):
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, 0.95, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.95],
            ],
            dtype=np.float64,
        )

    @pytest.fixture
    def reference_labels(self):
        return ["a", "a", "b", "b", "c", "c"]

    @pytest.fixture
    def strategy(self, reference_embeddings, reference_labels):
        s = MahalanobisDistanceStrategy()
        s.initialize(reference_embeddings, reference_labels, MahalanobisConfig())
        return s

    def test_ood_sample_flagged(self, strategy):
        ood = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        flags, metrics = strategy.detect(
            texts=["ood"],
            embeddings=ood,
            predicted_classes=["a"],
            confidences=np.array([0.5]),
        )
        assert isinstance(flags, set)
        assert isinstance(metrics, dict)
        assert "mahalanobis_distance" in metrics[0]
        assert "mahalanobis_novelty_score" in metrics[0]

    def test_in_dist_not_flagged(self, strategy):
        in_dist = np.array([[0.99, 0.01, 0.0]], dtype=np.float64)
        flags, metrics = strategy.detect(
            texts=["in"],
            embeddings=in_dist,
            predicted_classes=["a"],
            confidences=np.array([0.9]),
        )
        assert 0 not in flags or metrics[0]["mahalanobis_distance"] < 3.0

    def test_empty_batch(self, strategy):
        flags, metrics = strategy.detect(
            texts=[],
            embeddings=np.empty((0, 3), dtype=np.float64),
            predicted_classes=[],
            confidences=np.array([]),
        )
        assert flags == set()
        assert metrics == {}

    def test_config_schema(self, strategy):
        assert strategy.config_schema is MahalanobisConfig

    def test_get_weight(self, strategy):
        assert strategy.get_weight() == pytest.approx(0.35)

    def test_unseen_class_uses_global_mean(self, strategy):
        unseen = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        _flags, metrics = strategy.detect(
            texts=["unseen"],
            embeddings=unseen,
            predicted_classes=["unknown_class"],
            confidences=np.array([0.3]),
        )
        assert "mahalanobis_distance" in metrics[0]
        assert np.isfinite(metrics[0]["mahalanobis_distance"])
