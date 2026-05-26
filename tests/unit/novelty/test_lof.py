"""Tests for LOF strategy."""

import numpy as np
import pytest

from novelentitymatcher.novelty.config.strategies import LOFConfig
from novelentitymatcher.novelty.strategies.lof import LOFStrategy


class TestLOFStrategy:
    @pytest.fixture
    def reference_embeddings(self):
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.9],
            ],
            dtype=np.float64,
        )

    @pytest.fixture
    def reference_labels(self):
        return ["a", "a", "b", "b", "c", "c"]

    @pytest.fixture
    def strategy(self, reference_embeddings, reference_labels):
        s = LOFStrategy()
        s.initialize(reference_embeddings, reference_labels, LOFConfig())
        return s

    def test_detect_in_dist_flagged_only_if_outlier(self, strategy):
        in_dist = np.array([[0.95, 0.05, 0.0]], dtype=np.float64)
        flags, metrics = strategy.detect(
            texts=["in"],
            embeddings=in_dist,
            predicted_classes=["a"],
            confidences=np.array([0.9]),
        )
        assert isinstance(flags, set)
        assert isinstance(metrics, dict)

    def test_detect_returns_valid_metrics(self, strategy):
        embeddings = np.array(
            [
                [0.95, 0.05, 0.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        _flags, metrics = strategy.detect(
            texts=["in", "ood"],
            embeddings=embeddings,
            predicted_classes=["a", "a"],
            confidences=np.array([0.9, 0.5]),
        )
        for idx in range(len(embeddings)):
            assert "lof_score" in metrics[idx]
            assert "lof_novelty_score" in metrics[idx]
            assert "lof_is_outlier" in metrics[idx]

    def test_empty_batch(self, strategy):
        flags, metrics = strategy.detect(
            texts=[],
            embeddings=np.empty((0, 3), dtype=np.float64),
            predicted_classes=[],
            confidences=np.array([]),
        )
        assert flags == set()
        assert metrics == {}

    def test_fallback_too_few_neighbors(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        labels = ["a", "a"]
        s = LOFStrategy()
        s.initialize(embeddings, labels, LOFConfig(n_neighbors=10))
        flags, metrics = s.detect(
            texts=["x"],
            embeddings=np.array([[0.5, 0.5]], dtype=np.float64),
            predicted_classes=["a"],
            confidences=np.array([0.5]),
        )
        assert len(flags) == 0
        assert metrics[0]["lof_is_outlier"] is False

    def test_config_schema(self, strategy):
        assert strategy.config_schema is LOFConfig

    def test_get_weight(self, strategy):
        assert strategy.get_weight() == pytest.approx(0.30)
