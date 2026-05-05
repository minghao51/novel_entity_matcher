"""Tests for ReActEnergyStrategy."""

import numpy as np
import pytest

from novelentitymatcher.novelty.config.strategies import ReActConfig
from novelentitymatcher.novelty.strategies.react_hybrid import (
    ReActEnergyStrategy,
    trim_activations,
)


class TestTrimActivations:
    def test_clips_above_percentile(self):
        embs = np.array([[0.1, 0.5, 0.9, 1.2]], dtype=np.float32)
        trimmed = trim_activations(embs, percentile=0.75)
        # 75th percentile of [0.1, 0.5, 0.9, 1.2] = 0.975
        # Values above 0.975 should be clipped to 0.975
        threshold = np.percentile(embs, 75)
        assert trimmed[0, 3] == pytest.approx(threshold)
        assert np.all(trimmed <= embs)
        assert np.all(trimmed <= threshold + 1e-6)

    def test_preserves_shape(self):
        embs = np.random.randn(10, 64).astype(np.float32)
        trimmed = trim_activations(embs, percentile=0.9)
        assert trimmed.shape == embs.shape


class TestReActEnergyStrategy:
    @pytest.fixture
    def reference_embeddings(self):
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, 0.95, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.05, 0.95],
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def reference_labels(self):
        return ["a", "a", "b", "b", "c", "c"]

    @pytest.fixture
    def strategy(self, reference_embeddings, reference_labels):
        s = ReActEnergyStrategy()
        s.initialize(reference_embeddings, reference_labels, ReActConfig())
        return s

    def test_react_lowers_extreme_activations(self, strategy):
        ood = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        flags, metrics = strategy.detect(
            texts=["ood"],
            embeddings=ood,
            predicted_classes=["a"],
            confidences=np.array([0.5]),
        )
        assert "react_trim_percentile" in metrics[0]
        assert isinstance(flags, set)

    def test_config_schema(self, strategy):
        assert strategy.config_schema is ReActConfig

    def test_get_weight(self, strategy):
        assert strategy.get_weight() == pytest.approx(0.30)
