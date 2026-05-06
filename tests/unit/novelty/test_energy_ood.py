"""Tests for EnergyOODStrategy."""

import numpy as np
import pytest

from novelentitymatcher.novelty.config.strategies import EnergyConfig
from novelentitymatcher.novelty.strategies.energy import EnergyOODStrategy


class TestEnergyOODStrategy:
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
        s = EnergyOODStrategy()
        s.initialize(reference_embeddings, reference_labels, EnergyConfig())
        return s

    def test_in_distribution_has_lower_energy_than_ood(self, strategy):
        """In-distribution samples should have lower (more negative) energy than OOD."""
        in_dist = np.array([[0.99, 0.01, 0.0]], dtype=np.float32)
        ood = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

        _, in_metrics = strategy.detect(
            texts=["in"],
            embeddings=in_dist,
            predicted_classes=["a"],
            confidences=np.array([0.9]),
        )
        _, ood_metrics = strategy.detect(
            texts=["ood"],
            embeddings=ood,
            predicted_classes=["a"],
            confidences=np.array([0.5]),
        )

        assert in_metrics[0]["energy_score"] < ood_metrics[0]["energy_score"]

    def test_ood_samples_flagged(self, strategy):
        """Far-from-distribution samples should be flagged."""
        ood = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        flags, metrics = strategy.detect(
            texts=["ood"],
            embeddings=ood,
            predicted_classes=["a"],
            confidences=np.array([0.5]),
        )
        assert 0 in flags
        assert metrics[0]["energy_score"] > metrics[0]["energy_threshold"]

    def test_config_schema(self, strategy):
        assert strategy.config_schema is EnergyConfig

    def test_get_weight(self, strategy):
        assert strategy.get_weight() == pytest.approx(0.30)

    def test_empty_batch(self, strategy):
        flags, metrics = strategy.detect(
            texts=[],
            embeddings=np.empty((0, 3), dtype=np.float32),
            predicted_classes=[],
            confidences=np.array([]),
        )
        assert flags == set()
        assert metrics == {}

    def test_energy_stable_under_large_scale(
        self, reference_embeddings, reference_labels
    ):
        strategy = EnergyOODStrategy()
        strategy.initialize(
            reference_embeddings,
            reference_labels,
            EnergyConfig(scale=1e4, temperature=0.5),
        )
        _flags, metrics = strategy.detect(
            texts=["x"],
            embeddings=np.array([[0.9, 0.1, 0.0]], dtype=np.float32),
            predicted_classes=["a"],
            confidences=np.array([0.9]),
        )
        assert np.isfinite(metrics[0]["energy_score"])
