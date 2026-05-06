"""Tests for MixtureGaussianStrategy."""

import numpy as np
import pytest

from novelentitymatcher.novelty.config.strategies import MixtureGaussianConfig
from novelentitymatcher.novelty.strategies.mixture_gaussian import (
    MixtureGaussianStrategy,
)


class TestMixtureGaussianStrategy:
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
        s = MixtureGaussianStrategy()
        s.initialize(reference_embeddings, reference_labels, MixtureGaussianConfig())
        return s

    def test_in_distribution_higher_ll_than_ood(self, strategy):
        """In-distribution should have higher log-likelihood than OOD."""
        in_dist = np.array([[0.99, 0.01, 0.0]], dtype=np.float32)
        ood = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

        _, in_m = strategy.detect(
            texts=["in"],
            embeddings=in_dist,
            predicted_classes=["a"],
            confidences=np.array([0.9]),
        )
        _, ood_m = strategy.detect(
            texts=["ood"],
            embeddings=ood,
            predicted_classes=["a"],
            confidences=np.array([0.5]),
        )

        assert in_m[0]["log_likelihood"] > ood_m[0]["log_likelihood"]

    def test_regularization_prevents_singular_covariance(self):
        """With small regularization, identical embeddings should not crash."""
        embs = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
        )
        labels = ["a", "a", "b", "b"]
        s = MixtureGaussianStrategy()
        s.initialize(embs, labels, MixtureGaussianConfig(regularization=1e-4))

        _flags, metrics = s.detect(
            texts=["t"],
            embeddings=np.array([[0.5, 0.5]], dtype=np.float32),
            predicted_classes=["a"],
            confidences=np.array([0.5]),
        )
        assert isinstance(metrics[0]["log_likelihood"], float)

    def test_ood_flagged(self, strategy):
        ood = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        flags, metrics = strategy.detect(
            texts=["ood"],
            embeddings=ood,
            predicted_classes=["a"],
            confidences=np.array([0.5]),
        )
        assert 0 in flags
        assert metrics[0]["log_likelihood"] < metrics[0]["log_likelihood_threshold"]

    def test_config_schema(self, strategy):
        assert strategy.config_schema is MixtureGaussianConfig

    def test_get_weight(self, strategy):
        assert strategy.get_weight() == pytest.approx(0.35)

    def test_empty_batch(self, strategy):
        flags, metrics = strategy.detect(
            texts=[],
            embeddings=np.empty((0, 3), dtype=np.float32),
            predicted_classes=[],
            confidences=np.array([]),
        )
        assert flags == set()
        assert metrics == {}

    def test_log_likelihood_accounts_for_covariance_scale(self):
        embs = np.array(
            [
                [0.0, 0.0],
                [0.01, -0.01],
                [0.0, 5.0],
                [3.0, 8.0],
            ],
            dtype=np.float32,
        )
        labels = ["tight", "tight", "wide", "wide"]
        s = MixtureGaussianStrategy()
        s.initialize(embs, labels, MixtureGaussianConfig(regularization=1e-4))

        probe = np.array([[0.02, 0.0]], dtype=np.float32)
        _flags, metrics = s.detect(
            texts=["probe-tight"],
            embeddings=probe,
            predicted_classes=["tight"],
            confidences=np.array([0.9]),
        )
        tight_ll = metrics[0]["log_likelihood"]

        _flags, metrics = s.detect(
            texts=["probe-wide"],
            embeddings=probe,
            predicted_classes=["wide"],
            confidences=np.array([0.9]),
        )
        wide_ll = metrics[0]["log_likelihood"]

        assert tight_ll > wide_ll
