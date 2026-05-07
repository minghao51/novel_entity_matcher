"""Tests for UncertaintySampler."""

import numpy as np
import pytest

from novelentitymatcher.novelty.active_learning.sampler import UncertaintySampler

TEXTS = ["cat", "dog", "fish", "bird", "lizard"]
CLASSES = ["mammal", "mammal", "fish", "bird", "reptile"]
CONFIDENCES = [0.95, 0.50, 0.80, 0.60, 0.99]


class TestUncertaintySampler:
    def test_least_confident_returns_most_uncertain(self):
        sampler = UncertaintySampler(strategy="least_confident")
        result = sampler.sample(TEXTS, CONFIDENCES, CLASSES, n_samples=2)
        assert len(result) == 2
        assert result[0]["text"] == "dog"
        assert result[1]["text"] == "bird"

    def test_entropy_requires_probabilities(self):
        sampler = UncertaintySampler(strategy="entropy")
        result = sampler.sample(TEXTS, CONFIDENCES, CLASSES, n_samples=2)
        assert len(result) == 2

    def test_entropy_with_probabilities(self):
        sampler = UncertaintySampler(strategy="entropy")
        probs = np.array(
            [
                [0.95, 0.03, 0.01, 0.01],
                [0.50, 0.40, 0.05, 0.05],
                [0.80, 0.10, 0.05, 0.05],
                [0.60, 0.30, 0.05, 0.05],
                [0.99, 0.005, 0.003, 0.002],
            ]
        )
        result = sampler.sample(
            TEXTS, CONFIDENCES, CLASSES, probabilities=probs, n_samples=2
        )
        assert len(result) == 2
        assert result[0]["text"] == "dog"

    def test_margin_returns_low_margin_samples(self):
        sampler = UncertaintySampler(strategy="margin")
        probs = np.array(
            [
                [0.95, 0.03, 0.01, 0.01],
                [0.50, 0.40, 0.05, 0.05],
                [0.80, 0.10, 0.05, 0.05],
                [0.60, 0.30, 0.05, 0.05],
                [0.99, 0.005, 0.003, 0.002],
            ]
        )
        result = sampler.sample(
            TEXTS, CONFIDENCES, CLASSES, probabilities=probs, n_samples=3
        )
        assert len(result) == 3
        assert result[0]["text"] == "dog"

    def test_n_samples_zero_returns_empty(self):
        sampler = UncertaintySampler()
        result = sampler.sample(TEXTS, CONFIDENCES, CLASSES, n_samples=0)
        assert result == []

    def test_n_samples_exceeds_length(self):
        sampler = UncertaintySampler(strategy="least_confident")
        result = sampler.sample(TEXTS[:2], CONFIDENCES[:2], CLASSES[:2], n_samples=10)
        assert len(result) == 2

    def test_empty_inputs(self):
        sampler = UncertaintySampler()
        result = sampler.sample([], [], [], n_samples=5)
        assert result == []

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            UncertaintySampler(strategy="invalid")

    def test_output_keys(self):
        sampler = UncertaintySampler(strategy="least_confident")
        result = sampler.sample(TEXTS, CONFIDENCES, CLASSES, n_samples=1)
        assert "text" in result[0]
        assert "confidence" in result[0]
        assert "predicted_class" in result[0]
        assert "uncertainty_score" in result[0]
        assert "strategy" in result[0]
