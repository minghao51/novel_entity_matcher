"""Tests for SelfKnowledgeStrategy."""

import numpy as np
import pytest

from novelentitymatcher.novelty.config.strategies import SelfKnowledgeConfig
from novelentitymatcher.novelty.strategies.self_knowledge import SelfKnowledgeStrategy


class TestSelfKnowledgeStrategy:
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
            dtype=np.float32,
        )

    @pytest.fixture
    def reference_labels(self):
        return ["a", "a", "b", "b", "c", "c"]

    @pytest.fixture
    def strategy(self, reference_embeddings, reference_labels):
        s = SelfKnowledgeStrategy()
        s.initialize(
            reference_embeddings,
            reference_labels,
            SelfKnowledgeConfig(hidden_dim=8, threshold=0.5),
        )
        return s

    def test_in_dist_not_flagged(self, strategy):
        in_dist = np.array([[0.99, 0.01, 0.0]], dtype=np.float32)
        flags, metrics = strategy.detect(
            texts=["in"],
            embeddings=in_dist,
            predicted_classes=["a"],
            confidences=np.array([0.9]),
        )
        assert isinstance(flags, set)
        assert isinstance(metrics, dict)

    def test_ood_sample_flagged(self, strategy):
        ood = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        flags, metrics = strategy.detect(
            texts=["ood"],
            embeddings=ood,
            predicted_classes=["a"],
            confidences=np.array([0.5]),
        )
        assert isinstance(flags, set)
        assert isinstance(metrics, dict)
        assert "self_knowledge_reconstruction_error" in metrics[0]
        assert "self_knowledge_novelty_score" in metrics[0]

    def test_empty_batch(self, strategy):
        flags, metrics = strategy.detect(
            texts=[],
            embeddings=np.empty((0, 3), dtype=np.float32),
            predicted_classes=[],
            confidences=np.array([]),
        )
        assert flags == set()
        assert metrics == {}

    def test_config_schema(self, strategy):
        assert strategy.config_schema is SelfKnowledgeConfig

    def test_get_weight(self, strategy):
        assert strategy.get_weight() == pytest.approx(0.15)
