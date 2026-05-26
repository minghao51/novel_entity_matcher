"""Tests for ClusteringStrategy fast centroid-assignment path."""

import numpy as np
import pytest

from novelentitymatcher.novelty.config.strategies import ClusteringConfig
from novelentitymatcher.novelty.strategies.clustering import ClusteringStrategy


def _hdbscan_available() -> bool:
    try:
        import hdbscan  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _hdbscan_available(), reason="hdbscan not installed")
class TestClusteringFastPath:
    @pytest.fixture
    def reference_data(self):
        np.random.seed(42)
        n_per_cluster = 15
        dim = 8
        cluster_a = np.random.randn(n_per_cluster, dim) + np.array(
            [5.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64
        )
        cluster_b = np.random.randn(n_per_cluster, dim) + np.array(
            [0, 5.0, 0, 0, 0, 0, 0, 0], dtype=np.float64
        )
        cluster_c = np.random.randn(n_per_cluster, dim) + np.array(
            [0, 0, 5.0, 0, 0, 0, 0, 0], dtype=np.float64
        )
        embeddings = np.vstack([cluster_a, cluster_b, cluster_c]).astype(np.float64)
        labels = ["a"] * n_per_cluster + ["b"] * n_per_cluster + ["c"] * n_per_cluster
        return embeddings, labels

    @pytest.fixture
    def strategy(self, reference_data):
        embeddings, labels = reference_data
        config = ClusteringConfig(
            min_cluster_size=3,
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
            noise_percentile=90,
        )
        strat = ClusteringStrategy()
        strat.initialize(embeddings, labels, config)
        return strat

    def test_centroids_populated_after_init(self, strategy):
        assert strategy._ref_centroids is not None
        assert len(strategy._ref_centroids) > 0

    def test_fast_path_uses_centroids(self, strategy):
        query = np.random.randn(5, 8) + np.array(
            [5.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64
        )
        flags, metrics = strategy._detect(
            texts=["q1", "q2", "q3", "q4", "q5"],
            embeddings=query,
            predicted_classes=["a"] * 5,
            confidences=np.array([0.9] * 5),
        )
        assert isinstance(flags, set)
        assert isinstance(metrics, dict)
        for idx in range(5):
            assert idx in metrics
            assert "cluster_support_score" in metrics[idx]
            assert "cluster_is_novel" in metrics[idx]

    def test_single_query_embedding(self, strategy):
        query = np.array([[5.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
        _flags, metrics = strategy._detect(
            texts=["q1"],
            embeddings=query,
            predicted_classes=["a"],
            confidences=np.array([0.9]),
        )
        assert 0 in metrics
        assert metrics[0]["cluster_is_novel"] is False

    def test_far_embedding_flagged_as_noise(self, strategy):
        far_point = np.array(
            [[50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]], dtype=np.float64
        )
        flags, metrics = strategy._detect(
            texts=["far"],
            embeddings=far_point,
            predicted_classes=["unknown"],
            confidences=np.array([0.1]),
        )
        assert 0 in flags
        assert metrics[0]["cluster_label"] == -1
        assert metrics[0]["cluster_is_novel"] is True

    def test_no_centroids_falls_back_to_hdbscan(self, reference_data):
        embeddings, labels = reference_data
        config = ClusteringConfig(
            min_cluster_size=3,
            hdbscan_min_cluster_size=100,
            hdbscan_min_samples=100,
        )
        strat = ClusteringStrategy()
        strat.initialize(embeddings, labels, config)
        assert not strat._ref_centroids

        query = embeddings[:3]
        flags, metrics = strat._detect(
            texts=["q1", "q2", "q3"],
            embeddings=query,
            predicted_classes=["a", "a", "a"],
            confidences=np.array([0.9, 0.9, 0.9]),
        )
        assert isinstance(flags, set)
        assert len(metrics) == 3
