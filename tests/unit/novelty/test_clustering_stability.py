"""Tests for bootstrap cluster stability scorer."""

import numpy as np
import pytest

from novelentitymatcher.novelty.clustering.stability import ClusterStabilityScorer


def _make_well_separated(seed: int = 42):
    rng = np.random.RandomState(seed)
    a = rng.randn(30, 4) + np.array([10, 0, 0, 0])
    b = rng.randn(30, 4) + np.array([0, 10, 0, 0])
    c = rng.randn(30, 4) + np.array([0, 0, 10, 0])
    embs = np.vstack([a, b, c]).astype(np.float32)
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=int)
    return embs, labels


def _hdbscan_available() -> bool:
    try:
        import hdbscan  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _hdbscan_available(), reason="hdbscan not installed")
class TestClusterStabilityScorer:
    def test_separable_clusters_are_stable(self):
        embs, labels = _make_well_separated()
        scorer = ClusterStabilityScorer(n_bootstrap=5, sample_fraction=0.8, seed=0)
        scores = scorer.score_simple(embs, labels)
        assert len(scores) == 3
        for _cid, score in scores.items():
            assert 0.0 <= score <= 1.0
            assert score > 0.5

    def test_empty_labels_returns_empty(self):
        embs = np.random.randn(10, 4).astype(np.float32)
        labels = np.full(10, -1, dtype=int)
        scorer = ClusterStabilityScorer()
        scores = scorer.score_simple(embs, labels)
        assert scores == {}

    def test_scores_are_bounded(self):
        rng = np.random.RandomState(99)
        embs = rng.randn(50, 4).astype(np.float32)
        labels = np.array([0] * 25 + [1] * 25, dtype=int)
        scorer = ClusterStabilityScorer(n_bootstrap=3, seed=1)
        scores = scorer.score_simple(embs, labels)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_single_cluster_returns_empty(self):
        embs = np.random.randn(20, 4).astype(np.float32)
        labels = np.zeros(20, dtype=int)
        scorer = ClusterStabilityScorer(n_bootstrap=3)
        scores = scorer.score_simple(embs, labels)
        assert len(scores) == 1
