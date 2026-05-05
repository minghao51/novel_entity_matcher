"""Tests for DistributionSnapshot."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from novelentitymatcher.novelty.drift.snapshot import DistributionSnapshot


class TestDistributionSnapshot:
    @pytest.fixture
    def embeddings(self):
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def labels(self):
        return ["a", "a", "b", "b"]

    def test_from_embeddings_computes_mean_and_cov(self, embeddings, labels):
        snap = DistributionSnapshot.from_embeddings(embeddings, labels)
        assert snap.n_points == 4
        assert snap.mean.shape == (3,)
        assert snap.covariance.shape == (3, 3)
        assert set(snap.per_class_stats.keys()) == {"a", "b"}
        assert snap.per_class_stats["a"]["count"] == 2

    def test_equality_for_identical_inputs(self, embeddings, labels):
        snap1 = DistributionSnapshot.from_embeddings(embeddings, labels)
        snap2 = DistributionSnapshot.from_embeddings(embeddings, labels)
        assert snap1 == snap2

    def test_inequality_for_perturbed_inputs(self, embeddings, labels):
        snap1 = DistributionSnapshot.from_embeddings(embeddings, labels)
        perturbed = embeddings + np.random.randn(*embeddings.shape) * 0.01
        snap2 = DistributionSnapshot.from_embeddings(perturbed, labels)
        assert snap1 != snap2

    def test_save_load_roundtrip(self, embeddings, labels):
        snap = DistributionSnapshot.from_embeddings(embeddings, labels)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot"
            snap.save(path)
            loaded = DistributionSnapshot.load(path)

        assert loaded.n_points == snap.n_points
        assert loaded.embedding_hash == snap.embedding_hash
        assert np.allclose(loaded.mean, snap.mean)
        assert np.allclose(loaded.covariance, snap.covariance)
        assert set(loaded.per_class_stats.keys()) == set(snap.per_class_stats.keys())

    def test_1d_embedding(self):
        embs = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        labels = ["a", "a", "b", "b"]
        snap = DistributionSnapshot.from_embeddings(embs, labels)
        assert snap.covariance.shape == (1, 1)
        assert snap.mean.shape == (1,)
