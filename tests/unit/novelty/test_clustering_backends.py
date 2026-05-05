"""Tests for clustering backend condensed tree and stability extraction."""

import numpy as np
import pytest

from novelentitymatcher.novelty.clustering.backends import (
    ClusteringBackendRegistry,
    HDBSCANBackend,
)
from novelentitymatcher.novelty.clustering.scalable import ScalableClusterer


def _hdbscan_available() -> bool:
    try:
        import hdbscan  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _hdbscan_available(), reason="hdbscan not installed")
class TestHDBSCANCondensedTree:
    @pytest.fixture
    def sample_embeddings(self):
        # Three well-separated clusters
        np.random.seed(42)
        cluster_a = np.random.randn(20, 8) + np.array(
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        cluster_b = np.random.randn(20, 8) + np.array(
            [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        cluster_c = np.random.randn(20, 8) + np.array(
            [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        return np.vstack([cluster_a, cluster_b, cluster_c]).astype(np.float32)

    def test_get_condensed_tree_returns_expected_keys(self, sample_embeddings):
        """The condensed tree should contain structural keys."""
        backend = HDBSCANBackend()
        backend.fit_predict(sample_embeddings, min_cluster_size=5, metric="euclidean")
        tree = backend.get_condensed_tree()
        assert "tree" in tree
        assert "n_points" in tree
        assert "n_clusters" in tree
        assert "persistence" in tree
        assert tree["n_points"] == len(sample_embeddings)
        assert tree["n_clusters"] >= 1

    def test_extract_clusters_at_stability_produces_valid_labels(
        self, sample_embeddings
    ):
        """Re-extraction at a higher threshold should reduce clusters or increase noise."""
        backend = HDBSCANBackend()
        labels_orig, _probs, _info = backend.fit_predict(
            sample_embeddings, min_cluster_size=5, metric="euclidean"
        )
        n_clusters_orig = len(set(labels_orig)) - (1 if -1 in labels_orig else 0)

        labels_new, info = backend.extract_clusters_at_stability(min_persistence=0.5)
        assert labels_new.dtype == labels_orig.dtype
        assert len(labels_new) == len(sample_embeddings)
        assert info["min_persistence"] == 0.5
        assert "n_clusters" in info
        assert "n_noise" in info
        assert "cluster_map" in info

        # Higher threshold should not increase the number of clusters
        n_clusters_new = info["n_clusters"]
        assert n_clusters_new <= n_clusters_orig

    def test_scalable_clusterer_delegates_condensed_tree(self, sample_embeddings):
        """ScalableClusterer should delegate get_condensed_tree to its backend."""
        clusterer = ScalableClusterer(backend="hdbscan", min_cluster_size=5)
        clusterer.fit_predict(sample_embeddings, metric="euclidean")
        tree = clusterer.get_condensed_tree()
        assert "tree" in tree
        assert tree["n_points"] == len(sample_embeddings)

    def test_scalable_clusterer_delegates_stability_extraction(self, sample_embeddings):
        """ScalableClusterer should delegate extract_clusters_at_stability."""
        clusterer = ScalableClusterer(backend="hdbscan", min_cluster_size=5)
        _labels_orig, _probs, _info = clusterer.fit_predict(
            sample_embeddings, metric="euclidean"
        )
        labels_new, info = clusterer.extract_clusters_at_stability(min_persistence=0.3)
        assert len(labels_new) == len(sample_embeddings)
        assert info["n_clusters"] >= 0

    def test_condensed_tree_raises_before_fit(self):
        """Calling get_condensed_tree before fit_predict should raise RuntimeError."""
        backend = HDBSCANBackend()
        with pytest.raises(RuntimeError, match="fit_predict"):
            backend.get_condensed_tree()

    def test_stability_extraction_raises_before_fit(self):
        """Calling extract_clusters_at_stability before fit_predict should raise RuntimeError."""
        backend = HDBSCANBackend()
        with pytest.raises(RuntimeError, match="fit_predict"):
            backend.extract_clusters_at_stability()

    def test_soptics_backend_does_not_support_condensed_tree(self):
        """sOPTICS backend should raise NotImplementedError for condensed tree."""
        try:
            backend = ClusteringBackendRegistry.create("soptics")
        except ValueError:
            pytest.skip("sOPTICS backend not available")
        with pytest.raises(NotImplementedError):
            backend.get_condensed_tree()
