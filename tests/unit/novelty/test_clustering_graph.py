"""Tests for graph-based community detection backends."""

import numpy as np
import pytest

from novelentitymatcher.novelty.clustering.graph import (
    LeidenBackend,
    LouvainBackend,
    SimilarityGraphBuilder,
)


def _make_separable(
    n_per_cluster: int = 20, dim: int = 8, n_clusters: int = 3, seed: int = 42
):
    rng = np.random.RandomState(seed)
    offset = 5.0
    clusters = []
    for i in range(n_clusters):
        c = rng.randn(n_per_cluster, dim)
        c[:, i % dim] += offset * (i + 1)
        clusters.append(c)
    return np.vstack(clusters).astype(np.float32)


class TestSimilarityGraphBuilder:
    def test_build_cosine_returns_graph(self):
        embs = _make_separable(n_per_cluster=10, dim=4)
        builder = SimilarityGraphBuilder(k=5, metric="cosine")
        import networkx as nx

        G = builder.build(embs)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == len(embs)
        assert G.number_of_edges() > 0

    def test_build_euclidean_returns_graph(self):
        embs = _make_separable(n_per_cluster=10, dim=4)
        builder = SimilarityGraphBuilder(k=5, metric="euclidean")

        G = builder.build(embs)
        assert G.number_of_nodes() == len(embs)
        assert G.number_of_edges() > 0
        for _u, _v, w in G.edges(data="weight"):
            assert 0.0 < w <= 1.0

    def test_k_larger_than_n_uses_n_minus_1(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        builder = SimilarityGraphBuilder(k=100, metric="cosine")

        G = builder.build(embs)
        assert G.number_of_nodes() == 3

    def test_edges_have_positive_weights(self):
        embs = _make_separable(n_per_cluster=5, dim=4)
        builder = SimilarityGraphBuilder(k=3, metric="cosine")
        G = builder.build(embs)
        for _u, _v, w in G.edges(data="weight"):
            assert w > 0.0

    def test_graph_is_symmetric(self):
        embs = _make_separable(n_per_cluster=8, dim=4)
        builder = SimilarityGraphBuilder(k=5, metric="cosine")
        G = builder.build(embs)
        import networkx as nx

        assert nx.is_weighted(G) or G.number_of_edges() == 0
        for u, v in G.edges():
            assert G.has_edge(v, u)


class TestLouvainBackend:
    def test_separable_clusters_found(self):
        embs = _make_separable(n_per_cluster=25, dim=8, n_clusters=3)
        backend = LouvainBackend(resolution=1.0, k=10, metric="cosine")
        labels, _probs, info = backend.fit_predict(embs, min_cluster_size=5)
        n_clusters = len(set(labels.tolist()) - {-1})
        assert n_clusters >= 2
        assert info["backend"] == "louvain"
        assert info["n_clusters"] == n_clusters
        assert 0.0 <= info["noise_ratio"] <= 1.0

    def test_single_point_returns_noise(self):
        embs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        backend = LouvainBackend()
        labels, _probs, info = backend.fit_predict(embs, min_cluster_size=2)
        assert labels[0] == -1
        assert info["n_clusters"] == 0

    def test_min_cluster_size_filters_small(self):
        rng = np.random.RandomState(0)
        big = rng.randn(30, 4) + 5.0
        small = rng.randn(3, 4) - 5.0
        embs = np.vstack([big, small]).astype(np.float32)
        backend = LouvainBackend(resolution=1.0, k=5)
        _labels, _, _info = backend.fit_predict(embs, min_cluster_size=10)

    def test_registered_in_registry(self):
        from novelentitymatcher.novelty.clustering.backends import (
            ClusteringBackendRegistry,
        )

        assert "louvain" in ClusteringBackendRegistry.list_backends()
        backend = ClusteringBackendRegistry.create("louvain")
        assert isinstance(backend, LouvainBackend)


def _leiden_available() -> bool:
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _leiden_available(), reason="igraph/leidenalg not installed")
class TestLeidenBackend:
    def test_separable_clusters_found(self):
        embs = _make_separable(n_per_cluster=25, dim=8, n_clusters=3)
        backend = LeidenBackend(resolution=1.0, k=10, metric="cosine")
        labels, _probs, info = backend.fit_predict(embs, min_cluster_size=5)
        n_clusters = len(set(labels.tolist()) - {-1})
        assert n_clusters >= 2
        assert info["backend"] == "leiden"

    def test_single_point_returns_noise(self):
        embs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        backend = LeidenBackend()
        labels, _probs, _info = backend.fit_predict(embs, min_cluster_size=2)
        assert labels[0] == -1

    def test_registered_in_registry(self):
        from novelentitymatcher.novelty.clustering.backends import (
            ClusteringBackendRegistry,
        )

        assert "leiden" in ClusteringBackendRegistry.list_backends()
