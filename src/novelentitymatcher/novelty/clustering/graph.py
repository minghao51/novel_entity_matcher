"""Graph-based community detection backends: Leiden and Louvain.

Builds a k-NN similarity graph from embeddings, then applies graph
community detection algorithms via igraph/leidenalg (Leiden) or
networkx (Louvain).
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ...utils.logging_config import get_logger
from .backends import ClusteringBackendRegistry
from .base import ClusteringBackend

logger = get_logger(__name__)

__all__ = ["LeidenBackend", "LouvainBackend", "SimilarityGraphBuilder"]


class SimilarityGraphBuilder:
    """Build a k-NN similarity graph from embeddings.

    Returns a ``networkx.Graph`` with edge weights equal to cosine
    similarity.  The graph is symmetrised so ``(u, v)`` and ``(v, u)``
    both exist with the same weight.
    """

    def __init__(self, k: int = 15, metric: str = "cosine"):
        self.k = k
        self.metric = metric

    def build(self, embeddings: np.ndarray) -> nx.Graph:
        X = np.asarray(embeddings, dtype=np.float32)
        n = X.shape[0]
        k = min(self.k, n - 1)

        if self.metric == "cosine":
            sim_matrix = cosine_similarity(X)
            np.fill_diagonal(sim_matrix, 0.0)
            rows: list[int] = []
            cols: list[int] = []
            weights: list[float] = []
            for i in range(n):
                top_idx = np.argpartition(sim_matrix[i], -k)[-k:]
                for j in top_idx:
                    if i == j:
                        continue
                    rows.append(int(i))
                    cols.append(int(j))
                    weights.append(float(sim_matrix[i, j]))
            sparse = coo_matrix((weights, (rows, cols)), shape=(n, n))
            sparse = sparse.maximum(sparse.T)
        else:
            from sklearn.metrics import pairwise_distances

            dist = pairwise_distances(X, metric=self.metric)
            np.fill_diagonal(dist, np.inf)
            finite_dist = dist[np.isfinite(dist)]
            scale = float(np.median(finite_dist)) if finite_dist.size else 1.0
            if scale <= 0:
                scale = 1.0
            rows = []
            cols = []
            weights = []
            for i in range(n):
                top_idx = np.argpartition(dist[i], k)[:k]
                for j in top_idx:
                    rows.append(int(i))
                    cols.append(int(j))
                    weights.append(float(np.exp(-dist[i, j] / scale)))
            sparse = coo_matrix((weights, (rows, cols)), shape=(n, n))
            sparse = sparse.maximum(sparse.T)

        G = nx.Graph()
        G.add_nodes_from(range(n))
        cx = sparse.tocoo()
        for i, j, w in zip(
            cx.row.tolist(), cx.col.tolist(), cx.data.tolist(), strict=False
        ):
            if i < j and w > 0:
                G.add_edge(int(i), int(j), weight=w)
        return G


@ClusteringBackendRegistry.register
class LeidenBackend(ClusteringBackend):
    """Leiden community detection via igraph + leidenalg."""

    name = "leiden"

    def __init__(
        self,
        resolution: float = 1.0,
        k: int = 15,
        metric: str = "cosine",
    ):
        self.resolution = resolution
        self.k = k
        self.metric = metric
        self._partition: Any = None

    def fit_predict(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError(
                "python-igraph and leidenalg are required for Leiden clustering. "
                "Install with: pip install python-igraph leidenalg"
            ) from None

        X = np.asarray(embeddings, dtype=np.float32)
        n = X.shape[0]
        if n < 2:
            return (
                np.full(n, -1, dtype=int),
                np.ones(n),
                {
                    "backend": self.name,
                    "n_clusters": 0,
                    "noise_ratio": 1.0,
                    "mean_cluster_size": 0.0,
                },
            )

        k = kwargs.get("k", self.k)
        metric = kwargs.get("metric", self.metric)
        nx_graph = SimilarityGraphBuilder(k=k, metric=metric).build(X)

        ig_graph = ig.Graph(n=n, edges=list(nx_graph.edges()))
        weights = [nx_graph[u][v]["weight"] for u, v in nx_graph.edges()]
        if weights:
            ig_graph.es["weight"] = weights

        self._partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            weights="weight" if weights else None,
            seed=42,
        )
        membership = np.array(self._partition.membership, dtype=int)

        unique_labels, counts = np.unique(membership, return_counts=True)
        small = unique_labels[counts < min_cluster_size]
        for label in small:
            membership[membership == label] = -1

        remap: dict[int, int] = {}
        next_id = 0
        for label in sorted(set(membership.tolist()) - {-1}):
            remap[label] = next_id
            next_id += 1
        labels = np.array([remap.get(int(v), -1) for v in membership], dtype=int)

        n_clusters = len(remap)
        probabilities = np.ones(n)
        for cid in range(n_clusters):
            mask = labels == cid
            if mask.sum() > 1:
                sub = nx_graph.subgraph(np.where(mask)[0].tolist())
                if sub.number_of_edges() > 0:
                    probabilities[mask] = (
                        2.0 * sub.number_of_edges() / (mask.sum() * (mask.sum() - 1))
                    )

        info: dict[str, Any] = {
            "backend": self.name,
            "persistences": np.ones(n_clusters),
            "n_clusters": n_clusters,
            "noise_ratio": float(np.sum(labels == -1)) / n,
            "mean_cluster_size": float(
                np.mean([np.sum(labels == c) for c in range(n_clusters)])
            )
            if n_clusters > 0
            else 0.0,
        }
        return labels, probabilities, info


@ClusteringBackendRegistry.register
class LouvainBackend(ClusteringBackend):
    """Louvain community detection via networkx."""

    name = "louvain"

    def __init__(
        self,
        resolution: float = 1.0,
        k: int = 15,
        metric: str = "cosine",
        threshold: float = 1e-7,
        seed: int = 42,
    ):
        self.resolution = resolution
        self.k = k
        self.metric = metric
        self.threshold = threshold
        self.seed = seed

    def fit_predict(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        X = np.asarray(embeddings, dtype=np.float32)
        n = X.shape[0]
        if n < 2:
            return (
                np.full(n, -1, dtype=int),
                np.ones(n),
                {
                    "backend": self.name,
                    "n_clusters": 0,
                    "noise_ratio": 1.0,
                    "mean_cluster_size": 0.0,
                },
            )

        k = kwargs.get("k", self.k)
        metric = kwargs.get("metric", self.metric)
        nx_graph = SimilarityGraphBuilder(k=k, metric=metric).build(X)

        louvain_kwargs: dict[str, Any] = {
            "weight": "weight",
            "resolution": self.resolution,
            "threshold": self.threshold,
            "seed": self.seed,
        }
        communities = nx.community.louvain_communities(nx_graph, **louvain_kwargs)

        labels = np.full(n, -1, dtype=int)
        community_id = 0
        for community in communities:
            if len(community) < min_cluster_size:
                continue
            for node in community:
                labels[int(node)] = community_id
            community_id += 1

        n_clusters = community_id
        probabilities = np.ones(n)
        for cid in range(n_clusters):
            mask = labels == cid
            if mask.sum() > 1:
                sub = nx_graph.subgraph(np.where(mask)[0].tolist())
                if sub.number_of_edges() > 0:
                    probabilities[mask] = (
                        2.0 * sub.number_of_edges() / (mask.sum() * (mask.sum() - 1))
                    )

        info: dict[str, Any] = {
            "backend": self.name,
            "persistences": np.ones(n_clusters),
            "n_clusters": n_clusters,
            "noise_ratio": float(np.sum(labels == -1)) / n,
            "mean_cluster_size": float(
                np.mean([np.sum(labels == c) for c in range(n_clusters)])
            )
            if n_clusters > 0
            else 0.0,
        }
        return labels, probabilities, info
