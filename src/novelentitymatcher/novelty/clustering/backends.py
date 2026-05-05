"""Concrete clustering backend implementations and registry."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.metrics import pairwise_distances

from ...utils.logging_config import get_logger
from .base import ClusteringBackend

logger = get_logger(__name__)


class ClusteringBackendRegistry:
    """Registry for clustering backends."""

    _backends: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, backend_cls: type) -> type:
        cls._backends[getattr(backend_cls, "name", backend_cls.__name__)] = backend_cls
        return backend_cls

    @classmethod
    def get(cls, name: str) -> type:
        if name == "auto":
            return cls._backends["hdbscan"]
        return cls._backends[name]

    @classmethod
    def list_backends(cls) -> list[str]:
        return list(cls._backends.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> ClusteringBackend:
        return cls.get(name)(**kwargs)


@ClusteringBackendRegistry.register
class HDBSCANBackend(ClusteringBackend):
    """HDBSCAN clustering backend."""

    name = "hdbscan"

    def __init__(
        self,
        min_samples: int = 5,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "cosine",
        prediction_data: bool = True,
    ):
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.prediction_data = prediction_data
        self._clusterer: Any = None

    def fit_predict(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is required for HDBSCAN clustering. "
                "Install with: pip install hdbscan"
            ) from None

        distance_matrix = self._compute_distances(embeddings)

        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric="precomputed",
            prediction_data=self.prediction_data,
        )
        labels = self._clusterer.fit_predict(distance_matrix.astype(np.float64))
        probabilities = getattr(self._clusterer, "probabilities_", np.ones(len(labels)))
        persistences = getattr(self._clusterer, "cluster_persistence_", [])

        info: dict[str, Any] = {
            "backend": self.name,
            "persistences": persistences,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "noise_ratio": float(np.sum(labels == -1)) / len(labels),
            "mean_cluster_size": float(
                np.mean([np.sum(labels == c) for c in set(labels) if c >= 0])
            )
            if any(labels >= 0)
            else 0.0,
        }
        return labels, probabilities, info

    def get_condensed_tree(self) -> dict[str, Any]:
        if self._clusterer is None:
            raise RuntimeError("Must call fit_predict first")
        tree_data = self._clusterer.condensed_tree_.to_numpy()
        return {
            "tree": tree_data.tolist(),
            "n_points": len(self._clusterer.labels_),
            "n_clusters": len(set(self._clusterer.labels_))
            - (1 if -1 in self._clusterer.labels_ else 0),
            "persistence": getattr(self._clusterer, "cluster_persistence_", []),
        }

    def extract_clusters_at_stability(
        self, min_persistence: float = 0.1
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._clusterer is None:
            raise RuntimeError("Must call fit_predict first")
        labels = self._clusterer.labels_
        cluster_persistence = getattr(self._clusterer, "cluster_persistence_", [])
        stable_clusters = {
            i for i, p in enumerate(cluster_persistence) if p >= min_persistence
        }
        new_labels = np.full_like(labels, -1)
        cluster_map: dict[int, int] = {}
        next_id = 0
        for orig_label in sorted(stable_clusters):
            mask = labels == orig_label
            new_labels[mask] = next_id
            cluster_map[orig_label] = next_id
            next_id += 1
        info: dict[str, Any] = {
            "min_persistence": min_persistence,
            "n_clusters": len(stable_clusters),
            "n_noise": int(np.sum(new_labels == -1)),
            "cluster_map": cluster_map,
        }
        return new_labels, info

    def _compute_distances(self, embeddings: np.ndarray) -> np.ndarray:
        if self.metric == "precomputed":
            return embeddings
        return pairwise_distances(embeddings, metric=self.metric).astype(np.float32)


@ClusteringBackendRegistry.register
class SOPTICSBackend(ClusteringBackend):
    """sOPTICS (LSH-accelerated OPTICS) clustering backend."""

    name = "soptics"

    def __init__(
        self,
        min_samples: int = 5,
        metric: str = "cosine",
    ):
        self.min_samples = min_samples
        self.metric = metric

    def fit_predict(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        distance_matrix = self._compute_distances(embeddings)
        n = distance_matrix.shape[0]

        core_distances = np.zeros(n)
        for i in range(n):
            row = distance_matrix[i]
            sorted_dists = np.partition(row, self.min_samples - 1)
            if self.min_samples < len(sorted_dists):
                core_distances[i] = sorted_dists[self.min_samples - 1]
            else:
                core_distances[i] = sorted_dists[-1]

        reachability = (
            np.maximum(core_distances[:, np.newaxis], core_distances[np.newaxis, :])
            - distance_matrix
        )
        reachability = np.maximum(reachability, 0)
        np.fill_diagonal(reachability, 0)

        avg_reachability = np.mean(reachability[np.triu_indices(n, k=1)])
        std_reachability = np.std(reachability[np.triu_indices(n, k=1)])

        threshold = avg_reachability + 0.5 * std_reachability
        is_core = core_distances <= threshold

        labels = np.full(n, -1, dtype=int)
        cluster_id = 0

        for i in range(n):
            if not is_core[i] or labels[i] != -1:
                continue

            queue = [i]
            labels[i] = cluster_id

            while queue:
                current = queue.pop(0)
                if not is_core[current]:
                    continue

                for j in range(n):
                    if labels[j] == -1 and reachability[current, j] <= threshold:
                        labels[j] = cluster_id
                        queue.append(j)

            cluster_id += 1

        noise_mask = labels == -1
        if np.sum(noise_mask) > 0 and min_cluster_size <= 3:
            small_clusters = []
            for i in range(n):
                if labels[i] == -1:
                    neighbors = np.where(distance_matrix[i] <= threshold)[0]
                    if len(neighbors) >= min_cluster_size:
                        small_clusters.append(i)

            for idx in small_clusters:
                labels[idx] = cluster_id
                cluster_id += 1

        noise_mask = labels == -1
        labels[noise_mask] = -1

        probabilities = np.ones(n)
        for i in range(n):
            if labels[i] >= 0:
                cluster_members = np.where(labels == labels[i])[0]
                if len(cluster_members) > 1:
                    intra_dist = distance_matrix[i][cluster_members]
                    probabilities[i] = (
                        1.0 / (1.0 + np.mean(intra_dist[intra_dist > 0]))
                        if np.any(intra_dist > 0)
                        else 1.0
                    )

        logger.info(
            f"sOPTICS: found {cluster_id} clusters, {np.sum(labels == -1)} noise points"
        )

        info: dict[str, Any] = {
            "backend": self.name,
            "persistences": np.ones(cluster_id),
            "n_clusters": cluster_id,
            "noise_ratio": float(np.sum(labels == -1)) / len(labels),
            "mean_cluster_size": float(
                np.mean([np.sum(labels == c) for c in range(cluster_id)])
            )
            if cluster_id > 0
            else 0.0,
        }
        return labels, probabilities, info

    def _compute_distances(self, embeddings: np.ndarray) -> np.ndarray:
        if self.metric == "precomputed":
            return embeddings
        return pairwise_distances(embeddings, metric=self.metric).astype(np.float32)


@ClusteringBackendRegistry.register
class UMAPHDBSCANBackend(ClusteringBackend):
    """UMAP preprocessing followed by HDBSCAN clustering backend."""

    name = "umap_hdbscan"

    def __init__(
        self,
        min_samples: int = 5,
        cluster_selection_epsilon: float = 0.0,
        n_neighbors: int = 15,
        umap_dim: int = 10,
        umap_metric: str = "cosine",
        prediction_data: bool = True,
    ):
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.n_neighbors = n_neighbors
        self.umap_dim = umap_dim
        self.umap_metric = umap_metric
        self.prediction_data = prediction_data
        self._umap_model: Any = None
        self._clusterer: Any = None

    def fit_predict(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP preprocessing. "
                "Install with: pip install umap-learn"
            ) from None

        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is required for HDBSCAN clustering. "
                "Install with: pip install hdbscan"
            ) from None

        logger.info(
            f"Applying UMAP reduction: {embeddings.shape} -> ({embeddings.shape[0]}, {self.umap_dim})"
        )

        self._umap_model = umap.UMAP(
            n_components=self.umap_dim,
            n_neighbors=self.n_neighbors,
            metric=self.umap_metric,
            min_dist=0.0,
            random_state=42,
        )
        reduced = self._umap_model.fit_transform(embeddings)

        distance_matrix = pairwise_distances(reduced, metric="euclidean").astype(
            np.float32
        )

        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric="precomputed",
            prediction_data=self.prediction_data,
        )
        labels = self._clusterer.fit_predict(distance_matrix.astype(np.float64))
        probabilities = getattr(self._clusterer, "probabilities_", np.ones(len(labels)))
        persistences = getattr(self._clusterer, "cluster_persistence_", [])

        info: dict[str, Any] = {
            "backend": self.name,
            "persistences": persistences,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "noise_ratio": float(np.sum(labels == -1)) / len(labels),
            "mean_cluster_size": float(
                np.mean([np.sum(labels == c) for c in set(labels) if c >= 0])
            )
            if any(labels >= 0)
            else 0.0,
        }
        return labels, probabilities, info
