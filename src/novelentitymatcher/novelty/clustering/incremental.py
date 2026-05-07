"""Incremental clustering: assign new points to existing clusters.

Provides online/incremental point assignment without full reclustering,
plus merge detection for clusters that have drifted together.
"""

from __future__ import annotations

import numpy as np

from ...utils.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["IncrementalClusterer", "detect_merges"]


class IncrementalClusterer:
    """Assign new points to existing clusters; create new clusters when needed.

    Strategy:
    1. Compute cosine similarity of each new point to all existing centroids.
    2. Assign to the best-matching centroid if similarity > threshold.
    3. Cluster remaining unassigned points among themselves via HDBSCAN.
    4. Update centroids incrementally.
    """

    def __init__(
        self,
        assignment_threshold: float = 0.7,
        new_cluster_min_size: int = 3,
        metric: str = "cosine",
    ):
        self.assignment_threshold = assignment_threshold
        self.new_cluster_min_size = new_cluster_min_size
        self.metric = metric

        self._centroids: dict[int, np.ndarray] = {}
        self._cluster_sizes: dict[int, int] = {}
        self._next_cluster_id: int = 0

    @property
    def centroids(self) -> dict[int, np.ndarray]:
        return dict(self._centroids)

    @property
    def cluster_sizes(self) -> dict[int, int]:
        return dict(self._cluster_sizes)

    def initialize(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Initialize from an existing clustering result.

        Args:
            embeddings: Embedding matrix (n, dim).
            labels: Cluster labels (-1 for noise).
        """
        X = np.asarray(embeddings, dtype=np.float32)
        unique_clusters = sorted({int(lb) for lb in labels if int(lb) >= 0})

        self._centroids.clear()
        self._cluster_sizes.clear()

        for cid in unique_clusters:
            mask = labels == cid
            self._centroids[cid] = X[mask].mean(axis=0)
            self._cluster_sizes[cid] = int(mask.sum())

        self._next_cluster_id = max(unique_clusters) + 1 if unique_clusters else 0

    def assign(self, new_embeddings: np.ndarray) -> np.ndarray:
        """Assign new points to existing clusters or create new ones.

        Args:
            new_embeddings: New embedding matrix (m, dim).

        Returns:
            Cluster labels for the new points (-1 for noise).
        """
        X = np.asarray(new_embeddings, dtype=np.float32)
        n = X.shape[0]
        labels = np.full(n, -1, dtype=int)

        if not self._centroids:
            self._cluster_new_points(X, labels)
            return labels

        centroid_ids = sorted(self._centroids.keys())
        centroid_matrix = np.array([self._centroids[cid] for cid in centroid_ids])

        if self.metric == "cosine":
            norms_x = np.linalg.norm(X, axis=1, keepdims=True)
            norms_x = np.clip(norms_x, 1e-12, None)
            X_norm = X / norms_x

            norms_c = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
            norms_c = np.clip(norms_c, 1e-12, None)
            C_norm = centroid_matrix / norms_c

            sim = X_norm @ C_norm.T
        else:
            from sklearn.metrics import pairwise_distances

            dist = pairwise_distances(X, centroid_matrix, metric=self.metric)
            sim = 1.0 - dist

        best_sim = np.max(sim, axis=1)
        best_idx = np.argmax(sim, axis=1)

        assigned_mask = best_sim >= self.assignment_threshold
        for i in range(n):
            if assigned_mask[i]:
                cid = centroid_ids[int(best_idx[i])]
                labels[i] = cid
                self._update_centroid(cid, X[i])

        unassigned_mask = ~assigned_mask
        unassigned_indices = np.where(unassigned_mask)[0]
        if len(unassigned_indices) > 0:
            self._cluster_new_points(X[unassigned_indices], labels, unassigned_indices)

        return labels

    def _update_centroid(self, cluster_id: int, new_point: np.ndarray) -> None:
        """Incrementally update a cluster centroid with a new point."""
        old_size = self._cluster_sizes.get(cluster_id, 0)
        old_centroid = self._centroids.get(cluster_id, np.zeros_like(new_point))
        new_size = old_size + 1
        self._centroids[cluster_id] = (old_centroid * old_size + new_point) / new_size
        self._cluster_sizes[cluster_id] = new_size

    def _cluster_new_points(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        index_map: np.ndarray | None = None,
    ) -> None:
        """Cluster unassigned points among themselves via HDBSCAN fallback."""
        n = embeddings.shape[0]
        if n == 0:
            return

        if n == 1:
            cid = self._next_cluster_id
            self._next_cluster_id += 1
            if index_map is not None:
                labels[index_map[0]] = cid
            else:
                labels[0] = cid
            self._centroids[cid] = embeddings[0]
            self._cluster_sizes[cid] = 1
            return

        sub_labels = self._fallback_clustering(embeddings)

        for local_i in range(n):
            cid = int(sub_labels[local_i])
            if cid < 0:
                continue
            global_cid = cid + self._next_cluster_id
            if index_map is not None:
                labels[index_map[local_i]] = global_cid
            else:
                labels[local_i] = global_cid

        unique_sub = sorted({int(lb) for lb in sub_labels if int(lb) >= 0})
        for sub_cid in unique_sub:
            global_cid = sub_cid + self._next_cluster_id
            mask = sub_labels == sub_cid
            self._centroids[global_cid] = embeddings[mask].mean(axis=0)
            self._cluster_sizes[global_cid] = int(mask.sum())

        self._next_cluster_id += max(unique_sub) + 1 if unique_sub else 0

    def _fallback_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Simple connected-components fallback for small unassigned batches."""
        n = embeddings.shape[0]
        if n < self.new_cluster_min_size:
            return np.full(n, -1, dtype=int)

        try:
            from .scalable import ScalableClusterer

            clusterer = ScalableClusterer(
                backend="auto",
                min_cluster_size=max(2, self.new_cluster_min_size),
            )
            labels, _, _ = clusterer.fit_predict(embeddings, metric=self.metric)
            return labels
        except (ImportError, ValueError, RuntimeError):
            pass

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        normed = embeddings / norms
        sim = normed @ normed.T

        labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        for start in range(n):
            if labels[start] != -1:
                continue
            stack = [start]
            component: list[int] = []
            labels[start] = cluster_id
            while stack:
                current = stack.pop()
                component.append(current)
                neighbors = np.where(sim[current] >= self.assignment_threshold)[0]
                for nb in neighbors:
                    if labels[int(nb)] == -1:
                        labels[int(nb)] = cluster_id
                        stack.append(int(nb))
            if len(component) < self.new_cluster_min_size:
                for member in component:
                    labels[member] = -1
            else:
                cluster_id += 1

        return labels


def detect_merges(
    centroids: dict[int, np.ndarray],
    merge_threshold: float = 0.85,
    metric: str = "cosine",
) -> list[tuple[int, int, float]]:
    """Detect cluster pairs that should merge based on centroid similarity.

    Args:
        centroids: Dict mapping cluster_id → centroid vector.
        merge_threshold: Minimum similarity to recommend a merge.
        metric: Similarity metric ('cosine').

    Returns:
        List of (cluster_a, cluster_b, similarity) tuples where
        similarity >= merge_threshold.
    """
    if len(centroids) < 2:
        return []

    cids = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[cid] for cid in cids])

    if metric == "cosine":
        norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        normed = centroid_matrix / norms
        sim = normed @ normed.T
    else:
        from sklearn.metrics import pairwise_distances

        dist = pairwise_distances(centroid_matrix, metric=metric)
        sim = 1.0 - dist

    merges: list[tuple[int, int, float]] = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            s = float(sim[i, j])
            if s >= merge_threshold:
                merges.append((cids[i], cids[j], s))

    merges.sort(key=lambda t: t[2], reverse=True)
    return merges
