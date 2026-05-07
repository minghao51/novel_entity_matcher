"""Tests for incremental clustering and merge detection."""

import numpy as np

from novelentitymatcher.novelty.clustering.incremental import (
    IncrementalClusterer,
    detect_merges,
)


def _make_clusters(seed: int = 42):
    rng = np.random.RandomState(seed)
    a = rng.randn(20, 4) + np.array([10, 0, 0, 0])
    b = rng.randn(20, 4) + np.array([0, 10, 0, 0])
    c = rng.randn(20, 4) + np.array([0, 0, 10, 0])
    return np.vstack([a, b, c]).astype(np.float32), np.array(
        [0] * 20 + [1] * 20 + [2] * 20, dtype=int
    )


class TestIncrementalClusterer:
    def test_initialize_sets_centroids(self):
        embs, labels = _make_clusters()
        ic = IncrementalClusterer()
        ic.initialize(embs, labels)
        assert len(ic.centroids) == 3
        assert set(ic.cluster_sizes.values()) == {20}

    def test_assign_existing_points_to_correct_cluster(self):
        embs, labels = _make_clusters()
        ic = IncrementalClusterer(assignment_threshold=0.5)
        ic.initialize(embs, labels)

        rng = np.random.RandomState(7)
        new_a = rng.randn(5, 4) + np.array([10, 0, 0, 0])
        new_labels = ic.assign(new_a.astype(np.float32))
        assert all(label == 0 for label in new_labels)

    def test_novel_points_create_new_clusters(self):
        embs, labels = _make_clusters()
        ic = IncrementalClusterer(assignment_threshold=0.95, new_cluster_min_size=2)
        ic.initialize(embs, labels)

        rng = np.random.RandomState(3)
        novel = rng.randn(10, 4) + np.array([0, 0, 0, 50])
        new_labels = ic.assign(novel.astype(np.float32))
        non_noise = [label for label in new_labels if label >= 0]
        assert len(non_noise) >= 2
        assert any(label >= 3 for label in new_labels)

    def test_centroids_update_after_assignment(self):
        embs, labels = _make_clusters()
        ic = IncrementalClusterer(assignment_threshold=0.3)
        ic.initialize(embs, labels)

        old_centroid_0 = ic.centroids[0].copy()
        rng = np.random.RandomState(5)
        new_pt = rng.randn(1, 4) + np.array([10, 0, 0, 0])
        ic.assign(new_pt.astype(np.float32))
        assert not np.allclose(ic.centroids[0], old_centroid_0)
        assert ic.cluster_sizes[0] == 21

    def test_empty_centroids_triggers_new_clustering(self):
        ic = IncrementalClusterer()
        rng = np.random.RandomState(0)
        embs = rng.randn(20, 4).astype(np.float32)
        labels = ic.assign(embs)
        non_noise = [label for label in labels if label >= 0]
        assert len(non_noise) > 0


class TestDetectMerges:
    def test_identical_centroids_produce_merge(self):
        c = np.array([1.0, 2.0, 3.0])
        centroids = {0: c, 1: c.copy()}
        merges = detect_merges(centroids, merge_threshold=0.9)
        assert len(merges) == 1
        assert merges[0][2] >= 0.99

    def test_distant_centroids_no_merge(self):
        centroids = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0]),
        }
        merges = detect_merges(centroids, merge_threshold=0.9)
        assert len(merges) == 0

    def test_single_centroid_no_merges(self):
        centroids = {0: np.array([1.0, 0.0])}
        merges = detect_merges(centroids)
        assert merges == []

    def test_merges_sorted_by_similarity_desc(self):
        centroids = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.95, 0.0]),
            2: np.array([0.0, 1.0]),
        }
        merges = detect_merges(centroids, merge_threshold=0.5)
        if len(merges) >= 2:
            assert merges[0][2] >= merges[1][2]
