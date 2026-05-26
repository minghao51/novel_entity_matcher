"""Performance benchmarks for novel_entity_matcher.

Measures:
1. argpartition vs argsort top-k (embedding_matcher.py match logic)
2. Vectorized vs loop prototypical scoring (score_batch vs per-item is_novel)
3. LRU cache hit rate over repeated queries (LRUEmbeddingCache)
4. Pre-computed clusters vs full HDBSCAN re-cluster (fast centroid vs HDBSCAN)

Run: uv run benchmarks/perf_benchmark.py
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

sys.path.insert(0, "src")

N_ENTITIES = 2000
N_QUERIES = 500
EMBEDDING_DIM = 128
N_CLASSES = 20
TOP_K = 5
N_REPETITIONS = 5


def _timer(label: str, fn, n_rep: int = N_REPETITIONS) -> tuple[float, float]:
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    best = min(times)
    mean = float(np.mean(times))
    print(f"  {label}: best={best*1000:.2f}ms  mean={mean*1000:.2f}ms")
    return best, mean


def bench_argpartition_vs_argsort():
    """Benchmark 1: argpartition vs argsort for top-k selection.

    Mirrors the logic in EmbeddingMatcher.match() at
    src/novelentitymatcher/core/embedding_matcher.py:202-209.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 1: argpartition vs argsort top-k")
    print(f"  entities={N_ENTITIES}  queries={N_QUERIES}  dim={EMBEDDING_DIM}  top_k={TOP_K}")
    print("=" * 70)

    rng = np.random.default_rng(42)
    query_embs = rng.standard_normal((N_QUERIES, EMBEDDING_DIM)).astype(np.float32)
    query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)
    entity_embs = rng.standard_normal((N_ENTITIES, EMBEDDING_DIM)).astype(np.float32)
    entity_embs /= np.linalg.norm(entity_embs, axis=1, keepdims=True)
    sim_matrix = cosine_similarity(query_embs, entity_embs)

    def argsort_topk():
        results = []
        for row in sim_matrix:
            sorted_idx = np.argsort(row)[::-1]
            results.append(sorted_idx[:TOP_K])
        return results

    def argpartition_topk():
        results = []
        fetch_n = min(TOP_K * 3, sim_matrix.shape[1])
        for row in sim_matrix:
            if fetch_n < sim_matrix.shape[1]:
                part_idx = np.argpartition(row, -fetch_n)[-fetch_n:]
                sorted_idx = part_idx[np.argsort(row[part_idx])[::-1]]
            else:
                sorted_idx = np.argsort(row)[::-1]
            results.append(sorted_idx[:TOP_K])
        return results

    argsort_best, argsort_mean = _timer("argsort     ", argsort_topk)
    argpart_best, argpart_mean = _timer("argpartition", argpartition_topk)

    r_best = argsort_best / argpart_best
    r_mean = argsort_mean / argpart_mean
    print(f"\n  Speedup (best): {r_best:.2f}x")
    print(f"  Speedup (mean): {r_mean:.2f}x")
    print(f"  argsort best={argsort_best*1000:.2f}ms  ->  argpartition best={argpart_best*1000:.2f}ms")
    return {
        "argsort_best_ms": argsort_best * 1000,
        "argpartition_best_ms": argpart_best * 1000,
        "speedup_best": r_best,
        "speedup_mean": r_mean,
    }


def bench_prototypical_vectorized_vs_loop():
    """Benchmark 2: vectorized score_batch vs per-item is_novel loop.

    Mirrors PrototypicalDetector.score_batch() vs repeated is_novel() calls
    in src/novelentitymatcher/novelty/strategies/prototypical_impl.py.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 2: vectorized score_batch vs per-item is_novel loop")
    print(f"  n_texts={N_QUERIES}  n_classes={N_CLASSES}  dim={EMBEDDING_DIM}")
    print("=" * 70)

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((N_QUERIES, EMBEDDING_DIM)).astype(np.float64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    prototypes = {}
    for i in range(N_CLASSES):
        proto = rng.standard_normal(EMBEDDING_DIM).astype(np.float64)
        proto /= np.linalg.norm(proto)
        prototypes[f"class_{i}"] = proto

    label_list = list(prototypes.keys())
    proto_matrix = np.array([prototypes[lbl] for lbl in label_list])
    threshold = 0.5

    def vectorized_batch():
        dist_matrix = cosine_distances(embeddings, proto_matrix)
        nearest_idx = np.argmin(dist_matrix, axis=1)
        min_distances = dist_matrix[np.arange(len(embeddings)), nearest_idx]
        results = []
        for i in range(len(embeddings)):
            is_novel = bool(min_distances[i] > threshold)
            results.append((is_novel, float(min_distances[i]), label_list[nearest_idx[i]]))
        return results

    def per_item_loop():
        results = []
        for i in range(len(embeddings)):
            emb = embeddings[i]
            nearest_label = None
            min_dist = float("inf")
            for label, proto in prototypes.items():
                d = float(cosine_distances(emb.reshape(1, -1), proto.reshape(1, -1))[0, 0])
                if d < min_dist:
                    min_dist = d
                    nearest_label = label
            is_novel = min_dist > threshold
            results.append((is_novel, min_dist, nearest_label))
        return results

    loop_best, loop_mean = _timer("per-item loop ", per_item_loop)
    vec_best, vec_mean = _timer("vectorized    ", vectorized_batch)

    r_best = loop_best / vec_best
    r_mean = loop_mean / vec_mean
    print(f"\n  Speedup (best): {r_best:.2f}x")
    print(f"  Speedup (mean): {r_mean:.2f}x")
    print(f"  loop best={loop_best*1000:.2f}ms  ->  vectorized best={vec_best*1000:.2f}ms")
    return {
        "loop_best_ms": loop_best * 1000,
        "vectorized_best_ms": vec_best * 1000,
        "speedup_best": r_best,
        "speedup_mean": r_mean,
    }


class _MiniLRUEmbeddingCache:
    """Re-implementation of LRUEmbeddingCache for isolated benchmarking.

    Mirrors src/novelentitymatcher/utils/embeddings.py:271-360.
    """

    def __init__(self, max_entries: int = 10_000):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> np.ndarray | None:
        key = text
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = text
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = embedding
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def get_batch(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[int]]:
        results: list[np.ndarray | None] = []
        uncached: list[int] = []
        for i, text in enumerate(texts):
            key = text
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                results.append(self._cache[key])
            else:
                self._misses += 1
                results.append(None)
                uncached.append(i)
        return results, uncached

    def put_batch(self, texts: list[str], embeddings: np.ndarray) -> None:
        for text, emb in zip(texts, embeddings, strict=False):
            key = text
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = emb
            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, int | float]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "hit_rate": self.hit_rate,
        }


def bench_lru_cache_hit_rate():
    """Benchmark 3: LRU cache hit rate over repeated queries.

    Mirrors LRUEmbeddingCache usage in
    src/novelentitymatcher/utils/embeddings.py:271-360
    and EmbeddingMatcher._encode_with_cache.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 3: LRU embedding cache hit rate")
    print(f"  total_texts={N_ENTITIES}  queries={N_QUERIES}  dim={EMBEDDING_DIM}")
    print("=" * 70)

    rng = np.random.default_rng(42)
    all_texts = [f"text_entity_{i}" for i in range(N_ENTITIES)]
    all_embeddings = rng.standard_normal((N_ENTITIES, EMBEDDING_DIM)).astype(np.float32)

    cache = _MiniLRUEmbeddingCache(max_entries=10000)
    for text, emb in zip(all_texts, all_embeddings, strict=False):
        cache.put(text, emb)

    # Simulate repeated queries: each query batch picks random texts with overlap
    n_rounds = 20
    query_sets = []
    for _r in range(n_rounds):
        indices = rng.choice(N_ENTITIES, size=N_QUERIES, replace=True)
        query_texts = [all_texts[i] for i in indices]
        query_sets.append(query_texts)

    def run_queries():
        for query_texts in query_sets:
            _cached, uncached = cache.get_batch(query_texts)
            if uncached:
                for idx in uncached:
                    cache.put(query_texts[idx], all_embeddings[idx])

    def no_cache_baseline():
        dummy_embeddings = rng.standard_normal((N_QUERIES, EMBEDDING_DIM)).astype(np.float32)
        for _ in query_sets:
            _ = dummy_embeddings.copy()

    cache_best, _cache_mean = _timer("with LRU cache   ", run_queries, n_rep=3)
    baseline_best, _baseline_mean = _timer(
        "no-cache baseline", no_cache_baseline, n_rep=3
    )

    stats = cache.stats
    print("\n  Cache stats after benchmark:")
    print(f"    hits:    {stats['hits']}")
    print(f"    misses:  {stats['misses']}")
    print(f"    size:    {stats['size']}")
    print(f"    hit_rate: {stats['hit_rate']:.2%}")
    print(f"\n  Cache overhead vs baseline: {cache_best/baseline_best:.2f}x")
    return {
        "cache_best_ms": cache_best * 1000,
        "baseline_best_ms": baseline_best * 1000,
        "hit_rate": stats["hit_rate"],
        "hits": stats["hits"],
        "misses": stats["misses"],
    }


def bench_precomputed_clusters_vs_hdbscan():
    """Benchmark 4: pre-computed centroids vs full HDBSCAN re-cluster.

    Mirrors the fast centroid-assignment path vs HDBSCAN fallback in
    src/novelentitymatcher/novelty/strategies/clustering.py:106-138.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 4: pre-computed cluster centroids vs full HDBSCAN re-cluster")
    print(f"  reference={N_ENTITIES}  queries={N_QUERIES}  dim={EMBEDDING_DIM}  n_clusters={N_CLASSES}")
    print("=" * 70)

    rng = np.random.default_rng(42)
    ref_embeddings = rng.standard_normal((N_ENTITIES, EMBEDDING_DIM)).astype(np.float32)
    ref_embeddings /= np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    query_embeddings = rng.standard_normal((N_QUERIES, EMBEDDING_DIM)).astype(np.float32)
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # Simulate pre-computed cluster structure
    ref_cluster_labels = rng.integers(0, N_CLASSES, size=N_ENTITIES)
    centroids = {}
    for lbl in range(N_CLASSES):
        mask = ref_cluster_labels == lbl
        if mask.any():
            centroids[int(lbl)] = np.mean(ref_embeddings[mask], axis=0)

    centroid_labels = list(centroids.keys())
    centroid_matrix = np.array([centroids[cl] for cl in centroid_labels])
    noise_percentile = 90

    def fast_centroid_path():
        dists = cosine_distances(query_embeddings, centroid_matrix)
        nearest_idx = np.argmin(dists, axis=1)
        query_labels = np.array([centroid_labels[i] for i in nearest_idx])
        min_dists = dists[np.arange(len(nearest_idx)), nearest_idx]
        if len(min_dists) > 1:
            noise_mask = min_dists > np.percentile(min_dists, noise_percentile)
        else:
            noise_mask = np.array([False])
        query_labels[noise_mask] = -1
        return query_labels

    def full_hdbscan_path():
        all_embeddings = np.vstack([ref_embeddings, query_embeddings])
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,
                min_samples=1,
                metric="cosine",
            )
            clusterer.fit(all_embeddings)
            return clusterer.labels_[N_ENTITIES:]
        except ImportError:
            # Fallback: use sklearn AgglomerativeClustering as stand-in
            from sklearn.cluster import AgglomerativeClustering

            clusterer = AgglomerativeClustering(
                n_clusters=N_CLASSES,
                metric="cosine",
                linkage="average",
            )
            labels = clusterer.fit_predict(all_embeddings)
            return labels[N_ENTITIES:]

    fast_best, fast_mean = _timer("fast centroid path", fast_centroid_path, n_rep=3)
    hdbscan_best, hdbscan_mean = _timer("full re-cluster   ", full_hdbscan_path, n_rep=3)

    r_best = hdbscan_best / fast_best
    r_mean = hdbscan_mean / fast_mean
    print(f"\n  Speedup (best): {r_best:.2f}x")
    print(f"  Speedup (mean): {r_mean:.2f}x")
    print(f"  HDBSCAN best={hdbscan_best*1000:.2f}ms  ->  centroid best={fast_best*1000:.2f}ms")
    return {
        "fast_centroid_best_ms": fast_best * 1000,
        "full_recluster_best_ms": hdbscan_best * 1000,
        "speedup_best": r_best,
        "speedup_mean": r_mean,
    }


def main():
    print("=" * 70)
    print("  novel_entity_matcher — Performance Benchmarks")
    print(f"  N_ENTITIES={N_ENTITIES}  N_QUERIES={N_QUERIES}  DIM={EMBEDDING_DIM}")
    print(f"  N_CLASSES={N_CLASSES}  TOP_K={TOP_K}  REPS={N_REPETITIONS}")
    print("=" * 70)

    results = {}
    results["1_argpartition"] = bench_argpartition_vs_argsort()
    results["2_prototypical"] = bench_prototypical_vectorized_vs_loop()
    results["3_lru_cache"] = bench_lru_cache_hit_rate()
    results["4_clustering"] = bench_precomputed_clusters_vs_hdbscan()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  1. argpartition vs argsort:   {results['1_argpartition']['speedup_best']:.2f}x speedup")
    print(f"  2. vectorized vs loop proto:  {results['2_prototypical']['speedup_best']:.2f}x speedup")
    print(f"  3. LRU cache hit rate:        {results['3_lru_cache']['hit_rate']:.2%}")
    print(f"  4. centroid vs HDBSCAN:       {results['4_clustering']['speedup_best']:.2f}x speedup")
    print("=" * 70)


if __name__ == "__main__":
    main()
