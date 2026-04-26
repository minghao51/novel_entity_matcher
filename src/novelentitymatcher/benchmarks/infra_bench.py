"""Infrastructure benchmarks: ANN backends and reranker models.

Benchmarks:
- ANN backends (hnswlib vs faiss vs exact): build time, query latency, recall@k
- Reranker models (bge-m3 vs bge-large vs ms-marco): accuracy, latency

Usage:
    novelentitymatcher-bench bench-ann --sizes 1000 10000 100000
    novelentitymatcher-bench bench-reranker --queries 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def benchmark_ann(
    sizes: list[int] | None = None,
    dim: int = 384,
    k: int = 10,
    n_queries: int = 100,
    output: str | None = None,
) -> dict[str, Any]:
    from ..novelty.storage.index import ANNIndex, ANNBackend

    sizes = sizes or [1000, 5000, 10000, 50000]
    results: dict[str, Any] = {}
    rng = np.random.RandomState(42)

    for size in sizes:
        print(f"\n--- ANN Benchmark: {size} vectors, dim={dim} ---")
        vectors = rng.randn(size, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        labels = [f"item_{i}" for i in range(size)]
        queries = rng.randn(n_queries, dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        exact_sims = None
        size_results: list[dict] = []

        for backend in [ANNBackend.HNSWLIB, ANNBackend.FAISS, ANNBackend.EXACT]:
            print(f"  Backend: {backend}")
            try:
                build_start = time.perf_counter()
                idx = ANNIndex(dim=dim, backend=backend, max_elements=size * 2)
                idx.add_vectors(vectors, labels)
                build_time = time.perf_counter() - build_start

                query_start = time.perf_counter()
                sims, indices = idx.knn_query(queries, k=k)
                query_time = time.perf_counter() - query_start
                qps = n_queries / query_time
                latency_ms = (query_time / n_queries) * 1000

                recall = 1.0
                if backend != ANNBackend.EXACT:
                    if exact_sims is None:
                        exact_idx = ANNIndex(dim=dim, backend=ANNBackend.EXACT, max_elements=size * 2)
                        exact_idx.add_vectors(vectors, labels)
                        exact_sims, exact_indices = exact_idx.knn_query(queries, k=k)

                    hits = 0
                    total = n_queries * k
                    for i in range(n_queries):
                        exact_set = set(exact_indices[i][:k])
                        approx_set = set(indices[i][:k])
                        hits += len(exact_set & approx_set)
                    recall = hits / total if total > 0 else 0.0

                result = {
                    "backend": backend,
                    "size": size,
                    "dim": dim,
                    "k": k,
                    "n_queries": n_queries,
                    "build_time_s": round(build_time, 4),
                    "query_time_s": round(query_time, 4),
                    "qps": round(qps, 1),
                    "latency_ms": round(latency_ms, 3),
                    "recall_at_k": round(recall, 4),
                }
                size_results.append(result)
                print(f"    Build: {build_time:.3f}s, Query: {query_time:.4f}s, "
                      f"QPS: {qps:.0f}, Latency: {latency_ms:.2f}ms, Recall@{k}: {recall:.4f}")
            except (ValueError, RuntimeError, ImportError) as e:
                print(f"    Failed: {e}")
                size_results.append({"backend": backend, "size": size, "error": str(e)})

        results[str(size)] = size_results

    print(f"\n{'=' * 80}")
    print("ANN BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Size':<10} {'Backend':<12} {'Build(s)':<10} {'QPS':<10} {'Latency(ms)':<14} {'Recall@K':<10}")
    print("-" * 66)
    for size_key, size_results in results.items():
        for r in size_results:
            if "error" not in r:
                print(f"{r['size']:<10} {r['backend']:<12} {r['build_time_s']:<10.3f} "
                      f"{r['qps']:<10.0f} {r['latency_ms']:<14.2f} {r['recall_at_k']:<10.4f}")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {output}")

    return results


def benchmark_reranker(
    models: list[str] | None = None,
    n_queries: int = 50,
    n_candidates: int = 20,
    output: str | None = None,
) -> dict[str, Any]:
    from ..core.reranker import CrossEncoderReranker

    models = models or ["bge-m3", "bge-large", "ms-marco"]
    queries = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "Explain quantum computing",
        "What causes climate change?",
        "How to make pasta carbonara",
    ] * max(1, n_queries // 5)

    results: dict[str, Any] = {}

    for model_alias in models:
        print(f"\n--- Reranker Benchmark: {model_alias} ---")
        try:
            reranker = CrossEncoderReranker(model=model_alias)

            candidates = [
                {"text": f"Candidate document {i} about various topics including science and technology", "id": str(i)}
                for i in range(n_candidates)
            ]

            load_start = time.perf_counter()
            _ = reranker.rerank(queries[0], candidates[:5], top_k=5)
            load_time = time.perf_counter() - load_start

            query_start = time.perf_counter()
            for q in queries:
                reranker.rerank(q, candidates, top_k=10)
            total_time = time.perf_counter() - query_start

            qps = len(queries) / total_time
            latency_ms = (total_time / len(queries)) * 1000

            result = {
                "model": model_alias,
                "n_queries": len(queries),
                "n_candidates": n_candidates,
                "load_time_s": round(load_time, 4),
                "total_time_s": round(total_time, 4),
                "qps": round(qps, 2),
                "latency_ms": round(latency_ms, 2),
            }
            results[model_alias] = result
            print(f"  Load: {load_time:.3f}s, Total: {total_time:.3f}s, "
                  f"QPS: {qps:.1f}, Latency: {latency_ms:.1f}ms")
        except (ValueError, RuntimeError, ImportError) as e:
            print(f"  Failed: {e}")
            results[model_alias] = {"model": model_alias, "error": str(e)}

    print(f"\n{'=' * 60}")
    print("RERANKER BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<15} {'Load(s)':<10} {'QPS':<10} {'Latency(ms)':<14}")
    print("-" * 49)
    for model_alias, r in results.items():
        if "error" not in r:
            print(f"{model_alias:<15} {r['load_time_s']:<10.3f} {r['qps']:<10.1f} {r['latency_ms']:<14.1f}")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {output}")

    return results


def main_ann(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ANN backend benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    benchmark_ann(sizes=args.sizes, dim=args.dim, k=args.k, n_queries=args.queries, output=args.output)
    return 0


def main_reranker(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reranker model benchmark")
    parser.add_argument("--models", nargs="+", default=["bge-m3", "bge-large", "ms-marco"])
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    benchmark_reranker(models=args.models, n_queries=args.queries, n_candidates=args.candidates, output=args.output)
    return 0
