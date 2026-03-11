"""
Batch Processing Demo - Efficient Bulk Operations

This example demonstrates how to use batch processing for handling multiple
queries efficiently.

**Estimated Runtime**: 1-2 minutes

**What this demonstrates**:
- Sequential EmbeddingMatcher.match() calls for bulk query workloads
- HybridMatcher.match_bulk() with parallel processing
- Performance comparison: single vs batch processing
- Parallel processing benefits with n_jobs parameter
- Timing benchmarks and throughput metrics

**When to use batch processing**:
- Processing multiple queries at once
- Improving throughput with parallel processing
- Reducing per-request overhead
- Batch processing pipelines
"""

import time
from semanticmatcher import EmbeddingMatcher, HybridMatcher, BM25Blocking


def main():
    """Demonstrate batch processing workflow."""

    print("=" * 80)
    print("Batch Processing Demo - Efficient Bulk Operations")
    print("=" * 80)
    print()

    # Sample data
    entities = [
        {
            "id": "DE",
            "name": "Germany",
            "aliases": ["Deutschland", "Bundesrepublik Deutschland"],
        },
        {
            "id": "FR",
            "name": "France",
            "aliases": ["Frankreich", "République française"],
        },
        {
            "id": "US",
            "name": "United States",
            "aliases": ["USA", "America", "United States of America"],
        },
        {"id": "JP", "name": "Japan", "aliases": ["Nippon", "Nihon"]},
        {
            "id": "GB",
            "name": "United Kingdom",
            "aliases": ["UK", "Great Britain", "England"],
        },
        {"id": "CA", "name": "Canada", "aliases": ["Canadia"]},
        {"id": "IT", "name": "Italy", "aliases": ["Italia"]},
        {"id": "ES", "name": "Spain", "aliases": ["España"]},
    ]

    # Test queries
    test_queries = [
        "Deutschland",
        "USA",
        "Nippon",
        "UK",
        "Canada",
        "Italia",
        "France",
        "Japan",
        "Germany",
        "United States",
        "Great Britain",
        "Deutschland",  # Duplicate
        "ESP",
        "Italia",  # Duplicate
        "United Kingdom",
    ]

    print(f"Entities: {len(entities)}")
    print(f"Test queries: {len(test_queries)}")
    print()

    # =========================================================================
    # Part 1: EmbeddingMatcher - Batch vs Sequential
    # =========================================================================
    print("=" * 80)
    print("Part 1: EmbeddingMatcher - Sequential Processing")
    print("=" * 80)
    print()

    # Initialize and build index
    print("Initializing EmbeddingMatcher...")
    matcher = EmbeddingMatcher(
        entities=entities,
        threshold=0.7,
        normalize=True,
    )
    matcher.build_index()
    print("✓ Index built")
    print()

    # Sequential processing (calling match() for each query)
    print("Processing queries sequentially...")
    start_time = time.time()
    sequential_results = []
    for query in test_queries:
        result = matcher.match(query)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    print(f"✓ Processed {len(test_queries)} queries in {sequential_time:.3f} seconds")
    print(f"  Throughput: {len(test_queries) / sequential_time:.1f} queries/second")
    print()

    # Show sample results
    print("Sample results:")
    for i in range(min(3, len(test_queries))):
        print(f"  '{test_queries[i]}' → {sequential_results[i]}")
    print()

    # =========================================================================
    # Part 2: HybridMatcher - Batch Processing with match_bulk()
    # =========================================================================
    print("=" * 80)
    print("Part 2: HybridMatcher - Batch Processing (match_bulk)")
    print("=" * 80)
    print()

    # Initialize HybridMatcher
    print("Initializing HybridMatcher...")
    hybrid_matcher = HybridMatcher(
        entities=entities,
        blocking_strategy=BM25Blocking(),
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        normalize=True,
    )
    print("✓ HybridMatcher initialized")
    print()

    # Test with different n_jobs settings
    n_jobs_settings = [1, -1]  # Sequential, Parallel (all cores)

    for n_jobs in n_jobs_settings:
        mode = (
            "Sequential (n_jobs=1)"
            if n_jobs == 1
            else "Parallel (n_jobs=-1, all cores)"
        )
        print(f"Processing with match_bulk() - {mode}...")

        start_time = time.time()
        bulk_results = hybrid_matcher.match_bulk(
            queries=test_queries,
            blocking_top_k=50,
            retrieval_top_k=10,
            final_top_k=3,
            n_jobs=n_jobs,
        )
        bulk_time = time.time() - start_time

        # Count non-empty results
        non_empty = sum(1 for r in bulk_results if r)

        print(f"✓ Processed {len(test_queries)} queries in {bulk_time:.3f} seconds")
        print(f"  Throughput: {len(test_queries) / bulk_time:.1f} queries/second")
        print(f"  Non-empty results: {non_empty}/{len(bulk_results)}")
        print()

        # Show sample results
        print("Sample results:")
        for i in range(min(2, len(test_queries))):
            if bulk_results[i]:
                top_result = bulk_results[i][0]
                print(
                    f"  '{test_queries[i]}' → {top_result['id']} (score: {top_result.get('cross_encoder_score', 'N/A')})"
                )
            else:
                print(f"  '{test_queries[i]}' → No match")
        print()

    # =========================================================================
    # Part 3: Performance Comparison
    # =========================================================================
    print("=" * 80)
    print("Part 3: Performance Comparison")
    print("=" * 80)
    print()

    # Larger batch for meaningful comparison
    large_batch = test_queries * 10  # 150 queries

    print(f"Testing with {len(large_batch)} queries...")
    print()

    # Sequential
    start_time = time.time()
    sequential_large = []
    for query in large_batch:
        result = hybrid_matcher.match(
            query, blocking_top_k=50, retrieval_top_k=10, final_top_k=3
        )
        sequential_large.append(result)
    sequential_large_time = time.time() - start_time
    print(
        f"Sequential (n_jobs=1):  {sequential_large_time:.3f}s ({len(large_batch) / sequential_large_time:.1f} q/s)"
    )

    # Parallel
    start_time = time.time()
    _parallel_large = hybrid_matcher.match_bulk(
        queries=large_batch,
        blocking_top_k=50,
        retrieval_top_k=10,
        final_top_k=3,
        n_jobs=-1,
    )
    parallel_large_time = time.time() - start_time
    print(
        f"Parallel (n_jobs=-1):    {parallel_large_time:.3f}s ({len(large_batch) / parallel_large_time:.1f} q/s)"
    )
    print()

    speedup = sequential_large_time / parallel_large_time
    print(f"Speedup: {speedup:.2f}x faster with parallel processing")
    print()

    # =========================================================================
    # Part 4: Best Practices
    # =========================================================================
    print("=" * 80)
    print("Part 4: Batch Processing Best Practices")
    print("=" * 80)
    print()

    print("When to use batch processing:")
    print("  ✓ Processing multiple queries in a batch/job")
    print("  ✓ Building batch processing pipelines")
    print("  ✓ Reducing per-request overhead")
    print("  ✓ Utilizing multiple CPU cores")
    print()
    print("Parameters to tune:")
    print("  - n_jobs=-1: Use all CPU cores (default)")
    print("  - n_jobs=1: Sequential processing")
    print("  - n_jobs=N: Use N specific workers")
    print("  - chunk_size: Queries per chunk (auto-calculated by default)")
    print()
    print("Example usage patterns:")
    print("""
    # Pattern 1: Process list of queries
    results = matcher.match_bulk(queries, n_jobs=-1)

    # Pattern 2: Process in chunks for very large batches
    chunk_size = 1000
    all_results = []
    for i in range(0, len(queries), chunk_size):
        chunk = queries[i:i + chunk_size]
        results = matcher.match_bulk(chunk, n_jobs=-1)
        all_results.extend(results)

    # Pattern 3: Sequential for small batches
    results = matcher.match_bulk(queries, n_jobs=1)
    """)
    print()


if __name__ == "__main__":
    main()
