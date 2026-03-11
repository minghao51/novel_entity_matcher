"""
Hybrid Matching Demo

Demonstrates the three-stage waterfall pipeline:
1. Blocking - Fast lexical filtering (NoOpBlocking for small datasets)
2. Bi-Encoder Retrieval - Semantic similarity search
3. Cross-Encoder Reranking - Precise scoring

For this demo with a small dataset, we use NoOpBlocking.
For larger datasets (>1000 entities), use BM25Blocking or TFIDFBlocking.
"""

import semanticmatcher as sm


def _format_score(value, precision: int = 3) -> str:
    """Format numeric scores safely for display."""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return "N/A"


def main():
    """Demonstrate hybrid matching pipeline."""

    # Sample product data
    # Note: For BM25Blocking to work properly, entities need a searchable text field.
    # For this small dataset (8 products), we use NoOpBlocking.
    # For larger datasets, use BM25Blocking and ensure entities have proper text fields.
    products = [
        {
            "id": "p1",
            "name": "Apple iPhone 15 Pro",
            "text": "Apple iPhone 15 Pro 256GB with A17 chip",
        },
        {
            "id": "p2",
            "name": "Samsung Galaxy S24",
            "text": "Samsung Galaxy S24 Ultra Android smartphone",
        },
        {
            "id": "p3",
            "name": "Google Pixel 8",
            "text": "Google Pixel 8 Pro with Tensor G3",
        },
        {
            "id": "p4",
            "name": "Apple MacBook Pro",
            "text": "Apple MacBook Pro 14-inch M3 Max laptop",
        },
        {
            "id": "p5",
            "name": "Dell XPS 15",
            "text": "Dell XPS 15 laptop Windows 11",
        },
        {
            "id": "p6",
            "name": "Sony WH-1000XM5",
            "text": "Sony WH-1000XM5 noise cancelling headphones",
        },
        {
            "id": "p7",
            "name": "AirPods Pro 2",
            "text": "Apple AirPods Pro 2nd generation USB-C",
        },
        {
            "id": "p8",
            "name": "Samsung Galaxy Buds",
            "text": "Samsung Galaxy Buds2 Pro true wireless",
        },
    ]

    print("=" * 80)
    print("Hybrid Matching Pipeline Demo")
    print("=" * 80)
    print(f"Dataset size: {len(products)} products")
    print("Note: Using NoOpBlocking for this small dataset (BM25 for 1000+ entities)")
    print()

    # Initialize hybrid matcher
    # Using NoOpBlocking since we have a small dataset (< 1000 entities)
    # For larger datasets, use BM25Blocking() for better performance
    print("Initializing HybridMatcher with NoOpBlocking...")
    matcher = sm.HybridMatcher(
        entities=products,
        blocking_strategy=sm.NoOpBlocking(),  # Pass-through for small datasets
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",  # Fast model
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast reranker
    )
    print("✓ Matcher initialized\n")

    # Test queries - designed to match product descriptions
    queries = [
        "iPhone 15",  # Should match "Apple iPhone 15 Pro"
        "MacBook Pro laptop",  # Should match "Apple MacBook Pro"
        "Sony WH-1000XM5",  # Should match "Sony WH-1000XM5"
        "Samsung Galaxy Buds",  # Should match "Samsung Galaxy Buds"
    ]

    print("Running queries through three-stage pipeline...\n")

    for query in queries:
        print(f"Query: '{query}'")
        print("-" * 40)

        results = matcher.match(
            query,
            blocking_top_k=8,  # Include all 8 products
            retrieval_top_k=3,
            final_top_k=2,
        )

        if results and len(results) > 0:
            for i, result in enumerate(results, 1):
                entity = next((p for p in products if p["id"] == result["id"]), None)
                if entity:
                    print(f"  {i}. {entity['name']}")
                    print(
                        f"     Bi-Encoder Score: {_format_score(result.get('score'))}"
                    )
                    print(
                        "     Cross-Encoder Score: "
                        f"{_format_score(result.get('cross_encoder_score'))}"
                    )
        else:
            print("  No matches found")
        print()

    # Demonstrate stage-by-stage
    print("=" * 80)
    print("Stage-by-Stage Breakdown for: 'iPhone 15'")
    print("=" * 80)

    query = "iPhone 15"

    # Stage 1: Blocking
    print("\nStage 1: Blocking (NoOpBlocking for small datasets)")
    blocked = matcher.blocker.block(query, products, top_k=8)  # Get all 8 products
    print(f"Candidates after blocking: {len(blocked)}")
    for p in blocked[:3]:
        print(f"  - {p['name']}")

    # Stage 2: Retrieval
    print("\nStage 2: Bi-Encoder Retrieval")
    retrieved = matcher.retriever.match(query, candidates=blocked, top_k=3)
    print(f"Retrieved result type: {type(retrieved)}")
    print(f"Retrieved: {retrieved}")

    if retrieved is not None:
        if isinstance(retrieved, list):
            if len(retrieved) > 0:
                print(f"Candidates after retrieval: {len(retrieved)}")
                for r in retrieved[:3]:
                    entity = next((p for p in products if p["id"] == r["id"]), None)
                    if entity:
                        print(f"  - {entity['name']} (score: {r['score']:.3f})")
            else:
                print("No candidates passed retrieval threshold (0.7)")
        else:
            # Single result
            entity = next((p for p in products if p["id"] == retrieved["id"]), None)
            if entity:
                print(
                    "Candidate after retrieval: "
                    f"{entity['name']} (score: {_format_score(retrieved.get('score'))})"
                )
            else:
                print("Single retrieved result but no entity found")
    else:
        print("No candidates retrieved (below threshold)")

    # Stage 3: Reranking
    print("\nStage 3: Cross-Encoder Reranking")

    # Convert to list if needed
    if retrieved is not None and not isinstance(retrieved, list):
        retrieved = [retrieved]

    if retrieved and isinstance(retrieved, list) and len(retrieved) > 0:
        final = matcher.reranker.rerank(query, retrieved, top_k=2)
        print(f"Final results: {len(final)}")
        for r in final:
            entity = next((p for p in products if p["id"] == r["id"]), None)
            if entity:
                print(
                    f"  - {entity['name']} "
                    f"(score: {_format_score(r.get('cross_encoder_score'))})"
                )
    else:
        print("No results to rerank")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
