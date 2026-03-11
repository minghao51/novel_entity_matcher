"""
Matcher Comparison Demo - EntityMatcher vs EmbeddingMatcher

This example demonstrates the differences between EntityMatcher and EmbeddingMatcher
to help you choose the right approach for your use case.

**Estimated Runtime**: 4-5 minutes (includes training time for EntityMatcher)

**What this demonstrates**:
- Same dataset, compare EntityMatcher vs EmbeddingMatcher
- Accuracy differences on various query types
- Speed differences (training time vs instant setup)
- Decision matrix: when to use which
- Training vs no-training tradeoff

**Key Takeaways**:
- **EmbeddingMatcher**: Fast setup, lower accuracy on complex cases
- **EntityMatcher**: Training required, higher accuracy on variations
"""

import time
from semanticmatcher import EntityMatcher, EmbeddingMatcher


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def compare_matchers(entities, training_data, test_queries):
    """Compare both matchers on the same dataset."""

    # =========================================================================
    # Part 1: EmbeddingMatcher (No Training)
    # =========================================================================
    print_section("Part 1: EmbeddingMatcher (No Training)")

    print("Initializing EmbeddingMatcher...")
    start_time = time.time()
    embedding_matcher = EmbeddingMatcher(
        entities=entities,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.7,
        normalize=True,
    )
    embedding_matcher.build_index()
    setup_time = time.time() - start_time
    print(f"✓ Setup complete in {setup_time:.2f} seconds (no training required)")
    print()

    # Run queries
    print("Running test queries...")
    start_time = time.time()
    embedding_results = []
    for query in test_queries:
        result = embedding_matcher.match(query, top_k=1)
        embedding_results.append(result)
    query_time = time.time() - start_time
    print(f"✓ Processed {len(test_queries)} queries in {query_time:.3f} seconds")
    print()

    # =========================================================================
    # Part 2: EntityMatcher (Training Required)
    # =========================================================================
    print_section("Part 2: EntityMatcher (Training Required)")

    print("Initializing EntityMatcher...")
    start_time = time.time()
    entity_matcher = EntityMatcher(
        entities=entities,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.7,
        normalize=True,
    )
    entity_setup_time = time.time() - start_time
    print("✓ Matcher initialized")
    print()

    print("Training model (this may take 1-2 minutes)...")
    train_start = time.time()
    entity_matcher.train(
        training_data=training_data,
        num_epochs=4,
        batch_size=16,
    )
    train_time = time.time() - train_start
    print(f"✓ Training complete in {train_time:.2f} seconds")
    print()

    # Run queries
    print("Running test queries...")
    start_time = time.time()
    entity_results = []
    for query in test_queries:
        result = entity_matcher.predict(query)
        entity_results.append(result)
    query_time_entity = time.time() - start_time
    print(f"✓ Processed {len(test_queries)} queries in {query_time_entity:.3f} seconds")
    print()

    # =========================================================================
    # Part 3: Side-by-Side Comparison
    # =========================================================================
    print_section("Part 3: Side-by-Side Comparison")

    print(
        f"{'Query':<30} {'EmbeddingMatcher':<20} {'EntityMatcher':<20} {'Match?':<10}"
    )
    print("-" * 80)

    matches = 0
    for query, emb_result, ent_result in zip(
        test_queries, embedding_results, entity_results
    ):
        emb_id = emb_result["id"] if emb_result else "None"
        ent_id = ent_result if ent_result else "None"

        # Check if they match
        match_symbol = "✓" if emb_id == ent_id else "✗"
        if emb_id == ent_id and emb_id != "None":
            matches += 1

        print(f"{query:<30} {emb_id:<20} {ent_id:<20} {match_symbol:<10}")

    print()
    print(
        f"Agreement: {matches}/{len(test_queries)} ({100 * matches / len(test_queries):.1f}%)"
    )
    print()

    # =========================================================================
    # Part 4: Performance Summary
    # =========================================================================
    print_section("Part 4: Performance Summary")

    print("EmbeddingMatcher:")
    print(f"  Setup time:     {setup_time:.2f}s")
    print(
        f"  Query time:     {query_time:.3f}s ({len(test_queries) / query_time:.1f} q/s)"
    )
    print(f"  Total time:     {setup_time + query_time:.2f}s")
    print("  Training:       Not required")
    print()

    print("EntityMatcher:")
    print(f"  Setup time:     {entity_setup_time:.2f}s (init)")
    print(f"  Training time:  {train_time:.2f}s")
    print(
        f"  Query time:     {query_time_entity:.3f}s ({len(test_queries) / query_time_entity:.1f} q/s)"
    )
    print(
        f"  Total time:     {entity_setup_time + train_time + query_time_entity:.2f}s"
    )
    print(f"  Training:       Required ({train_time:.2f}s)")
    print()

    # =========================================================================
    # Part 5: Decision Matrix
    # =========================================================================
    print_section("Part 5: When to Use Which Matcher")

    print("Decision Matrix:")
    print()
    print("Use EmbeddingMatcher if:")
    print("  ✓ You need quick results without training")
    print("  ✓ Text variations are minimal (standard names, no typos)")
    print("  ✓ Accuracy requirements are moderate")
    print("  ✓ You're prototyping or in development")
    print("  ✓ Your data changes frequently (retraining would be costly)")
    print()

    print("Use EntityMatcher if:")
    print("  ✓ You have labeled training examples (3-5 per entity)")
    print("  ✓ You need higher accuracy on complex cases")
    print("  ✓ Text has significant variations (typos, translations, abbreviations)")
    print("  ✓ You can afford 1-3 minutes of training time")
    print("  ✓ Your entities are stable (infrequent retraining needed)")
    print()

    print("Use HybridMatcher if:")
    print("  ✓ You have very large datasets (>10,000 entities)")
    print("  ✓ You need both speed and accuracy")
    print("  ✓ You can afford a more complex setup")
    print()

    # =========================================================================
    # Part 6: Accuracy by Query Type
    # =========================================================================
    print_section("Part 6: Analysis by Query Type")

    # Categorize queries by difficulty
    exact_matches = ["Germany", "France", "Japan"]
    variations = ["Deutschland", "USA", "Nippon", "UK"]
    _complex_variations = [
        "United States",
        "Great Britain",
        "Deutschland (typo: Deutchland)",
    ]

    print("Query Type Analysis:")
    print()

    # Exact matches
    print("Exact Match Queries:")
    for query in exact_matches:
        emb = (
            embedding_results[test_queries.index(query)]["id"]
            if embedding_results[test_queries.index(query)]
            else "None"
        )
        ent = entity_results[test_queries.index(query)]
        print(f"  '{query}': EmbeddingMatcher={emb}, EntityMatcher={ent}")
    print()

    # Variations
    print("Simple Variations:")
    for query in variations:
        idx = test_queries.index(query) if query in test_queries else None
        if idx is not None:
            emb = embedding_results[idx]["id"] if embedding_results[idx] else "None"
            ent = entity_results[idx]
            print(f"  '{query}': EmbeddingMatcher={emb}, EntityMatcher={ent}")
    print()

    print("Key Insight:")
    print("  - EmbeddingMatcher: Handles exact matches well, struggles with variations")
    print("  - EntityMatcher: Handles both exact matches and variations better")
    print("  - Training EntityMatcher with variation examples improves accuracy")
    print()


def main():
    """Run the matcher comparison."""

    print("=" * 80)
    print("Matcher Comparison Demo - EntityMatcher vs EmbeddingMatcher")
    print("=" * 80)
    print()
    print("This demo compares both matchers on the same dataset to help you choose")
    print("the right approach for your use case.")
    print()

    # Define entities
    entities = [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
        {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
        {"id": "JP", "name": "Japan", "aliases": ["Nippon", "Nihon"]},
        {"id": "GB", "name": "United Kingdom", "aliases": ["UK", "Great Britain"]},
    ]

    # Training data for EntityMatcher
    training_data = [
        # Germany
        {"text": "Germany", "label": "DE"},
        {"text": "Deutschland", "label": "DE"},
        # France
        {"text": "France", "label": "FR"},
        {"text": "Frankreich", "label": "FR"},
        # United States
        {"text": "United States", "label": "US"},
        {"text": "USA", "label": "US"},
        {"text": "America", "label": "US"},
        # Japan
        {"text": "Japan", "label": "JP"},
        {"text": "Nippon", "label": "JP"},
        # United Kingdom
        {"text": "United Kingdom", "label": "GB"},
        {"text": "UK", "label": "GB"},
        {"text": "Great Britain", "label": "GB"},
    ]

    # Test queries with varying difficulty
    test_queries = [
        "Germany",  # Exact match
        "France",  # Exact match
        "Japan",  # Exact match
        "Deutschland",  # Variation
        "USA",  # Variation
        "Nippon",  # Variation
        "UK",  # Variation
        "United States",  # Full name
        "Great Britain",  # Full name
        "Deutchland",  # Typo
    ]

    print(f"Entities: {len(entities)}")
    print(f"Training examples: {len(training_data)}")
    print(f"Test queries: {len(test_queries)}")
    print()

    compare_matchers(entities, training_data, test_queries)


if __name__ == "__main__":
    main()
