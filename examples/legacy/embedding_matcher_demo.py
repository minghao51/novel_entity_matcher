"""
EmbeddingMatcher Demo - Fast Similarity Matching (No Training)

This example demonstrates how to use EmbeddingMatcher for fast semantic
similarity matching without requiring any training.

**Estimated Runtime**: 30 seconds (no training, just index building)

**What this demonstrates**:
- EmbeddingMatcher initialization
- Building an embedding index from entities
- Matching queries with similarity scores
- Using the threshold parameter to filter low-confidence matches
- Using top_k to get multiple results
- Text normalization options
- Comparing results with vs without normalization

**When to use EmbeddingMatcher**:
- You need quick results without training
- You have simple, straightforward matching needs
- Text variations are minimal
- Accuracy requirements are moderate

**When to use EntityMatcher instead**:
- You have labeled training data
- You need higher accuracy on complex cases
- Text has significant variations (typos, translations, etc.)
"""

from semanticmatcher import EmbeddingMatcher, TextNormalizer
import time


def main():
    """Demonstrate EmbeddingMatcher workflow."""

    print("=" * 80)
    print("EmbeddingMatcher Demo - Fast Similarity Matching")
    print("=" * 80)
    print()

    # 1. Define your entities
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
        {"id": "CA", "name": "Canada", "aliases": ["Canadia", "True North"]},
    ]

    print(f"Entities defined: {len(entities)} countries")
    print()

    # 2. Initialize EmbeddingMatcher
    print("Initializing EmbeddingMatcher...")
    matcher = EmbeddingMatcher(
        entities=entities,
        model_name="sentence-transformers/paraphrase-mpnet-base-v2",
        threshold=0.7,  # Minimum similarity score (0-1)
        normalize=True,  # Apply text normalization
    )
    print("✓ Matcher initialized")
    print()

    # 3. Build the embedding index
    print("Building embedding index...")
    start_time = time.time()
    matcher.build_index()
    build_time = time.time() - start_time
    print(f"✓ Index built in {build_time:.2f} seconds")
    print()

    # 4. Basic matching - single query, single best match
    print("=" * 80)
    print("Basic Matching (Single Best Match)")
    print("=" * 80)
    print()

    queries = ["Deutschland", "USA", "Nippon", "UK"]

    for query in queries:
        result = matcher.match(query)
        if result:
            print(f"Query: '{query}'")
            print(f"  → Matched: {result['id']} (score: {result['score']:.3f})")
        else:
            print(f"Query: '{query}'")
            print("  → No match (below threshold)")
    print()

    # 5. Get top-k matches (multiple results)
    print("=" * 80)
    print("Top-K Matching (Multiple Results)")
    print("=" * 80)
    print()

    query = "United Kingdom"
    top_k = 3
    results = matcher.match(query, top_k=top_k)

    print(f"Query: '{query}' (showing top {top_k} results)")
    if results:
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['id']} - score: {result['score']:.3f}")
    else:
        print("  No matches found")
    print()

    # 6. Threshold demonstration
    print("=" * 80)
    print("Threshold Filtering")
    print("=" * 80)
    print()

    # Test with different thresholds
    test_query = "United"

    print(f"Query: '{test_query}'")
    print()

    for threshold in [0.5, 0.7, 0.9]:
        # Note: We need to create a new matcher or modify threshold
        # For this demo, we'll show the concept
        matcher_test = EmbeddingMatcher(
            entities=entities,
            threshold=threshold,
            normalize=True,
        )
        matcher_test.build_index()
        result = matcher_test.match(test_query, top_k=1)

        if result:
            print(
                f"  Threshold {threshold}: {result['id']} (score: {result['score']:.3f})"
            )
        else:
            print(f"  Threshold {threshold}: No match (too strict)")
    print()

    # 7. Text normalization comparison
    print("=" * 80)
    print("Text Normalization: With vs Without")
    print("=" * 80)
    print()

    # Create two matchers - one with normalization, one without
    matcher_normalized = EmbeddingMatcher(
        entities=entities,
        threshold=0.7,
        normalize=True,
    )
    matcher_normalized.build_index()

    matcher_no_norm = EmbeddingMatcher(
        entities=entities,
        threshold=0.7,
        normalize=False,
    )
    matcher_no_norm.build_index()

    messy_queries = [
        "  Deutschland  ",  # Extra spaces
        "UNITED STATES",  # All caps
        "japan",  # Lowercase
    ]

    print("Comparing results with and without text normalization:")
    print()

    for query in messy_queries:
        result_norm = matcher_normalized.match(query)
        result_no_norm = matcher_no_norm.match(query)

        norm_match = result_norm["id"] if result_norm else "None"
        no_norm_match = result_no_norm["id"] if result_no_norm else "None"

        print(f"Query: '{query}'")
        print(f"  With normalization:    {norm_match}")
        print(f"  Without normalization: {no_norm_match}")
        print()

    # 8. Using TextNormalizer directly
    print("=" * 80)
    print("TextNormalizer - Standalone Usage")
    print("=" * 80)
    print()

    normalizer = TextNormalizer(
        lowercase=True,
        remove_accents=True,
        remove_punctuation=True,
    )

    messy_texts = [
        "HÉLLO, World!",
        "Déutschland",
        "République française",
    ]

    print("TextNormalizer demonstration:")
    for text in messy_texts:
        normalized = normalizer.normalize(text)
        print(f"  '{text}' → '{normalized}'")
    print()

    # 9. Custom model selection
    print("=" * 80)
    print("Model Selection Options")
    print("=" * 80)
    print()

    print("Available model aliases (via model_name parameter):")
    print()
    print("  Speed vs Accuracy Tradeoffs:")
    print("  - 'minilm'      : Fastest, good for prototyping (0.1s)")
    print("  - 'mpnet'       : Balanced (default) (0.2s)")
    print("  - 'bge-base'    : High accuracy, English only (0.3s)")
    print("  - 'bge-m3'      : Multilingual, high accuracy (0.5s)")
    print()
    print("  Language-Specific:")
    print("  - 'paraphrase-multilingual-mpnet-base-v2' : Multilingual")
    print("  - 'sentence-transformers/LaBSE'           : Multilingual")
    print()
    print("Example usage:")
    print("  matcher = EmbeddingMatcher(")
    print("      entities=entities,")
    print("      model_name='bge-base'  # Use alias or full model name")
    print("  )")
    print()

    # 10. Performance tip
    print("=" * 80)
    print("Performance Tips")
    print("=" * 80)
    print()

    print("✓ Build index once, match many queries")
    print("✓ Use batch processing for multiple queries (match_bulk)")
    print("✓ Use 'minilm' for prototyping, switch to 'mpnet' for production")
    print("✓ Lower threshold for more matches, higher for precision")
    print("✓ Enable normalization for messy user input")
    print()


if __name__ == "__main__":
    main()
