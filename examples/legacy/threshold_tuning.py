"""
Threshold Tuning Demo - Finding the Optimal Threshold

This example demonstrates how the threshold parameter affects matching results
and how to find the optimal threshold for your use case.

**Estimated Runtime**: 2 minutes

**What this demonstrates**:
- How threshold affects precision and recall
- Visualizing threshold impact with examples
- Finding optimal threshold with validation data
- Recommendations for common use cases

**Key Concepts**:
- **Lower threshold** (e.g., 0.5): More matches, lower precision, higher recall
- **Higher threshold** (e.g., 0.9): Fewer matches, higher precision, lower recall
- **Optimal threshold**: Balance between precision and recall for your use case
"""

from semanticmatcher import EmbeddingMatcher, EntityMatcher
from typing import List, Any


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def count_matches(results: List[Any]) -> int:
    """Count non-None results."""
    return sum(1 for r in results if r is not None)


def main():
    """Demonstrate threshold tuning."""

    print("=" * 80)
    print("Threshold Tuning Demo - Finding the Optimal Threshold")
    print("=" * 80)
    print()

    # Sample data
    entities = [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
        {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
        {"id": "JP", "name": "Japan", "aliases": ["Nippon", "Nihon"]},
        {"id": "GB", "name": "United Kingdom", "aliases": ["UK", "Great Britain"]},
    ]

    # Test queries with varying similarity
    test_queries = [
        # High similarity (should match at all thresholds)
        "Germany",
        "France",
        "Japan",
        "United States",
        "United Kingdom",
        # Medium similarity (threshold dependent)
        "Deutschland",
        "USA",
        "UK",
        "Nippon",
        # Low similarity (only match at low thresholds)
        "Deutsch",  # Partial match
        "United",  # Partial match
        "Kingdom",  # Partial match
        "Jap",  # Partial match
        # Very low/no similarity (should not match at any threshold)
        "UnknownCountry",
        "NotFound",
        "XYZ",
    ]

    print(f"Entities: {len(entities)}")
    print(f"Test queries: {len(test_queries)}")
    print()

    # =========================================================================
    # Part 1: Threshold Impact on EmbeddingMatcher
    # =========================================================================
    print_section("Part 1: Threshold Impact on EmbeddingMatcher")

    thresholds_to_test = [0.5, 0.6, 0.7, 0.8, 0.9]

    results_by_threshold = {}

    for threshold in thresholds_to_test:
        print(f"Testing threshold={threshold}...")

        matcher = EmbeddingMatcher(
            entities=entities,
            threshold=threshold,
            normalize=True,
        )
        matcher.build_index()

        results = []
        for query in test_queries:
            result = matcher.match(query, top_k=1)
            results.append(result)

        results_by_threshold[threshold] = results
        match_count = count_matches(results)
        print(
            f"  Matches: {match_count}/{len(test_queries)} ({100 * match_count / len(test_queries):.1f}%)"
        )
    print()

    # Detailed comparison
    print("Detailed Results by Threshold:")
    print(f"{'Query':<25} ", end="")
    for threshold in thresholds_to_test:
        print(f"{threshold:>4} ", end="")
    print()
    print("-" * 80)

    for i, query in enumerate(test_queries):
        print(f"{query:<25} ", end="")
        for threshold in thresholds_to_test:
            result = results_by_threshold[threshold][i]
            output = result["id"] if result else "-"
            print(f"{output:>4} ", end="")
        print()
    print()

    # =========================================================================
    # Part 2: Threshold Impact on EntityMatcher
    # =========================================================================
    print_section("Part 2: Threshold Impact on EntityMatcher (with Confidence)")

    # Training data
    training_data = [
        {"text": "Germany", "label": "DE"},
        {"text": "Deutschland", "label": "DE"},
        {"text": "France", "label": "FR"},
        {"text": "United States", "label": "US"},
        {"text": "USA", "label": "US"},
        {"text": "Japan", "label": "JP"},
        {"text": "United Kingdom", "label": "GB"},
        {"text": "UK", "label": "GB"},
    ]

    print("Training EntityMatcher...")
    entity_matcher = EntityMatcher(
        entities=entities,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.7,
    )
    entity_matcher.train(training_data, num_epochs=3)
    print("✓ Training complete")
    print()

    # Test with different thresholds
    print("Testing EntityMatcher with different thresholds:")
    print()

    entity_thresholds = [0.5, 0.7, 0.9]

    for threshold in entity_thresholds:
        entity_matcher.threshold = threshold

        results = []
        for query in test_queries[:10]:  # Test subset
            result = entity_matcher.predict(query)
            results.append(result)

        match_count = sum(1 for r in results if r is not None)
        print(f"Threshold {threshold}: {match_count}/10 matches")

        # Show which queries matched
        matched = [q for q, r in zip(test_queries[:10], results) if r is not None]
        print(f"  Matched: {matched}")
    print()

    # =========================================================================
    # Part 3: Finding Optimal Threshold with Validation Data
    # =========================================================================
    print_section("Part 3: Finding Optimal Threshold with Validation")

    print("Approach: Use a validation set with known matches")
    print()

    # Validation data: (query, expected_match)
    validation_data = [
        # Clear matches - should match at all reasonable thresholds
        ("Germany", "DE"),
        ("France", "FR"),
        ("USA", "US"),
        # Variation matches - need moderate threshold
        ("Deutschland", "DE"),
        ("UK", "GB"),
        # Partial matches - need low threshold
        ("Deutsch", "DE"),  # May or may not match depending on threshold
        # Non-matches - should NOT match at any threshold
        ("Unknown", None),
        ("NotFound", None),
    ]

    print("Validation Data:")
    for query, expected in validation_data:
        print(f"  '{query}' → {expected}")
    print()

    # Test thresholds
    print("Evaluating thresholds on validation data:")
    print()

    best_threshold = 0.7
    best_accuracy = 0.0

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        matcher = EmbeddingMatcher(entities=entities, threshold=threshold)
        matcher.build_index()

        correct = 0
        for query, expected in validation_data:
            result = matcher.match(query, top_k=1)
            predicted = result["id"] if result else None

            if predicted == expected:
                correct += 1

        accuracy = correct / len(validation_data)
        print(
            f"  Threshold {threshold}: {correct}/{len(validation_data)} correct ({accuracy:.1%})"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print()
    print(f"✓ Optimal threshold: {best_threshold} (accuracy: {best_accuracy:.1%})")
    print()

    # =========================================================================
    # Part 4: Recommendations by Use Case
    # =========================================================================
    print_section("Part 4: Threshold Recommendations by Use Case")

    print("Recommended threshold values based on use case:")
    print()

    recommendations = [
        (
            "High Precision (strict)",
            0.8,
            [
                "Exact entity matching",
                "Database lookups",
                "Financial/legal applications",
            ],
        ),
        (
            "Balanced (default)",
            0.7,
            [
                "General purpose matching",
                "Search autocomplete",
                "User input normalization",
            ],
        ),
        (
            "High Recall (permissive)",
            0.5,
            [
                "Fuzzy search",
                "Exploratory analysis",
                "Data cleaning/deduplication",
            ],
        ),
    ]

    for name, threshold, use_cases in recommendations:
        print(f"{name} (threshold={threshold}):")
        for use_case in use_cases:
            print(f"  • {use_case}")
        print()

    print("Adjustment Guidelines:")
    print("  - Too many false positives? Increase threshold (0.7 → 0.8)")
    print("  - Too many false negatives? Decrease threshold (0.7 → 0.6)")
    print("  - Not sure? Start with 0.7 and tune based on results")
    print()

    # =========================================================================
    # Part 5: Visualizing Threshold Impact
    # =========================================================================
    print_section("Part 5: Visualizing Threshold Impact (Text-Based)")

    print("Conceptual visualization of threshold impact:")
    print()

    print("Similarity Score Spectrum:")
    print("  0.0 ───────────────────────────────────────────── 1.0")
    print("        ↓              ↓              ↓              ↓")
    print("     Very Low       Medium         High          Exact")
    print("     (0.2-0.4)      (0.5-0.7)      (0.8-0.9)      (0.95+)")
    print()

    print("Threshold Effect:")
    print()
    print("  Threshold 0.5 (Low):")
    print("    [████████████████████████░░░░░░░░░░░░░░░░░░░░]")
    print("    More matches, includes lower confidence results")
    print()
    print("  Threshold 0.7 (Default):")
    print("    [████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░]")
    print("    Balanced match quality")
    print()
    print("  Threshold 0.9 (High):")
    print("    [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]")
    print("    Fewer matches, only high confidence results")
    print()

    print("Practical Example:")
    print()
    print("  Query: 'United'")
    print("  Similarity to 'United States': 0.55")
    print("  Similarity to 'United Kingdom': 0.52")
    print()
    print("  At threshold=0.5: Matches 'United States' ✓")
    print("  At threshold=0.6: No match (both below threshold)")
    print("  At threshold=0.7: No match (both below threshold)")
    print()


if __name__ == "__main__":
    main()
