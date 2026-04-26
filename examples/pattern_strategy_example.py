"""
Example: Pattern-Based Novelty Detection

Demonstrates how to use the PatternBasedNoveltyStrategy to detect
novel entities based on orthographic and linguistic patterns.
"""

from novelentitymatcher.novelty.strategies.pattern_strategy import (
    PatternBasedNoveltyStrategy,
    score_batch_novelty,
)


def main():
    """Demonstrate pattern-based novelty detection."""

    # Known entities (company names)
    known_companies = [
        "Apple Inc",
        "Microsoft Corporation",
        "Google LLC",
        "Amazon.com Inc",
        "Tesla Inc",
        "Meta Platforms Inc",
        "Netflix Inc",
        "NVIDIA Corporation",
        "Adobe Inc",
        "Intel Corporation",
    ]

    print("Known companies:")
    for company in known_companies:
        print(f"  - {company}")

    # Initialize the pattern-based strategy
    strategy = PatternBasedNoveltyStrategy(known_companies)

    # Test entities
    test_entities = [
        "Apple Inc",  # Known - low novelty
        "Microsoft Corp",  # Similar - low novelty
        "OpenAI LP",  # Novel pattern - higher novelty
        "xyz123",  # Very different - high novelty
        "Samsung Electronics",  # Novel but similar structure - moderate novelty
    ]

    print("\n" + "=" * 60)
    print("Novelty Detection Results:")
    print("=" * 60)

    for entity in test_entities:
        novelty_score = strategy.score_novelty(entity)

        # Interpret the score
        if novelty_score < 0.3:
            interpretation = "Known pattern"
        elif novelty_score < 0.6:
            interpretation = "Somewhat novel"
        else:
            interpretation = "Highly novel"

        print(f"\nEntity: {entity}")
        print(f"  Novelty Score: {novelty_score:.3f}")
        print(f"  Interpretation: {interpretation}")

    # Batch scoring example
    print("\n" + "=" * 60)
    print("Batch Scoring Example:")
    print("=" * 60)

    batch_entities = [
        "Apple Inc",
        "Google LLC",
        "Tesla Inc",
        "IBM Corporation",
        "ABC123",
    ]

    scores = score_batch_novelty(batch_entities, strategy)

    print(f"\nProcessed {len(batch_entities)} entities:")
    for entity, score in zip(batch_entities, scores, strict=False):
        print(f"  {entity:25s} -> {score:.3f}")

    # Pattern analysis
    print("\n" + "=" * 60)
    print("Extracted Patterns:")
    print("=" * 60)

    patterns = strategy.patterns

    print(f"\nCharacter n-grams (3-grams): {len(patterns['char_ngrams'])} unique")
    print(f"Character 4-grams: {len(patterns['char_4grams'])} unique")

    min_len, max_len = patterns["length_range"]
    print(f"\nLength range: {min_len} - {max_len} characters")

    print("\nCapitalization patterns:")
    for pattern in sorted(patterns["capitalization"]):
        print(f"  - {pattern}")

    print("\nPrefix distribution (top 5):")
    prefix_dist = patterns["prefix_distribution"]
    sorted_prefixes = sorted(prefix_dist.items(), key=lambda x: x[1], reverse=True)
    for prefix, count in sorted_prefixes[:5]:
        print(f"  {prefix:10s}: {count}")

    print("\nSuffix distribution (top 5):")
    suffix_dist = patterns["suffix_distribution"]
    sorted_suffixes = sorted(suffix_dist.items(), key=lambda x: x[1], reverse=True)
    for suffix, count in sorted_suffixes[:5]:
        print(f"  {suffix:10s}: {count}")


if __name__ == "__main__":
    main()
