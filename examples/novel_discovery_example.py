"""
Novel Class Discovery Example

This example demonstrates how to use the NovelEntityMatcher to discover
and propose new classes in text data.
"""

import asyncio

from novelentitymatcher import Matcher, NovelEntityMatcher
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import ConfidenceConfig, KNNConfig


async def main():
    """Run novel class discovery example."""

    # Step 1: Prepare training data with known classes
    print("Step 1: Preparing training data...")
    entities = [
        {"id": "physics", "name": "Physics"},
        {"id": "cs", "name": "Computer Science"},
        {"id": "biology", "name": "Biology"},
    ]

    training_texts = [
        # Physics samples
        "quantum mechanics",
        "wave function",
        "Schrödinger equation",
        "quantum entanglement",
        "particle physics experiment",
        "condensed matter systems",
        "nuclear magnetic resonance",
        "laser spectroscopy",
        # CS samples
        "machine learning",
        "neural networks",
        "algorithm design",
        "data structures",
        "distributed systems",
        "database query optimization",
        "computer vision models",
        "software architecture patterns",
        # Biology samples
        "gene expression",
        "protein synthesis",
        "cell division",
        "DNA replication",
        "microbial ecology study",
        "genome sequencing pipeline",
        "cell signaling pathways",
        "metabolic pathway analysis",
    ]

    training_labels = [
        "physics",
        "physics",
        "physics",
        "physics",
        "physics",
        "physics",
        "physics",
        "physics",
        "cs",
        "cs",
        "cs",
        "cs",
        "cs",
        "cs",
        "cs",
        "cs",
        "biology",
        "biology",
        "biology",
        "biology",
        "biology",
        "biology",
        "biology",
        "biology",
    ]

    # Step 2: Train the matcher
    print("Step 2: Training matcher...")
    matcher = Matcher(
        entities=entities,
        model="minilm",  # Use 'model' instead of 'model_name'
        threshold=0.6,
    )
    matcher.fit(texts=training_texts, labels=training_labels)

    # Step 3: Prepare test queries with known and novel classes
    print("\nStep 3: Testing with mixed known/novel queries...")
    test_queries = [
        # Known classes (ideally should NOT be detected as novel)
        "quantum superposition",
        "deep learning models",
        "CRISPR gene editing",
        "database indexing strategies",
        "cell signaling regulation",
        # Novel/interdisciplinary candidates
        "quantum biology applications",
        "bioinformatics algorithms",
        "computational chemistry methods",
        "autonomous lab robotics",
        # Potential outliers / noise
        "discount coupon redemption policy",
        "weather forecast for the ferry terminal",
        "gibberish zxqv lorem placeholder",
    ]

    # Step 4: Create NovelEntityMatcher
    print("\nStep 4: Creating NovelEntityMatcher...")
    novel_matcher = NovelEntityMatcher(
        matcher=matcher,
        detection_config=DetectionConfig(
            strategies=["confidence", "knn_distance"],
            confidence=ConfidenceConfig(threshold=0.45),
            knn_distance=KNNConfig(distance_threshold=0.45),
            combine_method="weighted",
        ),
        llm_provider=None,  # Set to "openrouter", "anthropic", or "openai" to use LLM
        auto_save=True,
        output_dir="./proposals",
    )

    # Step 5: Run novel class discovery
    print("\nStep 5: Running novel class discovery...")
    report = await novel_matcher.discover_novel_classes(
        queries=test_queries,
        existing_classes=["physics", "cs", "biology"],
        context="Scientific research domain",
        run_llm_proposal=False,  # Set True to use LLM for class naming
    )

    # Step 6: Display results
    print("\n" + "=" * 60)
    print("DISCOVERY RESULTS")
    print("=" * 60)
    print(f"\nDiscovery ID: {report.discovery_id}")
    print(f"Total queries: {report.metadata['num_queries']}")
    print(f"Novel samples detected: {report.metadata['num_novel_samples']}")
    print(
        "Tip: the stricter default combine_method now requires multiple novelty "
        "signals for most samples before they are reported. This example keeps "
        "to confidence + centroid signals because clustering becomes more "
        "useful on larger batches."
    )

    print("\nNovel Samples:")
    print("-" * 60)
    for i, sample in enumerate(report.novel_sample_report.novel_samples, 1):
        print(f"\n{i}. {sample.text}")
        print(f"   Predicted: {sample.predicted_class}")
        print(f"   Confidence: {sample.confidence:.2%}")
        print(f"   Signals: {', '.join(sample.signals.keys())}")

    print("\n" + "-" * 60)
    print(
        f"\nDetection Strategies Used: {report.novel_sample_report.detection_strategies}"
    )
    print("Signal Counts:")
    for strategy, count in report.novel_sample_report.signal_counts.items():
        print(f"  - {strategy}: {count} samples")

    # Step 7: Check saved files
    if report.output_file:
        print(f"\nProposals saved to: {report.output_file}")
        if "summary_file" in report.metadata:
            print(f"Summary saved to: {report.metadata['summary_file']}")

    # Step 8: Demonstrate batch processing
    print("\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)

    query_batches = [
        ["quantum computing", "machine learning"],
        ["bioinformatics", "computational chemistry"],
    ]

    batch_reports = novel_matcher.batch_discover(
        queries_batch=query_batches,
        existing_classes=["physics", "cs", "biology"],
    )

    print(f"\nProcessed {len(batch_reports)} batches")
    for i, batch_report in enumerate(batch_reports, 1):
        print(
            f"Batch {i}: {len(batch_report.novel_sample_report.novel_samples)} novel samples"
        )

    # Step 9: Demonstrate loading saved proposals
    print("\n" + "=" * 60)
    print("LOADING SAVED PROPOSALS")
    print("=" * 60)

    if report.output_file:
        from novelentitymatcher.novelty.storage import list_proposals, load_proposals

        # List all proposals
        all_proposals = list_proposals("./proposals")
        print(f"\nTotal saved discoveries: {len(all_proposals)}")

        print(f"\nLoading: {report.output_file}")
        loaded_report = load_proposals(report.output_file)
        print(f"Loaded discovery ID: {loaded_report.discovery_id}")
        print(f"Novel samples: {len(loaded_report.novel_sample_report.novel_samples)}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


def test_metadata_return():
    """Demonstrate metadata return from match()."""
    print("\n" + "=" * 60)
    print("METADATA RETURN EXAMPLE")
    print("=" * 60)

    # Create a simple matcher
    entities = [
        {"id": "tech", "name": "Technology"},
        {"id": "science", "name": "Science"},
    ]

    matcher = Matcher(
        entities=entities,
        model="minilm",
        threshold=0.6,
    )

    # Train with some data
    training_texts = [
        "software development",
        "algorithm design",
        "physics experiment",
        "chemistry research",
    ]

    training_labels = ["tech", "tech", "science", "science"]
    matcher.fit(texts=training_texts, labels=training_labels)

    # Match with metadata
    queries = ["machine learning", "quantum mechanics"]
    result = matcher.match(queries, return_metadata=True)

    print("\nMatch Result with Metadata:")
    print(f"Predictions: {result.predictions}")
    print(f"Confidences: {result.confidences}")
    print(f"Embeddings shape: {result.embeddings.shape}")
    print(f"Metadata: {result.metadata}")


def test_sync_match_with_metadata():
    """Demonstrate synchronous matching with metadata."""
    print("\n" + "=" * 60)
    print("SYNC METADATA MATCHING EXAMPLE")
    print("=" * 60)

    entities = [{"id": "test", "name": "Test"}]
    matcher = Matcher(entities=entities, model="minilm")

    # Simple match without training (zero-shot mode)
    result = matcher.match("test query", return_metadata=True)

    print("\nSingle query result:")
    print(f"Prediction: {result.predictions[0]}")
    print(f"Confidence: {result.confidences[0]:.2%}")
    print(f"Embedding shape: {result.embeddings.shape}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

    # Run additional examples
    test_metadata_return()
    test_sync_match_with_metadata()
