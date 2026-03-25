"""
Example: One-Class SVM Novelty Detection

Demonstrates how to use the OneClassNoveltyDetector to detect
novel entities using one-class SVM with sentence transformer embeddings.
"""

from novelentitymatcher.novelty.strategies.oneclass_strategy import (
    OneClassNoveltyDetector,
)


def main():
    """Demonstrate One-Class SVM novelty detection."""

    # Known entities (AI/ML related terms)
    known_entities = [
        "machine learning",
        "neural networks",
        "deep learning",
        "artificial intelligence",
        "computer vision",
        "natural language processing",
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "transfer learning",
        "feature engineering",
        "model training",
        "neural architecture",
        "gradient descent",
        "backpropagation",
    ]

    print("Known entities (AI/ML domain):")
    for entity in known_entities[:5]:
        print(f"  - {entity}")
    print(f"  ... and {len(known_entities) - 5} more")

    # Initialize and train the detector
    print("\n" + "=" * 60)
    print("Training One-Class SVM detector...")
    print("=" * 60)

    detector = OneClassNoveltyDetector(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        nu=0.1,  # Expect 10% outliers
    )

    detector.train(known_entities, show_progress=True)

    # Get support vector info
    info = detector.get_support_vectors_info()
    print("\nSupport vector info:")
    print(f"  Number of support vectors: {info.get('n_support_vectors', 'N/A')}")

    # Test entities
    test_entities = [
        ("machine learning", "Known - should be low novelty"),
        ("deep neural networks", "Similar - should be low novelty"),
        ("organic farming", "Novel domain - should be high novelty"),
        ("stock trading", "Novel domain - should be high novelty"),
        ("computer graphics", "Related but different - medium novelty"),
    ]

    print("\n" + "=" * 60)
    print("Novelty Detection Results:")
    print("=" * 60)

    for entity, description in test_entities:
        is_novel, confidence = detector.is_novel(entity)

        status = "NOVEL" if is_novel else "KNOWN"
        print(f"\nEntity: {entity}")
        print(f"  Status: {status}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Description: {description}")

    # Batch scoring example
    print("\n" + "=" * 60)
    print("Batch Scoring Example:")
    print("=" * 60)

    batch_entities = [
        "machine learning",
        "organic farming",
        "neural networks",
        "crop rotation",
        "deep learning",
    ]

    results = detector.score_batch(batch_entities)

    print(f"\nProcessed {len(batch_entities)} entities:")
    for entity, (is_novel, confidence) in zip(batch_entities, results):
        status = "NOVEL" if is_novel else "KNOWN"
        print(f"  {entity:25s} -> {status:8s} (conf: {confidence:.3f})")

    # Save and load example
    print("\n" + "=" * 60)
    print("Model Persistence Example:")
    print("=" * 60)

    save_path = "./models/oneclass_novelty"

    # Save
    print(f"\nSaving model to: {save_path}")
    detector.save(save_path)
    print("Model saved successfully!")

    # Load
    print(f"\nLoading model from: {save_path}")
    loaded_detector = OneClassNoveltyDetector.load(save_path)
    print("Model loaded successfully!")

    # Verify loaded model works
    is_novel, confidence = loaded_detector.is_novel("machine learning")
    print("\nTest with loaded model:")
    print("  Entity: 'machine learning'")
    print(f"  Status: {'NOVEL' if is_novel else 'KNOWN'}")
    print(f"  Confidence: {confidence:.3f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
