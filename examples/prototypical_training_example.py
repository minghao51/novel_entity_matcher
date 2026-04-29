"""
Example: Prototypical Network Novelty Detection

Demonstrates how to use the PrototypicalNoveltyDetector to detect
novel entities using prototypical networks with class-based prototypes.
"""

from novelentitymatcher.novelty.strategies.prototypical_strategy import (
    PrototypicalNoveltyDetector,
)


def main():
    """Demonstrate prototypical novelty detection."""

    # Training data with labeled entities
    training_data = [
        # Machine Learning category
        {"text": "machine learning algorithms", "label": "ml"},
        {"text": "neural network architectures", "label": "ml"},
        {"text": "deep learning models", "label": "ml"},
        {"text": "supervised learning", "label": "ml"},
        {"text": "unsupervised learning", "label": "ml"},
        # Computer Vision category
        {"text": "computer vision tasks", "label": "cv"},
        {"text": "image processing", "label": "cv"},
        {"text": "object detection", "label": "cv"},
        {"text": "image segmentation", "label": "cv"},
        {"text": "visual recognition", "label": "cv"},
        # Natural Language Processing category
        {"text": "natural language processing", "label": "nlp"},
        {"text": "text analysis", "label": "nlp"},
        {"text": "language models", "label": "nlp"},
        {"text": "sentiment analysis", "label": "nlp"},
        {"text": "named entity recognition", "label": "nlp"},
    ]

    print("Training data by category:")
    by_label = {}
    for item in training_data:
        label = item["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(item["text"])

    for label, texts in sorted(by_label.items()):
        print(f"\n  {label.upper()}:")
        for text in texts[:3]:
            print(f"    - {text}")
        print(f"    ... and {len(texts) - 3} more")

    # Initialize and train the detector
    print("\n" + "=" * 60)
    print("Training Prototypical detector...")
    print("=" * 60)

    detector = PrototypicalNoveltyDetector(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        distance_threshold=0.5,
        distance_metric="cosine",
    )

    detector.train(training_data, show_progress=True)

    # Get prototype information
    print("\n" + "=" * 60)
    print("Prototype Information:")
    print("=" * 60)

    prototype_info = detector.get_prototype_info()
    for label, info in sorted(prototype_info.items()):
        print(f"\n{label.upper()}:")
        print(f"  Prototype norm: {info['prototype_norm']:.4f}")
        print(f"  Prototype mean: {info['prototype_mean']:.4f}")
        print(f"  Prototype std: {info['prototype_std']:.4f}")

    # Test entities
    test_entities = [
        ("machine learning", "Known ML term"),
        ("computer vision", "Known CV term"),
        ("text classification", "Related to NLP"),
        ("robotics automation", "Novel domain"),
        ("organic farming", "Novel domain"),
        ("reinforcement learning", "Related to ML"),
    ]

    print("\n" + "=" * 60)
    print("Novelty Detection Results:")
    print("=" * 60)

    for entity, description in test_entities:
        is_novel, distance, nearest_label = detector.is_novel(entity)

        status = "NOVEL" if is_novel else "KNOWN"
        print(f"\nEntity: {entity}")
        print(f"  Status: {status}")
        print(f"  Distance to nearest prototype: {distance:.4f}")
        print(f"  Nearest class: {nearest_label}")
        print(f"  Description: {description}")

    # Batch scoring example
    print("\n" + "=" * 60)
    print("Batch Scoring Example:")
    print("=" * 60)

    batch_entities = [
        "machine learning",
        "robotics",
        "computer vision",
        "agriculture",
        "natural language processing",
    ]

    results = detector.score_batch(batch_entities)

    print(f"\nProcessed {len(batch_entities)} entities:")
    for entity, (is_novel, distance, nearest_label) in zip(
        batch_entities, results, strict=False
    ):
        status = "NOVEL" if is_novel else "KNOWN"
        print(
            f"  {entity:30s} -> {status:8s} (dist: {distance:.3f}, class: {nearest_label})"
        )

    # Compare distance metrics
    print("\n" + "=" * 60)
    print("Comparing Distance Metrics:")
    print("=" * 60)

    test_entity = "machine learning algorithms"

    for metric in ["cosine", "euclidean"]:
        detector_metric = PrototypicalNoveltyDetector(
            distance_metric=metric,
            distance_threshold=0.5 if metric == "cosine" else 1.0,
        )
        detector_metric.train(training_data, show_progress=False)

        is_novel, distance, nearest_label = detector_metric.is_novel(test_entity)
        print(f"\n{metric.upper()} metric:")
        print(f"  Entity: {test_entity}")
        print(f"  Distance: {distance:.4f}")
        print(f"  Status: {'NOVEL' if is_novel else 'KNOWN'}")

    # Save and load example
    print("\n" + "=" * 60)
    print("Model Persistence Example:")
    print("=" * 60)

    save_path = "./models/prototypical_novelty"

    # Save
    print(f"\nSaving model to: {save_path}")
    detector.save(save_path)
    print("Model saved successfully!")

    # Load
    print(f"\nLoading model from: {save_path}")
    loaded_detector = PrototypicalNoveltyDetector.load(save_path)
    print("Model loaded successfully!")

    # Verify loaded model works
    is_novel, distance, nearest_label = loaded_detector.is_novel("machine learning")
    print("\nTest with loaded model:")
    print("  Entity: 'machine learning'")
    print(f"  Status: {'NOVEL' if is_novel else 'KNOWN'}")
    print(f"  Distance: {distance:.4f}")
    print(f"  Nearest class: {nearest_label}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
