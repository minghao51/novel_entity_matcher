"""
Example: SetFit Contrastive Novelty Detection

Demonstrates how to use the SetFitNoveltyDetector to detect
novel entities using SetFit with contrastive learning.
"""

from novelentitymatcher.novelty.strategies.setfit_novelty import SetFitNoveltyDetector


def main():
    """Demonstrate SetFit novelty detection."""

    # Known entities (technology companies)
    known_entities = [
        "Apple Inc",
        "Microsoft Corporation",
        "Google LLC",
        "Amazon.com Inc",
        "Meta Platforms Inc",
        "Tesla Inc",
        "Netflix Inc",
        "NVIDIA Corporation",
        "Adobe Inc",
        "Intel Corporation",
        "Advanced Micro Devices",
        "Qualcomm Incorporated",
        "Texas Instruments",
        "Broadcom Inc",
        "Cisco Systems",
    ]

    print("Known entities (technology companies):")
    for entity in known_entities[:5]:
        print(f"  - {entity}")
    print(f"  ... and {len(known_entities) - 5} more")

    # Generate synthetic novel examples for training
    print("\n" + "=" * 60)
    print("Generating synthetic novel examples...")
    print("=" * 60)

    # First create a temporary detector to generate synthetic novels
    temp_detector = SetFitNoveltyDetector(
        known_entities=known_entities,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    synthetic_novels = temp_detector.generate_synthetic_novels(
        num_samples=20,
        augmentation_methods=["typos", "case_change", "spacing", "substring"],
    )

    print(f"\nGenerated {len(synthetic_novels)} synthetic novel examples:")
    for novel in synthetic_novels[:5]:
        print(f"  - {novel}")
    print(f"  ... and {len(synthetic_novels) - 5} more")

    # Initialize and train the detector
    print("\n" + "=" * 60)
    print("Training SetFit detector with contrastive learning...")
    print("=" * 60)

    detector = SetFitNoveltyDetector(
        known_entities=known_entities,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        num_epochs=4,  # More epochs for better training
        margin=0.5,
    )

    detector.train(
        synthetic_novels=synthetic_novels,
        show_progress=True,
    )

    print(f"\nCalibrated novelty threshold: {detector.novelty_threshold:.4f}")

    # Test entities
    test_entities = [
        ("Apple Inc", "Known company - low novelty"),
        ("Microsoft Corp", "Slightly different - low-medium novelty"),
        ("OpenAI LP", "Novel company - high novelty"),
        "SpaceX",  # Novel tech company
        ("Ford Motor Company", "Different industry - high novelty"),
        ("Samsung Electronics", "Known tech company - low novelty"),
    ]

    print("\n" + "=" * 60)
    print("Novelty Detection Results:")
    print("=" * 60)

    for item in test_entities:
        if isinstance(item, tuple):
            entity, description = item
        else:
            entity = item
            description = ""

        is_novel, confidence = detector.is_novel(entity)

        status = "NOVEL" if is_novel else "KNOWN"
        print(f"\nEntity: {entity}")
        print(f"  Status: {status}")
        print(f"  Confidence: {confidence:.3f}")
        if description:
            print(f"  Description: {description}")

    # Batch scoring example
    print("\n" + "=" * 60)
    print("Batch Scoring Example:")
    print("=" * 60)

    batch_entities = [
        "Apple Inc",
        "OpenAI",
        "Microsoft Corporation",
        "Toyota Motor Corp",
        "Google LLC",
    ]

    results = detector.score_batch(batch_entities)

    print(f"\nProcessed {len(batch_entities)} entities:")
    for entity, (is_novel, confidence) in zip(batch_entities, results, strict=False):
        status = "NOVEL" if is_novel else "KNOWN"
        print(f"  {entity:30s} -> {status:8s} (conf: {confidence:.3f})")

    # Data augmentation examples
    print("\n" + "=" * 60)
    print("Data Augmentation Examples:")
    print("=" * 60)

    original = "Microsoft Corporation"
    print(f"\nOriginal: {original}")
    print(f"  With typos: {detector._add_typos(original)}")
    print(f"  Case change: {detector._change_case(original)}")
    print(f"  Modified spacing: {detector._modify_spacing(original)}")
    print(f"  Substring variant: {detector._create_substring_variant(original)}")

    # Save and load example
    print("\n" + "=" * 60)
    print("Model Persistence Example:")
    print("=" * 60)

    save_path = "./models/setfit_novelty"

    # Save
    print(f"\nSaving model to: {save_path}")
    detector.save(save_path)
    print("Model saved successfully!")

    # Load
    print(f"\nLoading model from: {save_path}")
    loaded_detector = SetFitNoveltyDetector.load(save_path)
    print("Model loaded successfully!")

    # Verify loaded model works
    is_novel, confidence = loaded_detector.is_novel("Apple Inc")
    print("\nTest with loaded model:")
    print("  Entity: 'Apple Inc'")
    print(f"  Status: {'NOVEL' if is_novel else 'KNOWN'}")
    print(f"  Confidence: {confidence:.3f}")

    # Compare with different margins
    print("\n" + "=" * 60)
    print("Comparing Different Margin Values:")
    print("=" * 60)

    test_entity = "OpenAI LP"

    for margin in [0.3, 0.5, 0.7]:
        detector_margin = SetFitNoveltyDetector(
            known_entities=known_entities,
            margin=margin,
        )
        detector_margin.train(show_progress=False)

        is_novel, confidence = detector_margin.is_novel(test_entity)
        print(f"\nMargin = {margin}:")
        print(f"  Entity: {test_entity}")
        print(f"  Status: {'NOVEL' if is_novel else 'KNOWN'}")
        print(f"  Confidence: {confidence:.3f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
