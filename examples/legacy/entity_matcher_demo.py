"""
EntityMatcher Demo - Few-Shot Entity Matching with Training

This example demonstrates how to use EntityMatcher for few-shot text-to-entity
matching. EntityMatcher uses SetFit for training and requires labeled training data.

**Estimated Runtime**: 2-3 minutes (includes training time)

**What this demonstrates**:
- EntityMatcher initialization with entities
- Training with few-shot labeled data
- Single and batch prediction
- predict() vs predict_proba() usage
- Threshold parameter for confidence filtering
- Custom model selection

**When to use EntityMatcher**:
- You have labeled training examples (even just 3-5 per entity)
- You need higher accuracy on complex/ambiguous cases
- Text variations are significant (typos, abbreviations, translations)
- You can afford 1-3 minutes of training time
"""

from semanticmatcher import EntityMatcher


def main():
    """Demonstrate EntityMatcher workflow."""

    print("=" * 80)
    print("EntityMatcher Demo - Few-Shot Entity Matching")
    print("=" * 80)
    print()

    # 1. Define your entities (canonical entities with IDs)
    # These are the entities you want to match user queries to
    entities = [
        {
            "id": "DE",
            "name": "Germany",
            "aliases": ["Deutschland", "Bundesrepublik Deutschland"],
        },
        {
            "id": "FR",
            "name": "France",
            "aliases": ["République française", "La France"],
        },
        {
            "id": "US",
            "name": "United States",
            "aliases": ["USA", "United States of America", "America"],
        },
        {"id": "JP", "name": "Japan", "aliases": ["Nippon", "Nihon"]},
        {
            "id": "GB",
            "name": "United Kingdom",
            "aliases": ["UK", "Great Britain", "England"],
        },
    ]

    print(f"Entities defined: {len(entities)} countries")

    # 2. Prepare training data (text → entity label mappings)
    # You need just 3-5 examples per entity for good results
    training_data = [
        # Germany examples
        {"text": "Germany", "label": "DE"},
        {"text": "Deutschland", "label": "DE"},
        {"text": "Bundesrepublik Deutschland", "label": "DE"},
        {"text": "DE", "label": "DE"},
        # France examples
        {"text": "France", "label": "FR"},
        {"text": "French Republic", "label": "FR"},
        {"text": "République française", "label": "FR"},
        {"text": "La France", "label": "FR"},
        # United States examples
        {"text": "United States", "label": "US"},
        {"text": "USA", "label": "US"},
        {"text": "America", "label": "US"},
        {"text": "U.S.A.", "label": "US"},
        # Japan examples
        {"text": "Japan", "label": "JP"},
        {"text": "Nippon", "label": "JP"},
        {"text": "Nihon", "label": "JP"},
        # United Kingdom examples
        {"text": "United Kingdom", "label": "GB"},
        {"text": "UK", "label": "GB"},
        {"text": "Great Britain", "label": "GB"},
        {"text": "England", "label": "GB"},
    ]

    print(f"Training examples: {len(training_data)} labeled pairs")
    print()

    # 3. Initialize EntityMatcher
    # threshold=0.7 means predictions need 70% confidence
    # normalize=True handles lowercase, accents, punctuation
    print("Initializing EntityMatcher...")
    matcher = EntityMatcher(
        entities=entities,
        model_name="sentence-transformers/paraphrase-mpnet-base-v2",
        threshold=0.7,
        normalize=True,
    )
    print("✓ Matcher initialized")
    print()

    # 4. Train the model
    print("Training model (this may take 1-2 minutes)...")
    matcher.train(
        training_data=training_data,
        num_epochs=4,
        batch_size=16,
    )
    print("✓ Training complete")
    print()

    # 5. Make predictions on single queries
    print("=" * 80)
    print("Single Query Predictions")
    print("=" * 80)
    print()

    test_queries = [
        "Deutchland",  # Typo
        "U.S.A.",
        "République française",
        "Nippon",
        "Great Britain",
        "United Kingdom of Great Britain",  # Longer variant
    ]

    for query in test_queries:
        prediction = matcher.predict(query)
        print(f"Query: '{query}'")
        print(f"  → Predicted: {prediction}")
        print()

    # 6. Make batch predictions (more efficient for multiple queries)
    print("=" * 80)
    print("Batch Predictions")
    print("=" * 80)
    print()

    batch_queries = ["Deutschland", "USA", "Nihon", "UK"]
    predictions = matcher.predict(batch_queries)

    print("Batch results:")
    for query, pred in zip(batch_queries, predictions):
        print(f"  '{query}' → {pred}")
    print()

    # 7. Using predict_proba() to see confidence scores
    print("=" * 80)
    print("Probability Scores (predict_proba)")
    print("=" * 80)
    print()

    query = "United Kingdom"
    # Note: predict_proba is accessed through the internal classifier
    proba = matcher.classifier.predict_proba(query)

    print(f"Query: '{query}'")
    print("Confidence scores per entity:")
    for i, label in enumerate(matcher.classifier.labels):
        print(f"  {label}: {proba[i]:.3f}")
    print()

    # 8. Demonstrate threshold filtering
    print("=" * 80)
    print("Threshold Filtering")
    print("=" * 80)
    print()

    # Low-confidence query (ambiguous or unknown)
    ambiguous_queries = [
        "Kingdom",  # Ambiguous
        "United",  # Too generic
        "UnknownPlace",  # Unknown
    ]

    print("Queries with low confidence (below threshold return None):")
    for query in ambiguous_queries:
        prediction = matcher.predict(query)
        print(f"  '{query}' → {prediction}")
    print()

    # 9. Custom model selection
    print("=" * 80)
    print("Custom Model Selection")
    print("=" * 80)
    print()

    print("Available model aliases:")
    print("  - 'mpnet' (default): sentence-transformers/paraphrase-mpnet-base-v2")
    print("  - 'bge-base': BAAI/bge-base-en-v1.5 (English, fast)")
    print("  - 'bge-m3': BAAI/bge-m3 (multilingual)")
    print("  - 'minilm': sentence-transformers/all-MiniLM-L6-v2 (very fast)")
    print()
    print("You can also use full HuggingFace model names:")
    print("  model_name='sentence-transformers/LaBSE'  # Multilingual")
    print()


if __name__ == "__main__":
    main()
