"""
Model Persistence Demo - Save and Load Trained Models

This example demonstrates how to save and load trained models for production use.

**Estimated Runtime**: 3-4 minutes (includes training for demonstration)

**What this demonstrates**:
- Saving a trained SetFitClassifier to disk
- Loading a saved classifier
- Saving EntityMatcher models
- Model versioning patterns
- Error handling for missing models
- Production deployment considerations

**When to use model persistence**:
- Deploying to production
- Avoiding retraining time on every startup
- Sharing trained models across services
- Versioning and rollback capabilities
"""

import os
import shutil
from pathlib import Path
from semanticmatcher import EntityMatcher, SetFitClassifier


def main():
    """Demonstrate model persistence workflow."""

    print("=" * 80)
    print("Model Persistence Demo - Save and Load Trained Models")
    print("=" * 80)
    print()

    # Setup: Create a temporary directory for models
    model_dir = Path("./saved_models_demo")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir()

    print(f"Created model directory: {model_dir}")
    print()

    # =========================================================================
    # Part 1: Save and Load SetFitClassifier (Low-Level API)
    # =========================================================================
    print("=" * 80)
    print("Part 1: SetFitClassifier Save/Load")
    print("=" * 80)
    print()

    # 1. Train a classifier
    print("Step 1: Training a SetFitClassifier...")
    training_data = [
        {"text": "Germany", "label": "DE"},
        {"text": "Deutschland", "label": "DE"},
        {"text": "France", "label": "FR"},
        {"text": "Frankreich", "label": "FR"},
        {"text": "United States", "label": "US"},
        {"text": "USA", "label": "US"},
    ]

    classifier = SetFitClassifier(
        labels=["DE", "FR", "US"],
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster for demo
        num_epochs=3,  # Fewer epochs for demo
    )
    classifier.train(training_data)
    print("✓ Training complete")
    print()

    # 2. Test the classifier before saving
    print("Step 2: Testing predictions before saving...")
    test_query = "Deutschland"
    prediction_before = classifier.predict(test_query)
    print(f"  Query: '{test_query}' → {prediction_before}")
    print()

    # 3. Save the classifier
    model_path = model_dir / "country_classifier"
    print(f"Step 3: Saving classifier to: {model_path}")
    classifier.save(str(model_path))
    print("✓ Classifier saved")
    print(f"  Files created: {list(model_path.glob('*'))}")
    print()

    # 4. Load the classifier
    print("Step 4: Loading classifier from disk...")
    loaded_classifier = SetFitClassifier.load(str(model_path))
    print("✓ Classifier loaded")
    print()

    # 5. Verify the loaded classifier works
    print("Step 5: Verifying loaded classifier...")
    prediction_after = loaded_classifier.predict(test_query)
    print(f"  Query: '{test_query}' → {prediction_after}")
    print(f"  Predictions match: {prediction_before == prediction_after}")
    print()

    # =========================================================================
    # Part 2: EntityMatcher with Classifier Persistence
    # =========================================================================
    print("=" * 80)
    print("Part 2: EntityMatcher with Classifier Persistence")
    print("=" * 80)
    print()

    # 1. Train an EntityMatcher
    print("Step 1: Training EntityMatcher...")
    entities = [
        {"id": "DE", "name": "Germany"},
        {"id": "FR", "name": "France"},
        {"id": "US", "name": "United States"},
    ]

    matcher = EntityMatcher(
        entities=entities,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.7,
    )
    matcher.train(training_data, num_epochs=3)
    print("✓ EntityMatcher trained")
    print()

    # 2. Save the internal classifier
    matcher_model_path = model_dir / "entity_matcher_classifier"
    print("Step 2: Saving EntityMatcher's internal classifier...")
    matcher.classifier.save(str(matcher_model_path))
    print("✓ Classifier saved")
    print()

    # 3. Load into a new EntityMatcher
    print("Step 3: Creating new EntityMatcher and loading classifier...")
    new_matcher = EntityMatcher(
        entities=entities,
        threshold=0.7,
    )
    # Load the saved classifier
    new_matcher.classifier = SetFitClassifier.load(str(matcher_model_path))
    new_matcher.is_trained = True
    print("✓ New EntityMatcher created with loaded classifier")
    print()

    # 4. Verify it works
    print("Step 4: Testing loaded EntityMatcher...")
    test_queries = ["Deutschland", "USA", "Frankreich"]
    for query in test_queries:
        result = new_matcher.predict(query)
        print(f"  '{query}' → {result}")
    print()

    # =========================================================================
    # Part 3: Model Versioning Pattern
    # =========================================================================
    print("=" * 80)
    print("Part 3: Model Versioning Pattern")
    print("=" * 80)
    print()

    print("Recommended directory structure for versioned models:")
    print("""
    models/
    ├── country_classifier_v1/
    │   ├── config.json
    │   ├── model_head.pkl
    │   └── ...
    ├── country_classifier_v2/
    │   ├── config.json
    │   ├── model_head.pkl
    │   └── ...
    └── latest -> country_classifier_v2  # Symlink to latest version
    """)
    print()

    # Demonstrate versioned save
    print("Example: Saving model with version...")
    version = "v1"
    versioned_path = model_dir / f"country_classifier_{version}"
    classifier.save(str(versioned_path))
    print(f"✓ Model saved as: {versioned_path.name}")
    print()

    # =========================================================================
    # Part 4: Error Handling
    # =========================================================================
    print("=" * 80)
    print("Part 4: Error Handling")
    print("=" * 80)
    print()

    # 1. Handle missing model directory
    print("Example 1: Handling missing model directory...")
    missing_path = model_dir / "nonexistent_model"

    try:
        loaded = SetFitClassifier.load(str(missing_path))
    except Exception as e:
        print(f"  ✓ Error caught: {type(e).__name__}")
        print(f"    Message: {str(e)[:80]}...")
    print()

    # 2. Validate model before use
    print("Example 2: Validating loaded model...")

    def load_classifier_safely(path: str) -> SetFitClassifier | None:
        """Safely load a classifier with error handling."""
        try:
            if not os.path.exists(path):
                print(f"  ✗ Model path does not exist: {path}")
                return None

            classifier = SetFitClassifier.load(path)

            # Verify it's trained
            if not classifier.is_trained:
                print("  ✗ Loaded model is not trained")
                return None

            print(f"  ✓ Model loaded successfully from {path}")
            return classifier

        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            return None

    # Test with valid path
    loaded = load_classifier_safely(str(versioned_path))
    if loaded:
        print(f"    Model can make predictions: {loaded.predict('Germany')}")
    print()

    # =========================================================================
    # Part 5: Production Considerations
    # =========================================================================
    print("=" * 80)
    print("Part 5: Production Deployment Considerations")
    print("=" * 80)
    print()

    print("Key considerations for production deployment:")
    print()
    print("1. Model Storage:")
    print("   - Use persistent storage (S3, GCS, Azure Blob, or shared filesystem)")
    print("   - Include model metadata (training date, version, performance metrics)")
    print("   - Use environment variables for model paths")
    print()
    print("2. Model Loading:")
    print("   - Load models once at startup, not per-request")
    print("   - Use lazy loading for multiple models")
    print("   - Implement health checks for model availability")
    print()
    print("3. Model Versioning:")
    print("   - Track model versions in configuration")
    print("   - Support A/B testing with multiple model versions")
    print("   - Maintain rollback capability")
    print()
    print("4. Example production code structure:")
    print("""
    import os
    from semanticmatcher import SetFitClassifier

    class ModelService:
        def __init__(self):
            self.model_path = os.getenv('MODEL_PATH')
            self.classifier = None

        def load_model(self):
            self.classifier = SetFitClassifier.load(self.model_path)

        def predict(self, text: str) -> str:
            if not self.classifier:
                self.load_model()
            return self.classifier.predict(text)

    # Usage
    service = ModelService()
    service.load_model()  # Load once at startup
    prediction = service.predict("query")  # Fast inference
    """)
    print()

    # Cleanup
    print("=" * 80)
    print("Cleanup")
    print("=" * 80)
    print()
    print(f"Removing demo directory: {model_dir}")
    shutil.rmtree(model_dir)
    print("✓ Cleanup complete")
    print()

    print("Demo complete!")


if __name__ == "__main__":
    main()
