"""
Basic usage example for SemanticMatcher with SetFit.

This example demonstrates how to train an entity matcher using SetFit
for few-shot text-to-entity matching.
"""

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments


def basic_entity_matching():
    """Train a simple entity matching model."""

    # 1. Prepare training data (text → entity label)
    train_data = [
        # Germany variants
        {"text": "Germany", "label": "DE"},
        {"text": "Deutschland", "label": "DE"},
        {"text": "Deutchland", "label": "DE"},
        {"text": "Bundesrepublik Deutschland", "label": "DE"},
        # France variants
        {"text": "France", "label": "FR"},
        {"text": "French Republic", "label": "FR"},
        {"text": "La France", "label": "FR"},
        # United States variants
        {"text": "United States", "label": "US"},
        {"text": "USA", "label": "US"},
        {"text": "America", "label": "US"},
        {"text": "U.S.A.", "label": "US"},
    ]

    # 2. Create dataset
    dataset = Dataset.from_list(train_data)

    # 3. Load SetFit model
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2", labels=["DE", "FR", "US"]
    )

    # 4. Train
    args = TrainingArguments(
        num_epochs=4,
        batch_size=16,
        eval_strategy="no",  # Disable evaluation for this demo (no eval_dataset provided)
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()

    # 5. Predict
    test_queries = ["Deutchland", "U.S.A.", "French Republic", "America"]

    predictions = model.predict(test_queries)

    for query, pred in zip(test_queries, predictions):
        print(f"{query:20} → {pred}")

    # Output:
    # Deutchland            → DE
    # U.S.A.                → US
    # French Republic       → FR
    # America               → US


if __name__ == "__main__":
    basic_entity_matching()
