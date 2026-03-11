"""
Country matching example using SemanticMatcher with SetFit.

This example demonstrates training a model specifically for
matching country names to ISO 3166-1 alpha-2 codes.
"""

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

# Extended training data for country matching
TRAIN_DATA = [
    # Germany
    {"text": "Germany", "label": "DE"},
    {"text": "Deutschland", "label": "DE"},
    {"text": "Deutchland", "label": "DE"},
    {"text": "Bundesrepublik Deutschland", "label": "DE"},
    {"text": "GER", "label": "DE"},
    # France
    {"text": "France", "label": "FR"},
    {"text": "French Republic", "label": "FR"},
    {"text": "La France", "label": "FR"},
    {"text": "FRA", "label": "FR"},
    # United States
    {"text": "United States", "label": "US"},
    {"text": "USA", "label": "US"},
    {"text": "America", "label": "US"},
    {"text": "U.S.A.", "label": "US"},
    {"text": "U.S.", "label": "US"},
    # United Kingdom
    {"text": "United Kingdom", "label": "GB"},
    {"text": "UK", "label": "GB"},
    {"text": "Britain", "label": "GB"},
    {"text": "Great Britain", "label": "GB"},
    {"text": "England", "label": "GB"},
    # Japan
    {"text": "Japan", "label": "JP"},
    {"text": "Nippon", "label": "JP"},
    {"text": "Nihon", "label": "JP"},
    # China
    {"text": "China", "label": "CN"},
    {"text": "People's Republic of China", "label": "CN"},
    {"text": "PRC", "label": "CN"},
    # Canada
    {"text": "Canada", "label": "CA"},
    {"text": "Canadia", "label": "CA"},
]

LABELS = ["DE", "FR", "US", "GB", "JP", "CN", "CA"]


def train_country_matcher(model_name: str = "paraphrase-mpnet-base-v2"):
    """Train a country matching model."""

    dataset = Dataset.from_list(TRAIN_DATA)

    model = SetFitModel.from_pretrained(
        f"sentence-transformers/{model_name}", labels=LABELS
    )

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

    return model


def demo():
    """Demonstrate country matching."""
    print("Training country matcher...")
    model = train_country_matcher()

    test_cases = [
        "Deutchland",
        "U.S.A.",
        "Britain",
        "Nippon",
        "PRC",
        "Canadia",
    ]

    print("\nPredictions:")
    predictions = model.predict(test_cases)
    for text, pred in zip(test_cases, predictions):
        print(f"  {text:20} → {pred}")


if __name__ == "__main__":
    demo()
