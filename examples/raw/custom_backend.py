"""
Custom backend example: Using different Sentence Transformer models.

This example shows how to use different models with SetFit for
various use cases including multilingual support.
"""

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments


def multilingual_example():
    """Use a multilingual model for entity matching."""

    # Training data with multiple languages
    train_data = [
        {"text": "Germany", "label": "DE"},
        {"text": "Deutschland", "label": "DE"},
        {"text": "Allemagne", "label": "DE"},
        {"text": "Alemania", "label": "DE"},
        {"text": "德国", "label": "DE"},
        {"text": "ドイツ", "label": "DE"},
        {"text": "Japan", "label": "JP"},
        {"text": "Japon", "label": "JP"},
        {"text": "Japón", "label": "JP"},
        {"text": "日本", "label": "JP"},
    ]

    dataset = Dataset.from_list(train_data)

    # Use multilingual model
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        labels=["DE", "JP"],
    )

    args = TrainingArguments(num_epochs=4, batch_size=16)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    # Test cross-language matching
    test_cases = [
        "德国",  # Chinese
        " 日本",  # Japanese
        "Allemagne",  # French
    ]

    predictions = model.predict(test_cases)
    for text, pred in zip(test_cases, predictions):
        print(f"{text:10} → {pred}")


def small_model_example():
    """Use a smaller/faster model for development."""

    train_data = [
        {"text": "Germany", "label": "DE"},
        {"text": "France", "label": "FR"},
        {"text": "Japan", "label": "JP"},
    ]

    dataset = Dataset.from_list(train_data)

    # Use smaller model for faster training
    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", labels=["DE", "FR", "JP"]
    )

    args = TrainingArguments(num_epochs=4, batch_size=16)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    print(model.predict(["Germany"]))


def large_model_example():
    """Use a larger model for higher accuracy."""

    train_data = [
        {"text": "Germany", "label": "DE"},
        {"text": "France", "label": "FR"},
    ]

    dataset = Dataset.from_list(train_data)

    # Use larger model
    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2", labels=["DE", "FR"]
    )

    args = TrainingArguments(num_epochs=4, batch_size=16)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    print(model.predict(["Deutchland"]))


if __name__ == "__main__":
    multilingual_example()
