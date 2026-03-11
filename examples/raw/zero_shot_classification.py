"""
Zero-shot classification example using SetFit.

SetFit can be used for few-shot text classification.
While it requires training examples, it needs far fewer than
traditional approaches (8-16 per class vs thousands).
"""

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments


def sentiment_classification():
    """Train a sentiment classifier with few examples."""

    train_data = [
        # Positive
        {"text": "I love this product!", "label": "positive"},
        {"text": "Great service, highly recommend", "label": "positive"},
        {"text": "Amazing quality", "label": "positive"},
        {"text": "Best purchase ever", "label": "positive"},
        {"text": "Fantastic experience", "label": "positive"},
        # Negative
        {"text": "Terrible, would not recommend", "label": "negative"},
        {"text": "Very disappointed", "label": "negative"},
        {"text": "Worst purchase", "label": "negative"},
        {"text": "Poor quality", "label": "negative"},
        {"text": "Not worth the money", "label": "negative"},
    ]

    dataset = Dataset.from_list(train_data)

    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        labels=["positive", "negative"],
    )

    args = TrainingArguments(num_epochs=4, batch_size=16)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    # Test
    test_cases = [
        "I hate this!",
        "So pleased with my order",
        "What a waste of money",
    ]

    predictions = model.predict(test_cases)
    for text, pred in zip(test_cases, predictions):
        print(f"{text:30} → {pred}")


def intent_classification():
    """Train an intent classifier for customer service."""

    train_data = [
        # Refund
        {"text": "I want my money back", "label": "refund"},
        {"text": "Can I cancel my order?", "label": "refund"},
        {"text": "I'd like a refund please", "label": "refund"},
        # Support
        {"text": "How do I reset my password?", "label": "support"},
        {"text": "I can't login to my account", "label": "support"},
        {"text": "Help me with setup", "label": "support"},
        # Complaint
        {"text": "This is unacceptable!", "label": "complaint"},
        {"text": "I'm very angry about this", "label": "complaint"},
        {"text": "Your service is terrible", "label": "complaint"},
        # Praise
        {"text": "Thank you so much!", "label": "praise"},
        {"text": "I love your product", "label": "praise"},
        {"text": "Great job team!", "label": "praise"},
    ]

    dataset = Dataset.from_list(train_data)

    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        labels=["refund", "support", "complaint", "praise"],
    )

    args = TrainingArguments(num_epochs=4, batch_size=16)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    test_cases = [
        "I want my money back!",
        "I need help with my account",
        "This is the best service ever",
    ]

    predictions = model.predict(test_cases)
    for text, pred in zip(test_cases, predictions):
        print(f"{text:30} → {pred}")


if __name__ == "__main__":
    sentiment_classification()
