"""
Country Code Semantic Categorization Experiment

This script demonstrates training and evaluating a semantic classifier
for mapping country names/variations to ISO 3166-1 alpha-2 country codes.

Three scenarios are compared:
  A) No training - zero-shot using embedding similarity
  B) SetFit with only classifier head (faster training)
  C) Full SetFit training (already done)

Usage:
    uv run python experiments/country_classifier/country_classifier.py
"""

import sys
from pathlib import Path

# Support running from the repo without manually setting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report  # noqa: E402
from semanticmatcher.core.classifier import SetFitClassifier  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "country_training_data.csv"
TRAIN_RATIO = 0.8
MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load training data from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Labels: {sorted(df['label'].unique())}")
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split data into train/test sets, stratified by label."""
    train_df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(frac=train_ratio, random_state=42))
        .reset_index(drop=True)
    )

    test_df = df[~df.index.isin(train_df.index)].reset_index(drop=True)

    print(f"\nTrain: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")

    return train_df, test_df


def scenario_a_zero_shot(
    train_df: pd.DataFrame, test_data: list[dict], labels: list[str]
):
    """
    Scenario A: Zero-shot classification using embedding similarity.
    No training required - uses label names as reference embeddings.
    """
    print("\n" + "=" * 60)
    print("SCENARIO A: ZERO-SHOT (No Training)")
    print("=" * 60)
    print("Method: Compare test embeddings to label name embeddings")

    model = SentenceTransformer(MODEL_NAME)

    label_embeddings = model.encode(labels)

    texts = [item["text"] for item in test_data]
    true_labels = [item["label"] for item in test_data]

    text_embeddings = model.encode(texts)

    similarities = cosine_similarity(text_embeddings, label_embeddings)
    predictions = [labels[idx] for idx in similarities.argmax(axis=1)]

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.2%}")

    print("\nSample predictions:")
    for text, true, pred in zip(texts[:10], true_labels[:10], predictions[:10]):
        status = "✓" if true == pred else "✗"
        print(f"  {status} '{text}' → {pred} (expected: {true})")

    return accuracy, predictions


def scenario_b_head_only(
    train_data: list[dict], test_data: list[dict], labels: list[str]
):
    """
    Scenario B: SetFit with only classifier head training.
    Uses pre-trained embeddings, trains only the classification head.
    """
    print("\n" + "=" * 60)
    print("SCENARIO B: SETFIT HEAD-ONLY (Fast Training)")
    print("=" * 60)
    print("Method: Freeze embeddings, train only classification head")

    clf = SetFitClassifier(
        labels=labels,
        model_name=MODEL_NAME,
        num_epochs=2,
        batch_size=16,
    )

    clf.train(train_data)

    texts = [item["text"] for item in test_data]
    true_labels = [item["label"] for item in test_data]

    predictions = clf.predict(texts)

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.2%}")

    print("\nSample predictions:")
    for text, true, pred in zip(texts[:10], true_labels[:10], predictions[:10]):
        status = "✓" if true == pred else "✗"
        print(f"  {status} '{text}' → {pred} (expected: {true})")

    return accuracy, predictions


def scenario_c_full(train_data: list[dict], test_data: list[dict], labels: list[str]):
    """
    Scenario C: Full SetFit training.
    Trains both embeddings and classifier head.
    """
    print("\n" + "=" * 60)
    print("SCENARIO C: SETFIT FULL TRAINING")
    print("=" * 60)
    print("Method: Fine-tune embeddings AND train classification head")

    clf = SetFitClassifier(
        labels=labels,
        model_name=MODEL_NAME,
        num_epochs=4,
        batch_size=16,
    )

    clf.train(train_data)

    texts = [item["text"] for item in test_data]
    true_labels = [item["label"] for item in test_data]

    predictions = clf.predict(texts)

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.2%}")

    print("\nSample predictions:")
    for text, true, pred in zip(texts[:10], true_labels[:10], predictions[:10]):
        status = "✓" if true == pred else "✗"
        print(f"  {status} '{text}' → {pred} (expected: {true})")

    return accuracy, predictions


def print_comparison(results: dict):
    """Print comparison of all scenarios."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<35} {'Accuracy':>10} {'Training Time':>15}")
    print("-" * 60)
    print(
        f"{'A) Zero-shot (no training)':<35} {results['A']['accuracy']:>9.2%} {'~0s':>15}"
    )
    print(
        f"{'B) Head-only (2 epochs)':<35} {results['B']['accuracy']:>9.2%} {'~30s':>15}"
    )
    print(
        f"{'C) Full training (4 epochs)':<35} {results['C']['accuracy']:>9.2%} {'~3min':>15}"
    )
    print("-" * 60)

    best = max(results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\nBest: {best[0]}) {best[1]['accuracy']:.2%}")

    print("\n" + "-" * 40)
    print("Trade-offs:")
    print("-" * 40)
    print("A) Zero-shot: Fastest, no GPU needed, lower accuracy")
    print("B) Head-only: Fast training, good accuracy, less GPU")
    print("C) Full training: Best accuracy, longer training, more GPU")


def main():
    print("=" * 60)
    print("COUNTRY CODE SEMANTIC CATEGORIZATION")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)

    df = load_data(DATA_PATH)

    train_df, test_df = split_data(df, TRAIN_RATIO)

    train_data = train_df.to_dict("records")
    test_data = test_df.to_dict("records")
    labels = sorted(df["label"].unique())

    print(f"\nTotal labels: {len(labels)}")

    results = {}

    acc_a, preds_a = scenario_a_zero_shot(train_df, test_data, labels)
    results["A"] = {"accuracy": acc_a, "name": "Zero-shot", "predictions": preds_a}

    acc_b, preds_b = scenario_b_head_only(train_data, test_data, labels)
    results["B"] = {"accuracy": acc_b, "name": "Head-only", "predictions": preds_b}

    acc_c, preds_c = scenario_c_full(train_data, test_data, labels)
    results["C"] = {"accuracy": acc_c, "name": "Full training", "predictions": preds_c}

    print_comparison(results)

    print("\n" + "=" * 60)
    print("DETAILED RESULTS - Scenario C (Full)")
    print("=" * 60)

    true_labels = [item["label"] for item in test_data]

    print("\nClassification Report:")
    print(classification_report(true_labels, results["C"].get("predictions", [])))

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total training samples: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")
    print(f"Country codes: {len(labels)}")
    print(f"Data file: {DATA_PATH}")


if __name__ == "__main__":
    main()
