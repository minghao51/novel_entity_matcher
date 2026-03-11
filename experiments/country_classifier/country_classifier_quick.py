"""
Country Code Classifier - Quick Optimization Test

Quick test to improve beyond 90.91% baseline.

Usage:
    uv run python experiments/country_classifier/country_classifier_quick.py
"""

import sys
from pathlib import Path

# Support local imports if this script starts using project modules under src/.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from setfit import SetFitModel, Trainer, TrainingArguments  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from datasets import Dataset  # noqa: E402
import time  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "country_training_data.csv"
TRAIN_RATIO = 0.8


def load_and_split():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples")

    train_df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(frac=TRAIN_RATIO, random_state=42))
        .reset_index(drop=True)
    )
    test_df = df[~df.index.isin(train_df.index)].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return (
        train_df.to_dict("records"),
        test_df.to_dict("records"),
        sorted(df["label"].unique()),
    )


def train_and_eval(train_data, test_data, labels, name, model_name, epochs, head=None):
    print(f"\n--- {name} ---")
    start = time.time()

    if head:
        model_body = SentenceTransformer(model_name)
        model = SetFitModel(model_body, head)
    else:
        model = SetFitModel.from_pretrained(model_name, labels=labels)

    args = TrainingArguments(num_epochs=epochs, batch_size=16)
    trainer = Trainer(
        model=model, args=args, train_dataset=Dataset.from_list(train_data)
    )
    trainer.train()

    texts = [item["text"] for item in test_data]
    true_labels = [item["label"] for item in test_data]
    preds = model.predict(texts)
    acc = accuracy_score(true_labels, preds)

    print(f"  Accuracy: {acc:.2%} ({time.time() - start:.1f}s)")
    return acc, preds


def main():
    print("=" * 60)
    print("QUICK OPTIMIZATION TEST")
    print("=" * 60)

    train_data, test_data, labels = load_and_split()

    results = {}

    print("\n### BASELINE ###")
    acc, _ = train_and_eval(
        train_data,
        test_data,
        labels,
        "Baseline (4 epochs, mpnet)",
        "sentence-transformers/paraphrase-mpnet-base-v2",
        4,
    )
    results["Baseline"] = acc

    print("\n### SCENARIO D: More Epochs ###")
    acc, _ = train_and_eval(
        train_data,
        test_data,
        labels,
        "6 epochs, mpnet",
        "sentence-transformers/paraphrase-mpnet-base-v2",
        6,
    )
    results["6 epochs"] = acc

    print("\n### SCENARIO E: Better Embedding ###")
    acc, _ = train_and_eval(
        train_data,
        test_data,
        labels,
        "4 epochs, all-mpnet",
        "sentence-transformers/all-mpnet-base-v2",
        4,
    )
    results["all-mpnet"] = acc

    print("\n### SCENARIO F: LinearSVC Head ###")
    acc, _ = train_and_eval(
        train_data,
        test_data,
        labels,
        "4 epochs, mpnet + LinearSVC",
        "sentence-transformers/paraphrase-mpnet-base-v2",
        4,
        head=LinearSVC(C=1.0, max_iter=10000),
    )
    results["LinearSVC"] = acc

    print("\n### SCENARIO E+D: all-mpnet + 6 epochs ###")
    acc, _ = train_and_eval(
        train_data,
        test_data,
        labels,
        "6 epochs, all-mpnet",
        "sentence-transformers/all-mpnet-base-v2",
        6,
    )
    results["all-mpnet+6ep"] = acc

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<35} {'Accuracy':>10}")
    print("-" * 45)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for name, acc in sorted_results:
        print(f"{name:<35} {acc:>9.2%}")

    baseline = results["Baseline"]
    best_name, best_acc = sorted_results[0]
    improvement = best_acc - baseline
    print(f"\n*** BEST: {best_name} = {best_acc:.2%} (Δ{improvement:+.2%}) ***")
    print(f"Baseline was: {baseline:.2%}")


if __name__ == "__main__":
    main()
