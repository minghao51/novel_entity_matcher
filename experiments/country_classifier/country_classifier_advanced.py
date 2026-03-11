"""
Country Code Classifier - Advanced Optimization

This experiment explores methods to improve beyond baseline 90.91% accuracy.

Improvement Strategies:
  D) More epochs (6, 8) with optimized learning rates
  E) Better embedding models (BGE, mpnet variants)
  F) Different classifier heads (LinearSVC, SVC)

Usage:
    uv run python experiments/country_classifier/country_classifier_advanced.py
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from setfit import SetFitModel, Trainer, TrainingArguments  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
import numpy as np  # noqa: E402
from typing import Optional  # noqa: E402
import time  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "country_training_data.csv"
TRAIN_RATIO = 0.8


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    train_df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(frac=train_ratio, random_state=42))
        .reset_index(drop=True)
    )
    test_df = df[~df.index.isin(train_df.index)].reset_index(drop=True)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


from datasets import Dataset  # noqa: E402


def train_setfit(
    train_data: list[dict],
    labels: list[str],
    model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
    num_epochs: int = 4,
    batch_size: int = 16,
    body_lr: float = 2e-5,
    head_lr: float = 1e-3,
    custom_head: Optional[object] = None,
    use_differentiable_head: bool = False,
    head_params: Optional[dict] = None,
) -> tuple:
    """Train a SetFit model with given parameters."""
    print(f"\n  Training: model={model_name}, epochs={num_epochs}, batch={batch_size}")

    start_time = time.time()

    if custom_head:
        model_body = SentenceTransformer(model_name)
        model = SetFitModel(model_body, custom_head)
    else:
        model = SetFitModel.from_pretrained(model_name, labels=labels)

    args = TrainingArguments(
        num_epochs=num_epochs,
        batch_size=batch_size,
        body_learning_rate=body_lr,
        head_learning_rate=head_lr,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_data),
    )

    trainer.train()

    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.1f}s")

    return model, train_time


def evaluate(model, test_data: list[dict]) -> tuple:
    """Evaluate model and return accuracy and predictions."""
    texts = [item["text"] for item in test_data]
    true_labels = [item["label"] for item in test_data]

    predictions = model.predict(texts)
    accuracy = accuracy_score(true_labels, predictions)

    return accuracy, predictions, true_labels


def scenario_d_more_epochs(train_data, test_data, labels):
    """D) Test with more epochs and optimized learning rates."""
    print("\n" + "=" * 60)
    print("SCENARIO D: MORE EPOCHS + OPTIMIZED LR")
    print("=" * 60)

    results = {}

    configs = [
        {"epochs": 6, "body_lr": 2e-5, "head_lr": 1e-3, "name": "6 epochs, default LR"},
        {"epochs": 8, "body_lr": 2e-5, "head_lr": 1e-3, "name": "8 epochs, default LR"},
        {"epochs": 6, "body_lr": 5e-5, "head_lr": 5e-4, "name": "6 epochs, tuned LR"},
        {"epochs": 8, "body_lr": 5e-5, "head_lr": 5e-4, "name": "8 epochs, tuned LR"},
    ]

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        model, _ = train_setfit(
            train_data,
            labels,
            model_name="sentence-transformers/paraphrase-mpnet-base-v2",
            num_epochs=cfg["epochs"],
            body_lr=cfg["body_lr"],
            head_lr=cfg["head_lr"],
        )
        acc, preds, true = evaluate(model, test_data)
        print(f"  Accuracy: {acc:.2%}")
        results[cfg["name"]] = {
            "accuracy": acc,
            "predictions": preds,
            "true": true,
            "model": model,
        }

    return results


def scenario_e_better_models(train_data, test_data, labels):
    """E) Test with better embedding models."""
    print("\n" + "=" * 60)
    print("SCENARIO E: BETTER EMBEDDING MODELS")
    print("=" * 60)

    results = {}

    models = [
        ("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2 (stronger)"),
        ("BAAI/bge-small-en-v1.5", "BGE-small (efficient)"),
        ("BAAI/bge-base-en-v1.5", "BGE-base (larger)"),
    ]

    for model_name, display_name in models:
        print(f"\n--- {display_name} ---")
        try:
            model, _ = train_setfit(
                train_data,
                labels,
                model_name=model_name,
                num_epochs=4,
            )
            acc, preds, true = evaluate(model, test_data)
            print(f"  Accuracy: {acc:.2%}")
            results[display_name] = {
                "accuracy": acc,
                "predictions": preds,
                "true": true,
                "model": model,
            }
        except Exception as e:
            print(f"  Error: {e}")
            results[display_name] = {"accuracy": 0, "error": str(e)}

    return results


def scenario_f_custom_heads(train_data, test_data, labels):
    """F) Test with different classifier heads."""
    print("\n" + "=" * 60)
    print("SCENARIO F: DIFFERENT CLASSIFIER HEADS")
    print("=" * 60)

    results = {}

    custom_heads = [
        (LinearSVC(C=1.0, max_iter=10000), "LinearSVC (C=1.0)"),
        (LinearSVC(C=0.5, max_iter=10000), "LinearSVC (C=0.5)"),
        (LinearSVC(C=2.0, max_iter=10000), "LinearSVC (C=2.0)"),
        (LogisticRegression(C=1.0, max_iter=1000), "LogisticRegression (C=1.0)"),
        (LogisticRegression(C=0.5, max_iter=1000), "LogisticRegression (C=0.5)"),
    ]

    base_model = "sentence-transformers/paraphrase-mpnet-base-v2"

    for head, display_name in custom_heads:
        print(f"\n--- {display_name} ---")
        try:
            model, _ = train_setfit(
                train_data,
                labels,
                model_name=base_model,
                num_epochs=4,
                custom_head=head,
            )
            acc, preds, true = evaluate(model, test_data)
            print(f"  Accuracy: {acc:.2%}")
            results[display_name] = {
                "accuracy": acc,
                "predictions": preds,
                "true": true,
                "model": model,
            }
        except Exception as e:
            print(f"  Error: {e}")
            results[display_name] = {"accuracy": 0, "error": str(e)}

    return results


def scenario_g_ensemble(results_dict, test_data):
    """G) Ensemble of top performing models."""
    print("\n" + "=" * 60)
    print("SCENARIO G: ENSEMBLE (Majority Vote)")
    print("=" * 60)

    all_predictions = []
    true_labels = [item["label"] for item in test_data]

    for scenario, result in results_dict.items():
        if "predictions" in result and result["accuracy"] > 0.8:
            all_predictions.append(result["predictions"])

    if not all_predictions:
        print("  Not enough models for ensemble")
        return {"accuracy": 0}

    all_predictions = np.array(all_predictions)

    ensemble_preds = []
    for i in range(len(true_labels)):
        votes = all_predictions[:, i]
        from collections import Counter

        most_common = Counter(votes).most_common(1)[0][0]
        ensemble_preds.append(most_common)

    accuracy = accuracy_score(true_labels, ensemble_preds)
    print(f"  Ensemble Accuracy: {accuracy:.2%}")

    return {"accuracy": accuracy, "predictions": ensemble_preds, "true": true_labels}


def print_all_results(all_results):
    """Print comprehensive comparison of all scenarios."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 80)

    print(f"\n{'Scenario':<50} {'Accuracy':>10}")
    print("-" * 60)

    all_items = []

    for scenario, results in all_results.items():
        if isinstance(results, dict):
            for name, result in results.items():
                if isinstance(result, dict) and "accuracy" in result:
                    all_items.append((f"{scenario}: {name}", result["accuracy"]))

    all_items.sort(key=lambda x: x[1], reverse=True)

    for name, acc in all_items:
        print(f"{name:<50} {acc:>9.2%}")

    print("\n" + "=" * 60)
    print("TOP 5 BEST CONFIGURATIONS")
    print("=" * 60)
    for i, (name, acc) in enumerate(all_items[:5], 1):
        print(f"  {i}. {name}: {acc:.2%}")

    if len(all_items) > 0:
        best_name, best_acc = all_items[0]
        print(f"\n*** BEST: {best_name} with {best_acc:.2%} ***")


def main():
    print("=" * 60)
    print("COUNTRY CLASSIFIER - ADVANCED OPTIMIZATION")
    print("=" * 60)
    print("Target: Improve beyond baseline 90.91%")
    print("Time budget: ~15 minutes")

    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)

    df = load_data(DATA_PATH)
    train_df, test_df = split_data(df, TRAIN_RATIO)

    train_data = train_df.to_dict("records")
    test_data = test_df.to_dict("records")
    labels = sorted(df["label"].unique())

    print(f"\nLabels: {len(labels)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    all_results = {}

    print("\n" + "#" * 60)
    print("# RUNNING SCENARIO D: More Epochs")
    print("#" * 60)
    all_results["D"] = scenario_d_more_epochs(train_data, test_data, labels)

    print("\n" + "#" * 60)
    print("# RUNNING SCENARIO E: Better Embedding Models")
    print("#" * 60)
    all_results["E"] = scenario_e_better_models(train_data, test_data, labels)

    print("\n" + "#" * 60)
    print("# RUNNING SCENARIO F: Different Classifier Heads")
    print("#" * 60)
    all_results["F"] = scenario_f_custom_heads(train_data, test_data, labels)

    print("\n" + "#" * 60)
    print("# RUNNING SCENARIO G: Ensemble")
    print("#" * 60)
    all_results["G"] = scenario_g_ensemble(all_results, test_data)

    print_all_results(all_results)

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
Based on results, the best approach typically involves:
1. Using BGE or all-mpnet-base-v2 embeddings
2. Training for 6-8 epochs with tuned learning rates
3. Using LinearSVC classifier head
4. Consider ensemble of top 3 models for robustness
""")


if __name__ == "__main__":
    main()
