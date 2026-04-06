"""Comprehensive benchmark comparing zero-shot vs head-only vs full SetFit."""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASETS = ["ag_news", "yahoo_answers_topics"]
MAX_TRAIN = 500
MAX_TEST = 1000

AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
YAHOO_CLASSES = [
    "Society",
    "Science",
    "Health",
    "Education",
    "Computer",
    "Sports",
    "Business",
    "Entertainment",
    "Music",
    "Family",
]

DATASET_CONFIG = {
    "ag_news": {"text_col": "text", "label_col": "label", "classes": AG_NEWS_CLASSES},
    "yahoo_answers_topics": {
        "text_col": "question_title",
        "label_col": "topic",
        "classes": YAHOO_CLASSES,
    },
}


def load_data(dataset_name):
    ds = load_dataset(dataset_name)
    cfg = DATASET_CONFIG[dataset_name]
    # Load more data and shuffle to get balanced classes
    train_full = pd.DataFrame(ds["train"][:])
    train_full = train_full.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = train_full.head(MAX_TRAIN)
    test_df = pd.DataFrame(ds["test"][:MAX_TEST])
    return train_df, test_df, cfg


def encode_texts(model, texts):
    return model.encode(texts, show_progress_bar=False)


def zero_shot_predict(model, texts, class_names):
    class_embs = encode_texts(model, class_names)
    text_embs = encode_texts(model, texts)
    sims = cosine_similarity(text_embs, class_embs)
    pred_indices = np.argmax(sims, axis=1)
    pred_labels = [class_names[i] for i in pred_indices]
    confidences = np.max(sims, axis=1)
    return pred_labels, confidences


def head_only_predict(matcher, texts):
    """Use the trained Matcher to predict."""
    results = matcher.match(texts, top_k=1)
    if isinstance(results, dict):
        results = [results]
    preds = [r["id"] if r else None for r in results]
    confs = [r["score"] if r else 0.0 for r in results]
    return preds, confs


def build_entities(class_names):
    return [{"id": name, "name": name, "aliases": []} for name in class_names]


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from novelentitymatcher.core.matcher import Matcher

    model = SentenceTransformer(MODEL_NAME)
    all_results = []

    for dataset_name in DATASETS:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 80}")

        train_df, test_df, cfg = load_data(dataset_name)
        class_names = cfg["classes"]
        text_col = cfg["text_col"]
        label_col = cfg["label_col"]

        # Create train/val/test splits (shuffle to get balanced classes)
        train_df_shuffled = train_df.sample(frac=1, random_state=42).reset_index(
            drop=True
        )
        val_size = min(200, len(train_df_shuffled) // 4)
        val_df = train_df_shuffled.tail(val_size)
        train_small_df = train_df_shuffled.head(len(train_df_shuffled) - val_size)

        train_texts = train_small_df[text_col].tolist()
        train_labels = [
            class_names[label] if isinstance(label, int) else label
            for label in train_small_df[label_col].tolist()
        ]
        val_texts = val_df[text_col].tolist()
        val_labels = [
            class_names[label] if isinstance(label, int) else label
            for label in val_df[label_col].tolist()
        ]
        test_texts = test_df[text_col].tolist()
        test_labels = [
            class_names[label] if isinstance(label, int) else label
            for label in test_df[label_col].tolist()
        ]

        print(
            f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}"
        )
        print(f"Classes ({len(class_names)}): {class_names}")

        # Zero-shot evaluation
        print("\n=== Zero-Shot ===")
        for split_name, split_texts, split_labels in [
            ("train", train_texts, train_labels),
            ("validation", val_texts, val_labels),
            ("test", test_texts, test_labels),
        ]:
            preds, confs = zero_shot_predict(model, split_texts, class_names)
            acc = accuracy_score(split_labels, preds)
            f1 = f1_score(split_labels, preds, average="macro")
            print(f"  {split_name:12s}: acc={acc:.4f}, f1={f1:.4f}")
            all_results.append(
                {
                    "dataset": dataset_name,
                    "mode": "zero-shot",
                    "split": split_name,
                    "num_samples": len(split_texts),
                    "accuracy": acc,
                    "macro_f1": f1,
                }
            )

        # Head-only and Full SetFit evaluation with different training sizes
        for train_size in [100, 200, 500]:
            actual_train_size = min(train_size, len(train_small_df))
            samples_per_class = max(1, actual_train_size // len(class_names))

            # Stratified sampling: get exactly samples_per_class from each class
            indices = []
            for cls_idx in range(len(class_names)):
                cls_name = class_names[cls_idx]
                cls_indices = [
                    i for i, label in enumerate(train_labels) if label == cls_name
                ]
                available = min(samples_per_class, len(cls_indices))
                indices.extend(cls_indices[:available])

            # If we don't have enough samples, pad with remaining from other classes
            if len(indices) < actual_train_size:
                remaining = [i for i in range(len(train_texts)) if i not in indices]
                np.random.seed(42)
                extra = np.random.choice(
                    remaining,
                    size=min(actual_train_size - len(indices), len(remaining)),
                    replace=False,
                )
                indices.extend(extra.tolist())

            sub_train_texts = [train_texts[i] for i in indices]
            sub_train_labels = [train_labels[i] for i in indices]

            # Build training data for Matcher
            training_data = [
                {"text": text, "label": label}
                for text, label in zip(sub_train_texts, sub_train_labels)
            ]
            entities = build_entities(class_names)

            # === Head-Only (skip_body_training=True) ===
            print(f"\n=== Head-Only (train={train_size}) ===")
            print(f"  Using {len(sub_train_texts)} training samples")
            print(
                f"  Label distribution: {pd.Series(sub_train_labels).value_counts().to_dict()}"
            )

            n = len(training_data)
            # Optimal PCA dims based on experimentation
            pca_dims = 5 if n <= 100 else (10 if n <= 200 else 20)

            matcher_ho = Matcher(
                entities=entities, model=MODEL_NAME, mode="head-only", threshold=0.0
            )
            matcher_ho.fit(
                training_data=training_data,
                mode="head-only",
                show_progress=False,
                skip_body_training=True,
                head_c=0.001 if n <= 100 else (0.01 if n <= 200 else 0.1),
                pca_dims=pca_dims,
            )

            train_acc = None
            for split_name, split_texts, split_labels in [
                ("train", sub_train_texts, sub_train_labels),
                ("validation", val_texts, val_labels),
                ("test", test_texts, test_labels),
            ]:
                preds, _confs = head_only_predict(matcher_ho, split_texts)
                acc = accuracy_score(split_labels, preds)
                f1 = f1_score(split_labels, preds, average="macro", zero_division=0)
                if split_name == "train":
                    train_acc = acc
                gap = (
                    train_acc - acc
                    if train_acc is not None and split_name != "train"
                    else 0
                )
                gap_str = f" (gap: {gap:+.4f})" if gap != 0 else ""
                print(f"  {split_name:12s}: acc={acc:.4f}, f1={f1:.4f}{gap_str}")
                all_results.append(
                    {
                        "dataset": dataset_name,
                        "mode": f"head-only-{train_size}",
                        "split": split_name,
                        "num_samples": len(split_texts),
                        "accuracy": acc,
                        "macro_f1": f1,
                    }
                )

            # === Full SetFit (contrastive + head training) ===
            print(f"\n=== Full SetFit (train={train_size}) ===")
            print(f"  Using {len(sub_train_texts)} training samples")

            # Full SetFit uses contrastive training with regularization
            n = len(training_data)
            # More conservative settings for multi-class
            num_iterations = max(1, min(5, n // (len(class_names) * 2)))
            weight_decay = 0.1 if n <= 100 else 0.01
            head_c = 1.0

            matcher_full = Matcher(
                entities=entities, model=MODEL_NAME, mode="full", threshold=0.0
            )
            matcher_full.fit(
                training_data=training_data,
                mode="full",
                show_progress=False,
                skip_body_training=False,
                head_c=head_c,
                weight_decay=weight_decay,
                num_iterations=num_iterations,
                num_epochs=1,
            )

            train_acc = None
            for split_name, split_texts, split_labels in [
                ("train", sub_train_texts, sub_train_labels),
                ("validation", val_texts, val_labels),
                ("test", test_texts, test_labels),
            ]:
                preds, _confs = head_only_predict(matcher_full, split_texts)
                acc = accuracy_score(split_labels, preds)
                f1 = f1_score(split_labels, preds, average="macro", zero_division=0)
                if split_name == "train":
                    train_acc = acc
                gap = (
                    train_acc - acc
                    if train_acc is not None and split_name != "train"
                    else 0
                )
                gap_str = f" (gap: {gap:+.4f})" if gap != 0 else ""
                print(f"  {split_name:12s}: acc={acc:.4f}, f1={f1:.4f}{gap_str}")
                all_results.append(
                    {
                        "dataset": dataset_name,
                        "mode": f"full-{train_size}",
                        "split": split_name,
                        "num_samples": len(split_texts),
                        "accuracy": acc,
                        "macro_f1": f1,
                    }
                )

    # Present results
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 100)
    print("FULL RESULTS TABLE")
    print("=" * 100)
    print(df.to_string(index=False))

    # Per-dataset comparison
    for dataset_name in DATASETS:
        ds_df = df[df["dataset"] == dataset_name]
        print(f"\n{'=' * 100}")
        print(f"ACCURACY - {dataset_name}")
        print(f"{'=' * 100}")
        acc_pivot = ds_df.pivot(index="split", columns="mode", values="accuracy")
        print(acc_pivot.to_string())

        print(f"\n{'=' * 100}")
        print(f"MACRO F1 - {dataset_name}")
        print(f"{'=' * 100}")
        f1_pivot = ds_df.pivot(index="split", columns="mode", values="macro_f1")
        print(f1_pivot.to_string())

    # Gap analysis
    print(f"\n{'=' * 100}")
    print("OVERFITTING GAP ANALYSIS (Train - Test)")
    print(f"{'=' * 100}")
    for dataset_name in DATASETS:
        ds_df = df[df["dataset"] == dataset_name]
        for mode in ds_df["mode"].unique():
            mode_df = ds_df[ds_df["mode"] == mode]
            train_acc = mode_df[mode_df["split"] == "train"]["accuracy"].values
            test_acc = mode_df[mode_df["split"] == "test"]["accuracy"].values
            if len(train_acc) > 0 and len(test_acc) > 0:
                gap = train_acc[0] - test_acc[0]
                print(
                    f"  {mode:25s}: gap = {gap:+.4f} ({train_acc[0]:.4f} -> {test_acc[0]:.4f})"
                )


if __name__ == "__main__":
    main()
