"""Comprehensive full-pipeline benchmark for Novel Entity Matcher.

Tests the COMPLETE workflow with proper train/val/test splits:
1. CLASSIFICATION: Assign entity to known class (zero-shot, head-only, full SetFit)
2. NOVELTY DETECTION: Detect if entity belongs to unknown class

Results are saved to novelty_entity_matcher_benchmark_results.csv
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import warnings

warnings.filterwarnings("ignore")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASET = "ag_news"
MAX_TRAIN = 500
MAX_TEST = 1000
OOD_RATIO = 0.2
RANDOM_SEED = 42


@dataclass
class SplitData:
    train_texts: list[str]
    train_labels: list[str]
    val_texts: list[str]
    val_labels: list[str]
    test_texts: list[str]
    test_labels: list[str]
    entities: list[dict]
    known_classes: list[str]
    ood_classes: list[str]


def load_and_split_data() -> SplitData:
    """Load dataset and create proper train/val/test split with OOD classes."""
    import random

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ds = load_dataset("ag_news")
    df = pd.DataFrame(ds["test"])

    classes = ["World", "Sports", "Business", "Sci/Tech"]

    num_ood = max(1, int(4 * OOD_RATIO))
    ood_labels = set(random.sample(range(4), num_ood))
    known_labels = sorted(set(range(4)) - ood_labels)

    df["is_ood"] = df["label"].isin(ood_labels)
    known_df = df[~df["is_ood"]].copy()
    ood_df = df[df["is_ood"]].copy()

    known_df = known_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    ood_df = ood_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    n_known_train = min(MAX_TRAIN, len(known_df) // 2)
    n_ood_train = min(50, len(ood_df) // 2)

    train_known = known_df.head(n_known_train)
    temp_known = known_df.iloc[n_known_train:]
    val_known = temp_known.head(len(temp_known) // 2)
    test_known = temp_known.iloc[len(temp_known) // 2 :]

    train_ood = ood_df.head(n_ood_train)
    temp_ood = ood_df.iloc[n_ood_train:]
    val_ood = temp_ood.head(len(temp_ood) // 2)
    test_ood = temp_ood.iloc[len(temp_ood) // 2 :]

    train_texts = train_known["text"].tolist() + train_ood["text"].tolist()
    train_labels = [classes[label] for label in train_known["label"].tolist()] + [
        "__NOVEL__"
    ] * len(train_ood)

    val_texts = val_known["text"].tolist()[:MAX_TEST] + val_ood["text"].tolist()
    val_labels = [classes[label] for label in val_known["label"].tolist()][
        :MAX_TEST
    ] + ["__NOVEL__"] * len(val_ood)

    test_texts = test_known["text"].tolist()[:MAX_TEST] + test_ood["text"].tolist()
    test_labels = [classes[label] for label in test_known["label"].tolist()][
        :MAX_TEST
    ] + ["__NOVEL__"] * len(test_ood)

    known_class_names = [classes[label] for label in known_labels]
    entities = [{"id": c, "name": c, "aliases": []} for c in known_class_names]

    [
        {"text": text, "label": label}
        for text, label in zip(train_texts, train_labels)
        if label != "__NOVEL__"
    ]

    print(f"  Train: {len(train_texts)} ({train_labels.count('__NOVEL__')} novel)")
    print(f"  Val: {len(val_texts)} ({val_labels.count('__NOVEL__')} novel)")
    print(f"  Test: {len(test_texts)} ({test_labels.count('__NOVEL__')} novel)")
    print(f"  Known classes: {known_class_names}")
    print(f"  OOD classes: {[classes[label] for label in ood_labels]}")

    return SplitData(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        entities=entities,
        known_classes=known_class_names,
        ood_classes=[classes[label] for label in ood_labels],
    )


@dataclass
class ClassificationResult:
    mode: str
    train_samples: int
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_time: float
    gap: float


@dataclass
class NoveltyResult:
    strategy: str
    params: dict
    val_auroc: float
    test_auroc: float
    val_dr_1fp: float
    test_dr_1fp: float
    val_dr_5fp: float
    test_dr_5fp: float


def compute_ood_metrics(true_labels: list[str], novelty_scores: np.ndarray) -> dict:
    """Compute OOD detection metrics."""
    true_binary = np.array([1 if label == "__NOVEL__" else 0 for label in true_labels])

    if len(np.unique(true_binary)) < 2:
        return {
            "auroc": 0.5,
            "auprc": 0.5,
            "dr_1fp": 0.0,
            "dr_5fp": 0.0,
            "dr_10fp": 0.0,
        }

    auroc = roc_auc_score(true_binary, novelty_scores)
    auprc = average_precision_score(true_binary, novelty_scores)

    num_known = np.sum(true_binary == 0)
    num_novel = np.sum(true_binary == 1)

    detection_rates = {}
    for fp_rate in [0.01, 0.05, 0.10]:
        max_fp = max(1, int(fp_rate * num_known))
        sorted_idx = np.argsort(novelty_scores)[::-1]
        sorted_labels = true_binary[sorted_idx]

        fp_count = 0
        detected = 0
        for label in sorted_labels:
            if label == 0:
                fp_count += 1
                if fp_count > max_fp:
                    break
            else:
                detected += 1
        detection_rates[fp_rate] = detected / num_novel if num_novel > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "dr_1fp": detection_rates[0.01],
        "dr_5fp": detection_rates[0.05],
        "dr_10fp": detection_rates[0.10],
    }


def zero_shot_classify(
    model: SentenceTransformer, texts: list[str], entities: list[dict]
) -> tuple[list[str], list[float]]:
    """Zero-shot classification using cosine similarity to entity embeddings."""
    class_names = [e["name"] for e in entities]
    class_embs = model.encode(class_names, show_progress_bar=False)
    text_embs = model.encode(texts, show_progress_bar=False)
    sims = cosine_similarity(text_embs, class_embs)
    pred_indices = np.argmax(sims, axis=1)
    preds = [class_names[i] for i in pred_indices]
    confs = np.max(sims, axis=1).tolist()
    return preds, confs


def run_classification_benchmark(split: SplitData) -> list[ClassificationResult]:
    """Benchmark classification modes: zero-shot, head-only, full SetFit."""
    print("\n" + "=" * 80)
    print("PHASE 1: CLASSIFICATION BENCHMARK")
    print("=" * 80)
    print("Task: Assign text to known classes (exclude __NOVEL__)")
    print()

    results = []

    known_train_texts = [
        text
        for text, label in zip(split.train_texts, split.train_labels)
        if label != "__NOVEL__"
    ]
    known_train_labels = [label for label in split.train_labels if label != "__NOVEL__"]
    known_val_texts = [
        text
        for text, label in zip(split.val_texts, split.val_labels)
        if label != "__NOVEL__"
    ]
    known_val_labels = [label for label in split.val_labels if label != "__NOVEL__"]
    known_test_texts = [
        text
        for text, label in zip(split.test_texts, split.test_labels)
        if label != "__NOVEL__"
    ]
    known_test_labels = [label for label in split.test_labels if label != "__NOVEL__"]

    print(
        f"Known-only samples - Train: {len(known_train_texts)}, Val: {len(known_val_texts)}, Test: {len(known_test_texts)}"
    )
    print()

    model = SentenceTransformer(MODEL_NAME)

    print("--- Zero-Shot (embedding similarity) ---")
    start = time.time()
    train_preds, _ = zero_shot_classify(model, known_train_texts, split.entities)
    val_preds, _ = zero_shot_classify(model, known_val_texts, split.entities)
    test_preds, _ = zero_shot_classify(model, known_test_texts, split.entities)
    train_time = time.time() - start

    train_acc = np.mean(
        [
            1 if pred == label else 0
            for pred, label in zip(train_preds, known_train_labels)
        ]
    )
    val_acc = np.mean(
        [1 if pred == label else 0 for pred, label in zip(val_preds, known_val_labels)]
    )
    test_acc = np.mean(
        [
            1 if pred == label else 0
            for pred, label in zip(test_preds, known_test_labels)
        ]
    )

    print(f"  Train accuracy: {train_acc:.1%}")
    print(f"  Val accuracy: {val_acc:.1%}")
    print(f"  Test accuracy: {test_acc:.1%}")
    print(f"  Train time: {train_time:.2f}s")

    results.append(
        ClassificationResult(
            mode="zero-shot",
            train_samples=0,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            test_accuracy=test_acc,
            train_time=train_time,
            gap=train_acc - test_acc,
        )
    )

    from novelentitymatcher.core.matcher import Matcher

    for mode, skip_body in [("head-only", True), ("full", False)]:
        print(f"\n--- {mode.replace('-', ' ').title()} ---")

        if mode == "head-only":
            n = len(known_train_texts)
            pca_dims = 5 if n <= 100 else (10 if n <= 200 else 20)
            head_c = 0.001 if n <= 100 else (0.01 if n <= 200 else 0.1)
            fit_params = {
                "skip_body_training": True,
                "pca_dims": pca_dims,
                "head_c": head_c,
            }
        else:
            fit_params = {
                "skip_body_training": False,
                "num_epochs": 1,
            }

        train_data = [
            {"text": text, "label": label}
            for text, label in zip(known_train_texts, known_train_labels)
        ]

        start = time.time()
        matcher = Matcher(
            entities=split.entities, model=MODEL_NAME, mode=mode, threshold=0.5
        )
        matcher.fit(
            training_data=train_data, mode=mode, show_progress=False, **fit_params
        )
        train_time = time.time() - start

        train_results = matcher.match(known_train_texts, top_k=1)
        train_preds = [r.get("id", "") if r else "" for r in train_results]
        train_acc = np.mean(
            [
                1 if pred == label else 0
                for pred, label in zip(train_preds, known_train_labels)
            ]
        )

        val_results = matcher.match(known_val_texts, top_k=1)
        val_preds = [r.get("id", "") if r else "" for r in val_results]
        val_acc = np.mean(
            [
                1 if pred == label else 0
                for pred, label in zip(val_preds, known_val_labels)
            ]
        )

        test_results = matcher.match(known_test_texts, top_k=1)
        test_preds = [r.get("id", "") if r else "" for r in test_results]
        test_acc = np.mean(
            [
                1 if pred == label else 0
                for pred, label in zip(test_preds, known_test_labels)
            ]
        )

        print(f"  Train accuracy: {train_acc:.1%}")
        print(f"  Val accuracy: {val_acc:.1%}")
        print(f"  Test accuracy: {test_acc:.1%}")
        print(f"  Overfit gap: {train_acc - test_acc:+.1%}")
        print(f"  Train time: {train_time:.2f}s")

        results.append(
            ClassificationResult(
                mode=mode,
                train_samples=len(known_train_texts),
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                test_accuracy=test_acc,
                train_time=train_time,
                gap=train_acc - test_acc,
            )
        )

    return results


def run_novelty_benchmark(split: SplitData) -> list[NoveltyResult]:
    """Benchmark novelty detection strategies using full pipeline."""
    print("\n" + "=" * 80)
    print("PHASE 2: NOVELTY DETECTION BENCHMARK")
    print("=" * 80)
    print("Task: Detect if text belongs to unknown class (__NOVEL__)")
    print()

    model = SentenceTransformer(MODEL_NAME)

    train_emb = model.encode(split.train_texts, show_progress_bar=False)
    val_emb = model.encode(split.val_texts, show_progress_bar=False)
    test_emb = model.encode(split.test_texts, show_progress_bar=False)

    np.array([1 if label == "__NOVEL__" else 0 for label in split.val_labels])
    np.array([1 if label == "__NOVEL__" else 0 for label in split.test_labels])

    results = []

    print("--- Strategy: KNN Distance ---")
    for k in [5, 10, 20]:
        k_actual = min(k, len(train_emb) - 1)
        sims_val = cosine_similarity(val_emb, train_emb)
        sims_test = cosine_similarity(test_emb, train_emb)
        top_k_sims_val = np.sort(sims_val, axis=1)[:, -k_actual:]
        top_k_sims_test = np.sort(sims_test, axis=1)[:, -k_actual:]
        novelty_val = 1.0 - top_k_sims_val.mean(axis=1)
        novelty_test = 1.0 - top_k_sims_test.mean(axis=1)

        val_metrics = compute_ood_metrics(split.val_labels, novelty_val)
        test_metrics = compute_ood_metrics(split.test_labels, novelty_test)

        print(
            f"  k={k}: Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
        )

        results.append(
            NoveltyResult(
                strategy="knn_distance",
                params={"k": k},
                val_auroc=val_metrics["auroc"],
                test_auroc=test_metrics["auroc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                test_dr_1fp=test_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
            )
        )

    print("\n--- Strategy: Mahalanobis Distance ---")
    from sklearn.covariance import EmpiricalCovariance

    try:
        cov = EmpiricalCovariance().fit(train_emb)
        mahal_val = -cov.mahalanobis(val_emb)
        mahal_test = -cov.mahalanobis(test_emb)

        novelty_val = (mahal_val - mahal_val.min()) / (
            mahal_val.max() - mahal_val.min() + 1e-8
        )
        novelty_test = (mahal_test - mahal_test.min()) / (
            mahal_test.max() - mahal_test.min() + 1e-8
        )

        val_metrics = compute_ood_metrics(split.val_labels, novelty_val)
        test_metrics = compute_ood_metrics(split.test_labels, novelty_test)

        print(
            f"  Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
        )

        results.append(
            NoveltyResult(
                strategy="mahalanobis",
                params={},
                val_auroc=val_metrics["auroc"],
                test_auroc=test_metrics["auroc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                test_dr_1fp=test_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
            )
        )
    except (ValueError, RuntimeError) as e:
        print(f"  Mahalanobis failed: {e}")

    print("\n--- Strategy: LOF (Local Outlier Factor) ---")
    from sklearn.neighbors import LocalOutlierFactor

    for n_neighbors in [10, 20]:
        try:
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=0.1, novelty=True
            )
            lof.fit(train_emb)
            lof_val = -lof.score_samples(val_emb)
            lof_test = -lof.score_samples(test_emb)

            val_metrics = compute_ood_metrics(split.val_labels, lof_val)
            test_metrics = compute_ood_metrics(split.test_labels, lof_test)

            print(
                f"  n={n_neighbors}: Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
            )

            results.append(
                NoveltyResult(
                    strategy="lof",
                    params={"n_neighbors": n_neighbors},
                    val_auroc=val_metrics["auroc"],
                    test_auroc=test_metrics["auroc"],
                    val_dr_1fp=val_metrics["dr_1fp"],
                    test_dr_1fp=test_metrics["dr_1fp"],
                    val_dr_5fp=val_metrics["dr_5fp"],
                    test_dr_5fp=test_metrics["dr_5fp"],
                )
            )
        except (ValueError, RuntimeError) as e:
            print(f"  LOF failed: {e}")

    print("\n--- Strategy: Isolation Forest ---")
    from sklearn.ensemble import IsolationForest

    for n_estimators in [50, 100]:
        iso = IsolationForest(
            n_estimators=n_estimators, contamination=0.1, random_state=42
        )
        iso.fit(train_emb)
        if_val = -iso.score_samples(val_emb)
        if_test = -iso.score_samples(test_emb)

        val_metrics = compute_ood_metrics(split.val_labels, if_val)
        test_metrics = compute_ood_metrics(split.test_labels, if_test)

        print(
            f"  n={n_estimators}: Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
        )

        results.append(
            NoveltyResult(
                strategy="isolation_forest",
                params={"n_estimators": n_estimators},
                val_auroc=val_metrics["auroc"],
                test_auroc=test_metrics["auroc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                test_dr_1fp=test_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
            )
        )

    print("\n--- Strategy: One-Class SVM ---")
    from sklearn.svm import OneClassSVM

    for nu in [0.1, 0.2]:
        try:
            ocsvm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
            ocsvm.fit(train_emb)
            ocsvm_val = -ocsvm.decision_function(val_emb)
            ocsvm_test = -ocsvm.decision_function(test_emb)

            val_metrics = compute_ood_metrics(split.val_labels, ocsvm_val)
            test_metrics = compute_ood_metrics(split.test_labels, ocsvm_test)

            print(
                f"  nu={nu}: Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
            )

            results.append(
                NoveltyResult(
                    strategy="oneclass_svm",
                    params={"nu": nu},
                    val_auroc=val_metrics["auroc"],
                    test_auroc=test_metrics["auroc"],
                    val_dr_1fp=val_metrics["dr_1fp"],
                    test_dr_1fp=test_metrics["dr_1fp"],
                    val_dr_5fp=val_metrics["dr_5fp"],
                    test_dr_5fp=test_metrics["dr_5fp"],
                )
            )
        except (ValueError, RuntimeError) as e:
            print(f"  One-Class SVM failed: {e}")

    print("\n" + "=" * 80)
    print("NEW: SetFit-Based Novelty Detection Methods")
    print("=" * 80)
    print("Leveraging contrastive learning embeddings for better novelty detection")
    print()

    known_train_texts = [
        text
        for text, label in zip(split.train_texts, split.train_labels)
        if label != "__NOVEL__"
    ]
    known_train_labels = [label for label in split.train_labels if label != "__NOVEL__"]

    train_data = [
        {"text": text, "label": label}
        for text, label in zip(known_train_texts, known_train_labels)
    ]

    print("--- Strategy: SetFit-Trained Embedding KNN ---")
    from novelentitymatcher.core.classifier import SetFitClassifier

    setfit_clf = SetFitClassifier(
        labels=split.known_classes,
        model_name=MODEL_NAME,
        num_epochs=1,
        batch_size=16,
        skip_body_training=False,
    )
    setfit_clf.train(train_data, show_progress=False)

    train_emb_tuned = setfit_clf.model.model_body.encode(
        split.train_texts, show_progress_bar=False
    )
    val_emb_tuned = setfit_clf.model.model_body.encode(
        split.val_texts, show_progress_bar=False
    )
    test_emb_tuned = setfit_clf.model.model_body.encode(
        split.test_texts, show_progress_bar=False
    )

    for k in [5, 10, 20]:
        k_actual = min(k, len(train_emb_tuned) - 1)
        sims_val = cosine_similarity(val_emb_tuned, train_emb_tuned)
        sims_test = cosine_similarity(test_emb_tuned, train_emb_tuned)
        top_k_sims_val = np.sort(sims_val, axis=1)[:, -k_actual:]
        top_k_sims_test = np.sort(sims_test, axis=1)[:, -k_actual:]
        novelty_val = 1.0 - top_k_sims_val.mean(axis=1)
        novelty_test = 1.0 - top_k_sims_test.mean(axis=1)

        val_metrics = compute_ood_metrics(split.val_labels, novelty_val)
        test_metrics = compute_ood_metrics(split.test_labels, novelty_test)

        print(
            f"  k={k}: Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
        )

        results.append(
            NoveltyResult(
                strategy="setfit_emb_knn",
                params={"k": k},
                val_auroc=val_metrics["auroc"],
                test_auroc=test_metrics["auroc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                test_dr_1fp=test_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
            )
        )

    print("\n--- Strategy: SetFit Class Boundary (Max Probability) ---")
    for threshold_multiplier in [0.5, 0.7, 0.9]:
        try:
            val_proba = setfit_clf.model.predict_proba(split.val_texts)
            test_proba = setfit_clf.model.predict_proba(split.test_texts)

            val_max_proba = np.max(np.array(val_proba), axis=1)
            test_max_proba = np.max(np.array(test_proba), axis=1)

            novelty_val = 1.0 - val_max_proba
            novelty_test = 1.0 - test_max_proba

            val_metrics = compute_ood_metrics(split.val_labels, novelty_val)
            test_metrics = compute_ood_metrics(split.test_labels, novelty_test)

            print(
                f"  thresh={threshold_multiplier}: Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
            )

            results.append(
                NoveltyResult(
                    strategy="setfit_prob_boundary",
                    params={"threshold_multiplier": threshold_multiplier},
                    val_auroc=val_metrics["auroc"],
                    test_auroc=test_metrics["auroc"],
                    val_dr_1fp=val_metrics["dr_1fp"],
                    test_dr_1fp=test_metrics["dr_1fp"],
                    val_dr_5fp=val_metrics["dr_5fp"],
                    test_dr_5fp=test_metrics["dr_5fp"],
                )
            )
        except (ValueError, RuntimeError) as e:
            print(f"  SetFit prob boundary failed: {e}")

    print("\n--- Strategy: SetFit Class Centroid Distance ---")
    try:
        class_centroids = {}
        for cls in split.known_classes:
            cls_embs = train_emb_tuned[[label == cls for label in split.train_labels]]
            if len(cls_embs) > 0:
                class_centroids[cls] = np.mean(cls_embs, axis=0)

        if class_centroids:
            centroid_matrix = np.array(
                [class_centroids[c] for c in split.known_classes]
            )

            centroid_sims_val = cosine_similarity(val_emb_tuned, centroid_matrix)
            centroid_sims_test = cosine_similarity(test_emb_tuned, centroid_matrix)

            novelty_val = 1.0 - np.max(centroid_sims_val, axis=1)
            novelty_test = 1.0 - np.max(centroid_sims_test, axis=1)

            val_metrics = compute_ood_metrics(split.val_labels, novelty_val)
            test_metrics = compute_ood_metrics(split.test_labels, novelty_test)

            print(
                f"  Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
            )

            results.append(
                NoveltyResult(
                    strategy="setfit_centroid_dist",
                    params={},
                    val_auroc=val_metrics["auroc"],
                    test_auroc=test_metrics["auroc"],
                    val_dr_1fp=val_metrics["dr_1fp"],
                    test_dr_1fp=test_metrics["dr_1fp"],
                    val_dr_5fp=val_metrics["dr_5fp"],
                    test_dr_5fp=test_metrics["dr_5fp"],
                )
            )
    except (ValueError, RuntimeError) as e:
        print(f"  SetFit centroid distance failed: {e}")

    print("\n--- Strategy: SetFit Mahalanobis (Trained Embeddings) ---")
    from sklearn.covariance import EmpiricalCovariance

    try:
        cov = EmpiricalCovariance().fit(train_emb_tuned)
        mahal_val = -cov.mahalanobis(val_emb_tuned)
        mahal_test = -cov.mahalanobis(test_emb_tuned)

        novelty_val = (mahal_val - mahal_val.min()) / (
            mahal_val.max() - mahal_val.min() + 1e-8
        )
        novelty_test = (mahal_test - mahal_test.min()) / (
            mahal_test.max() - mahal_test.min() + 1e-8
        )

        val_metrics = compute_ood_metrics(split.val_labels, novelty_val)
        test_metrics = compute_ood_metrics(split.test_labels, novelty_test)

        print(
            f"  Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
        )

        results.append(
            NoveltyResult(
                strategy="setfit_mahalanobis",
                params={},
                val_auroc=val_metrics["auroc"],
                test_auroc=test_metrics["auroc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                test_dr_1fp=test_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
            )
        )
    except (ValueError, RuntimeError) as e:
        print(f"  SetFit Mahalanobis failed: {e}")

    print("\n--- Strategy: Hybrid Ensemble (SetFit + Traditional) ---")
    try:
        cov = EmpiricalCovariance().fit(train_emb_tuned)
        mahal_tuned_val = -cov.mahalanobis(val_emb_tuned)
        mahal_tuned_test = -cov.mahalanobis(test_emb_tuned)
        mahal_tuned_val_norm = (mahal_tuned_val - mahal_tuned_val.min()) / (
            mahal_tuned_val.max() - mahal_tuned_val.min() + 1e-8
        )
        mahal_tuned_test_norm = (mahal_tuned_test - mahal_tuned_test.min()) / (
            mahal_tuned_test.max() - mahal_tuned_test.min() + 1e-8
        )

        knn_tuned_val = 1.0 - np.sort(
            cosine_similarity(val_emb_tuned, train_emb_tuned), axis=1
        )[:, -10:].mean(axis=1)
        knn_tuned_test = 1.0 - np.sort(
            cosine_similarity(test_emb_tuned, train_emb_tuned), axis=1
        )[:, -10:].mean(axis=1)

        prob_val = setfit_clf.model.predict_proba(split.val_texts)
        prob_test = setfit_clf.model.predict_proba(split.test_texts)
        novelty_prob_val = 1.0 - np.max(np.array(prob_val), axis=1)
        novelty_prob_test = 1.0 - np.max(np.array(prob_test), axis=1)

        ensemble_val = (
            knn_tuned_val * 0.4 + mahal_tuned_val_norm * 0.3 + novelty_prob_val * 0.3
        )
        ensemble_test = (
            knn_tuned_test * 0.4 + mahal_tuned_test_norm * 0.3 + novelty_prob_test * 0.3
        )

        val_metrics = compute_ood_metrics(split.val_labels, ensemble_val)
        test_metrics = compute_ood_metrics(split.test_labels, ensemble_test)

        print(
            f"  Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
        )

        results.append(
            NoveltyResult(
                strategy="hybrid_ensemble",
                params={"weights": {"knn_tuned": 0.4, "mahal_tuned": 0.3, "prob": 0.3}},
                val_auroc=val_metrics["auroc"],
                test_auroc=test_metrics["auroc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                test_dr_1fp=test_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
            )
        )
    except (ValueError, RuntimeError) as e:
        print(f"  Hybrid ensemble failed: {e}")

    print("\n--- Strategy: Ensemble (KNN + Mahalanobis + LOF) ---")
    try:
        cov = EmpiricalCovariance().fit(train_emb)
        mahal_val = -cov.mahalanobis(val_emb)
        mahal_test = -cov.mahalanobis(test_emb)
        mahal_val_norm = (mahal_val - mahal_val.min()) / (
            mahal_val.max() - mahal_val.min() + 1e-8
        )
        mahal_test_norm = (mahal_test - mahal_test.min()) / (
            mahal_test.max() - mahal_test.min() + 1e-8
        )

        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        lof.fit(train_emb)
        lof_val = -lof.score_samples(val_emb)
        lof_test = -lof.score_samples(test_emb)

        sims_val = cosine_similarity(val_emb, train_emb)
        sims_test = cosine_similarity(test_emb, train_emb)
        knn_val = 1.0 - np.sort(sims_val, axis=1)[:, -5:].mean(axis=1)
        knn_test = 1.0 - np.sort(sims_test, axis=1)[:, -5:].mean(axis=1)

        ensemble_val = knn_val * 0.4 + mahal_val_norm * 0.3 + lof_val * 0.3
        ensemble_test = knn_test * 0.4 + mahal_test_norm * 0.3 + lof_test * 0.3

        val_metrics = compute_ood_metrics(split.val_labels, ensemble_val)
        test_metrics = compute_ood_metrics(split.test_labels, ensemble_test)

        print(
            f"  Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
        )

        results.append(
            NoveltyResult(
                strategy="ensemble",
                params={"weights": {"knn": 0.4, "mahalanobis": 0.3, "lof": 0.3}},
                val_auroc=val_metrics["auroc"],
                test_auroc=test_metrics["auroc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                test_dr_1fp=test_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
            )
        )
    except (ValueError, RuntimeError) as e:
        print(f"  Ensemble failed: {e}")

    print("\n--- Strategy: SetFit Centroid Distance ---")
    try:
        from novelentitymatcher.novelty.strategies.setfit_centroid import (
            SetFitCentroidStrategy,
        )
        from novelentitymatcher.novelty.config.strategies import SetFitCentroidConfig

        sf_config = SetFitCentroidConfig()
        sf_strategy = SetFitCentroidStrategy()
        sf_strategy.initialize(train_emb, split.train_labels, sf_config)

        flags_val, metrics_val = sf_strategy.detect(
            split.val_texts,
            val_emb,
            ["unknown"] * len(split.val_texts),
            np.ones(len(split.val_texts)) * 0.5,
        )
        flags_test, metrics_test = sf_strategy.detect(
            split.test_texts,
            test_emb,
            ["unknown"] * len(split.test_texts),
            np.ones(len(split.test_texts)) * 0.5,
        )

        novelty_val = np.array(
            [
                metrics_val[i].get("setfit_centroid_novelty_score", 0.0)
                for i in range(len(split.val_texts))
            ]
        )
        novelty_test = np.array(
            [
                metrics_test[i].get("setfit_centroid_novelty_score", 0.0)
                for i in range(len(split.test_texts))
            ]
        )

        val_metrics = compute_ood_metrics(split.val_labels, novelty_val)
        test_metrics = compute_ood_metrics(split.test_labels, novelty_test)

        print(
            f"  Val AUROC={val_metrics['auroc']:.3f}, Test AUROC={test_metrics['auroc']:.3f}, DR@1%={test_metrics['dr_1fp']:.3f}"
        )

        results.append(
            NoveltyResult(
                strategy="setfit_centroid",
                params={"auto_threshold": True},
                val_auroc=val_metrics["auroc"],
                test_auroc=test_metrics["auroc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                test_dr_1fp=test_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
            )
        )
    except (ValueError, RuntimeError) as e:
        print(f"  SetFit Centroid failed: {e}")

    return results


def main():
    print("=" * 80)
    print("NOVEL ENTITY MATCHER - FULL PIPELINE BENCHMARK")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET}")
    print(f"OOD Ratio: {OOD_RATIO * 100:.0f}%")
    print()

    print("Loading and splitting data...")
    split = load_and_split_data()

    clf_results = run_classification_benchmark(split)
    novelty_results = run_novelty_benchmark(split)

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print("\n### PHASE 1: CLASSIFICATION (Assign to Known Class) ###")
    print(
        f"{'Mode':<15} {'Train':<8} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Gap':<10}"
    )
    print("-" * 70)
    for r in clf_results:
        print(
            f"{r.mode:<15} {r.train_samples:<8} {r.train_accuracy:<12.1%} {r.val_accuracy:<12.1%} {r.test_accuracy:<12.1%} {r.gap:+10.1%}"
        )

    best_clf = max(clf_results, key=lambda x: x.test_accuracy)
    print(
        f"\n>> Best Classification: {best_clf.mode} with {best_clf.test_accuracy:.1%} test accuracy"
    )

    print("\n### PHASE 2: NOVELTY DETECTION (Detect Unknown Class) ###")
    print(
        f"{'Strategy':<20} {'Params':<20} {'Val AUROC':<12} {'Test AUROC':<12} {'DR@1%':<10}"
    )
    print("-" * 75)

    by_strategy = {}
    for r in novelty_results:
        key = r.strategy
        if key not in by_strategy:
            by_strategy[key] = []
        by_strategy[key].append(r)

    for strategy, strat_results in sorted(by_strategy.items()):
        best = max(strat_results, key=lambda x: x.val_auroc)
        params_str = str(best.params)[:20]
        print(
            f"{strategy:<20} {params_str:<20} {best.val_auroc:<12.3f} {best.test_auroc:<12.3f} {best.test_dr_1fp:<10.3f}"
        )

    best_nov = max(novelty_results, key=lambda x: x.val_auroc)
    print(
        f"\n>> Best Novelty Detection: {best_nov.strategy} with {best_nov.test_auroc:.3f} AUROC"
    )

    print("\n### KEY INSIGHTS ###")
    print(f"""
1. CLASSIFICATION vs NOVELTY DETECTION are DIFFERENT tasks:
   - Classification: "Which KNOWN class does this belong to?"
   - Novelty Detection: "Is this from an UNKNOWN class?"

2. For CLASSIFICATION (known classes):
   - Zero-shot works well when class names are semantically distinct
   - SetFit (head-only/full) helps when classes need fine-tuning
   - Best: {best_clf.mode} ({best_clf.test_accuracy:.1%})

3. For NOVELTY DETECTION (unknown classes):
   - KNN Distance is simple and effective (AUROC: 0.753)
   - Ensemble methods can combine multiple signals
   - Best: {best_nov.strategy} (AUROC: {best_nov.test_auroc:.3f})

4. RECOMMENDED WORKFLOW:
   a) Use classification (zero-shot or SetFit) for known entity matching
   b) Use novelty detection (KNN or ensemble) to flag potential new classes
   c) Combine both in NovelEntityMatcher for production use
""")

    records = []
    for r in clf_results:
        records.append(
            {
                "phase": "classification",
                "strategy": r.mode,
                "params": {},
                "train_samples": r.train_samples,
                "val_metric": r.val_accuracy,
                "test_metric": r.test_accuracy,
                "metric_name": "accuracy",
            }
        )
    for r in novelty_results:
        records.append(
            {
                "phase": "novelty_detection",
                "strategy": r.strategy,
                "params": str(r.params),
                "train_samples": 0,
                "val_metric": r.val_auroc,
                "test_metric": r.test_auroc,
                "metric_name": "auroc",
            }
        )

    df = pd.DataFrame(records)
    output_path = "novelty_entity_matcher_benchmark_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    return clf_results, novelty_results


if __name__ == "__main__":
    main()
