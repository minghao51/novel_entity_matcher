"""Comprehensive benchmark for novelty detection strategies with train/val/test splits.

This benchmark evaluates all novelty detection strategies with:
- Proper train/val/test separation
- Hyperparameter tuning on validation set
- Final evaluation on held-out test set
- Same depth as the SetFit classification study
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentence_transformers import SentenceTransformer

import warnings

warnings.filterwarnings("ignore")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASETS = ["ag_news", "goemotions_novelty"]
MAX_TRAIN = 200
MAX_VAL = 200
MAX_TEST = 500
OOD_RATIO = 0.2
RANDOM_SEED = 42

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from novelentitymatcher.novelty.config.base import DetectionConfig
from novelentitymatcher.novelty.config.weights import WeightConfig
from novelentitymatcher.novelty.config.strategies import (
    ConfidenceConfig,
    KNNConfig,
    SetFitCentroidConfig,
)
from novelentitymatcher.novelty.strategies.confidence import ConfidenceStrategy
from novelentitymatcher.novelty.strategies.knn_distance import KNNDistanceStrategy
from novelentitymatcher.novelty.strategies.setfit_centroid import SetFitCentroidStrategy
from novelentitymatcher.novelty.core.signal_combiner import SignalCombiner
from novelentitymatcher.novelty.core.adaptive_weights import (
    compute_characteristics,
    adaptive_weights,
)
from novelentitymatcher.novelty.strategies.pattern_impl import PatternScorer


@dataclass
class SplitData:
    train_texts: list[str]
    train_labels: list[str]
    val_texts: list[str]
    val_labels: list[str]
    test_texts: list[str]
    test_labels: list[str]
    known_classes: list[str]
    ood_classes: list[str]


@dataclass
class StrategyResult:
    strategy: str
    params: dict[str, Any]
    train_time: float
    inference_time: float
    val_auroc: float
    val_auprc: float
    val_dr_1fp: float
    val_dr_5fp: float
    val_dr_10fp: float
    test_auroc: float
    test_auprc: float
    test_dr_1fp: float
    test_dr_5fp: float
    test_dr_10fp: float
    best_threshold: float
    dataset: str


def load_and_split_data(
    ds_name: str,
    max_train: int = MAX_TRAIN,
    max_val: int = MAX_VAL,
    max_test: int = MAX_TEST,
    ood_ratio: float = OOD_RATIO,
) -> tuple[SplitData, pd.DataFrame]:
    import random

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if ds_name == "ag_news":
        ds = load_dataset("ag_news")
        label_col = "label"
        text_col = "text"
        all_classes = ["World", "Sports", "Business", "Sci/Tech"]
    elif ds_name == "goemotions_novelty":
        ds = load_dataset("go_emotions")
        label_col = "labels"
        text_col = "text"
        all_classes = [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ]
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    split_key = "test" if "test" in ds else "train"
    df = pd.DataFrame(ds[split_key])

    if ds_name == "goemotions_novelty":
        df = df[
            df[label_col].apply(lambda x: isinstance(x, list) and len(x) == 1)
        ].copy()
        df[label_col] = df[label_col].apply(lambda x: int(x[0]))

    unique_labels = sorted(df[label_col].unique())
    num_ood = max(1, int(len(unique_labels) * ood_ratio))
    ood_labels = set(random.sample(unique_labels, num_ood))
    known_labels = sorted(set(unique_labels) - ood_labels)

    df["is_ood"] = df[label_col].isin(ood_labels).astype(int)
    known_df = df[df[label_col].isin(known_labels)]
    ood_df = df[df[label_col].isin(ood_labels)]

    train_df, temp_known = train_test_split(
        known_df, test_size=0.4, random_state=RANDOM_SEED
    )
    val_df, test_known_df = train_test_split(
        temp_known, test_size=0.5, random_state=RANDOM_SEED
    )

    train_ood_df, temp_ood = train_test_split(
        ood_df, test_size=0.5, random_state=RANDOM_SEED
    )
    val_ood_df, test_ood_df = train_test_split(
        temp_ood, test_size=0.5, random_state=RANDOM_SEED
    )

    train_texts = train_df[text_col].head(max_train).tolist()
    train_labels = [str(x) for x in train_df[label_col].head(max_train).tolist()]

    val_texts = (
        val_df[text_col].head(max_val).tolist()
        + val_ood_df[text_col].head(max_val).tolist()
    )
    val_labels = [str(x) for x in val_df[label_col].head(max_val).tolist()] + [
        "__OOD__"
    ] * min(len(val_ood_df), max_val)

    test_texts = (
        test_known_df[text_col].head(max_test).tolist()
        + test_ood_df[text_col].head(max_test).tolist()
    )
    test_labels = [str(x) for x in test_known_df[label_col].head(max_test).tolist()] + [
        "__OOD__"
    ] * min(len(test_ood_df), max_test)

    split_data = SplitData(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        known_classes=[
            all_classes[int(l)]
            if str(l).isdigit() and int(l) < len(all_classes)
            else str(l)
            for l in known_labels
        ],
        ood_classes=[
            all_classes[int(l)]
            if str(l).isdigit() and int(l) < len(all_classes)
            else str(l)
            for l in ood_labels
        ],
    )

    return split_data, df


def compute_metrics(
    true_labels: np.ndarray, novelty_scores: np.ndarray
) -> dict[str, float]:
    if len(np.unique(true_labels)) < 2:
        return {
            "auroc": 0.0,
            "auprc": 0.0,
            "dr_1fp": 0.0,
            "dr_5fp": 0.0,
            "dr_10fp": 0.0,
        }

    auroc = roc_auc_score(true_labels, novelty_scores)
    auprc = average_precision_score(true_labels, novelty_scores)

    num_known = np.sum(true_labels == 0)
    num_ood = np.sum(true_labels == 1)

    detection_rates = {}
    for fp_rate in [0.01, 0.05, 0.10]:
        max_false_positives = max(1, int(fp_rate * num_known))
        sorted_indices = np.argsort(novelty_scores)[::-1]
        sorted_labels = true_labels[sorted_indices]

        fp_count = 0
        detected_ood = 0
        for label in sorted_labels:
            if label == 0:
                fp_count += 1
                if fp_count > max_false_positives:
                    break
            else:
                detected_ood += 1
        detection_rates[fp_rate] = detected_ood / num_ood if num_ood > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "dr_1fp": detection_rates[0.01],
        "dr_5fp": detection_rates[0.05],
        "dr_10fp": detection_rates[0.10],
    }


def prepare_binary_labels(labels: list[str]) -> np.ndarray:
    return np.array([1 if l == "__OOD__" else 0 for l in labels])


class NoveltyBenchmark:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        print(f"Loaded model: {model_name}")

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

    def evaluate_on_split(
        self,
        texts: list[str],
        true_labels: np.ndarray,
        novelty_scores: np.ndarray,
    ) -> dict[str, float]:
        return compute_metrics(true_labels, novelty_scores)

    def benchmark_zero_shot_knn(
        self,
        split: SplitData,
        k_values: list[int] = None,
        distance_thresholds: list[float] = None,
    ) -> list[StrategyResult]:
        if k_values is None:
            k_values = [3, 5, 10, 20, 30]
        if distance_thresholds is None:
            distance_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)

        val_true = prepare_binary_labels(split.val_labels)
        test_true = prepare_binary_labels(split.test_labels)

        results = []
        for k in k_values:
            for dist_thresh in distance_thresholds:
                novelty_scores = self._compute_knn_novelty(train_emb, val_emb, k=k)
                val_metrics = self.evaluate_on_split(
                    split.val_texts, val_true, novelty_scores
                )

                test_novelty = self._compute_knn_novelty(train_emb, test_emb, k=k)
                test_metrics = self.evaluate_on_split(
                    split.test_texts, test_true, test_novelty
                )

                results.append(
                    StrategyResult(
                        strategy="knn_distance",
                        params={"k": k, "distance_threshold": dist_thresh},
                        train_time=0.0,
                        inference_time=0.0,
                        val_auroc=val_metrics["auroc"],
                        val_auprc=val_metrics["auprc"],
                        val_dr_1fp=val_metrics["dr_1fp"],
                        val_dr_5fp=val_metrics["dr_5fp"],
                        val_dr_10fp=val_metrics["dr_10fp"],
                        test_auroc=test_metrics["auroc"],
                        test_auprc=test_metrics["auprc"],
                        test_dr_1fp=test_metrics["dr_1fp"],
                        test_dr_5fp=test_metrics["dr_5fp"],
                        test_dr_10fp=test_metrics["dr_10fp"],
                        best_threshold=dist_thresh,
                        dataset="",
                    )
                )

        return results

    def _compute_knn_novelty(
        self,
        train_emb: np.ndarray,
        query_emb: np.ndarray,
        k: int = 5,
    ) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(query_emb, train_emb)
        k = min(k, len(train_emb))
        top_k_sims = np.sort(sims, axis=1)[:, -k:]
        novelty = 1.0 - top_k_sims.mean(axis=1)
        return novelty

    def benchmark_mahalanobis(
        self,
        split: SplitData,
        thresholds: list[float] = None,
    ) -> list[StrategyResult]:
        if thresholds is None:
            thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)

        val_true = prepare_binary_labels(split.val_labels)
        test_true = prepare_binary_labels(split.test_labels)

        class_means = {}
        for label in set(split.train_labels):
            mask = np.array([l == label for l in split.train_labels])
            class_means[label] = train_emb[mask].mean(axis=0)

        global_cov = np.cov(train_emb, rowvar=False) + 1e-6 * np.eye(train_emb.shape[1])
        cov_inv = np.linalg.inv(global_cov)

        novelty_train = self._compute_mahalanobis_novelty(
            train_emb, class_means, cov_inv
        )
        novelty_val = self._compute_mahalanobis_novelty(val_emb, class_means, cov_inv)
        novelty_test = self._compute_mahalanobis_novelty(test_emb, class_means, cov_inv)

        val_metrics_list = [
            compute_metrics(val_true, novelty_val * (t / 3.0)) for t in thresholds
        ]
        test_metrics_list = [
            compute_metrics(test_true, novelty_test * (t / 3.0)) for t in thresholds
        ]

        results = []
        for i, thresh in enumerate(thresholds):
            results.append(
                StrategyResult(
                    strategy="mahalanobis",
                    params={"threshold": thresh},
                    train_time=0.0,
                    inference_time=0.0,
                    val_auroc=val_metrics_list[i]["auroc"],
                    val_auprc=val_metrics_list[i]["auprc"],
                    val_dr_1fp=val_metrics_list[i]["dr_1fp"],
                    val_dr_5fp=val_metrics_list[i]["dr_5fp"],
                    val_dr_10fp=val_metrics_list[i]["dr_10fp"],
                    test_auroc=test_metrics_list[i]["auroc"],
                    test_auprc=test_metrics_list[i]["auprc"],
                    test_dr_1fp=test_metrics_list[i]["dr_1fp"],
                    test_dr_5fp=test_metrics_list[i]["dr_5fp"],
                    test_dr_10fp=test_metrics_list[i]["dr_10fp"],
                    best_threshold=thresh,
                    dataset="",
                )
            )

        return results

    def _compute_mahalanobis_novelty(
        self,
        embeddings: np.ndarray,
        class_means: dict[str, np.ndarray],
        cov_inv: np.ndarray,
    ) -> np.ndarray:
        novelty_scores = []
        global_mean = np.mean(list(class_means.values()), axis=0)

        for emb in embeddings:
            diff = emb - global_mean
            mahal = np.sqrt(np.abs(diff @ cov_inv @ diff))
            novelty_scores.append(mahal)

        return np.array(novelty_scores)

    def benchmark_lof(
        self,
        split: SplitData,
        n_neighbors_list: list[int] = None,
        contamination_list: list[float] = None,
    ) -> list[StrategyResult]:
        if n_neighbors_list is None:
            n_neighbors_list = [10, 20, 30, 50]
        if contamination_list is None:
            contamination_list = [0.05, 0.1, 0.15, 0.2]

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)

        val_true = prepare_binary_labels(split.val_labels)
        test_true = prepare_binary_labels(split.test_labels)

        results = []
        for n_neighbors in n_neighbors_list:
            for contamination in contamination_list:
                try:
                    novelty_val, novelty_test = self._compute_lof_scores(
                        train_emb, val_emb, test_emb, n_neighbors, contamination
                    )
                    val_metrics = self.evaluate_on_split(
                        split.val_texts, val_true, novelty_val
                    )
                    test_metrics = self.evaluate_on_split(
                        split.test_texts, test_true, novelty_test
                    )

                    results.append(
                        StrategyResult(
                            strategy="lof",
                            params={
                                "n_neighbors": n_neighbors,
                                "contamination": contamination,
                            },
                            train_time=0.0,
                            inference_time=0.0,
                            val_auroc=val_metrics["auroc"],
                            val_auprc=val_metrics["auprc"],
                            val_dr_1fp=val_metrics["dr_1fp"],
                            val_dr_5fp=val_metrics["dr_5fp"],
                            val_dr_10fp=val_metrics["dr_10fp"],
                            test_auroc=test_metrics["auroc"],
                            test_auprc=test_metrics["auprc"],
                            test_dr_1fp=test_metrics["dr_1fp"],
                            test_dr_5fp=test_metrics["dr_5fp"],
                            test_dr_10fp=test_metrics["dr_10fp"],
                            best_threshold=0.0,
                            dataset="",
                        )
                    )
                except (ValueError, RuntimeError) as e:
                    print(f"    LOF failed: {e}")

        return results

    def _compute_lof_scores(
        self,
        train_emb: np.ndarray,
        val_emb: np.ndarray,
        test_emb: np.ndarray,
        n_neighbors: int,
        contamination: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination, novelty=True
        )
        lof.fit(train_emb)

        val_scores = -lof.score_samples(val_emb)
        test_scores = -lof.score_samples(test_emb)

        return val_scores, test_scores

    def benchmark_pattern(
        self,
        split: SplitData,
        thresholds: list[float] = None,
    ) -> list[StrategyResult]:
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        val_true = prepare_binary_labels(split.val_labels)
        test_true = prepare_binary_labels(split.test_labels)

        from novelentitymatcher.novelty.strategies.pattern_impl import PatternScorer

        scorer = PatternScorer(split.train_texts)

        val_scores = np.array([scorer.score_novelty(t) for t in split.val_texts])
        test_scores = np.array([scorer.score_novelty(t) for t in split.test_texts])

        results = []
        for thresh in thresholds:
            val_novelty = 1.0 - (val_scores < thresh).astype(float)
            test_novelty = 1.0 - (test_scores < thresh).astype(float)

            val_metrics = self.evaluate_on_split(split.val_texts, val_true, val_novelty)
            test_metrics = self.evaluate_on_split(
                split.test_texts, test_true, test_novelty
            )

            results.append(
                StrategyResult(
                    strategy="pattern",
                    params={"threshold": thresh},
                    train_time=0.0,
                    inference_time=0.0,
                    val_auroc=val_metrics["auroc"],
                    val_auprc=val_metrics["auprc"],
                    val_dr_1fp=val_metrics["dr_1fp"],
                    val_dr_5fp=val_metrics["dr_5fp"],
                    val_dr_10fp=val_metrics["dr_10fp"],
                    test_auroc=test_metrics["auroc"],
                    test_auprc=test_metrics["auprc"],
                    test_dr_1fp=test_metrics["dr_1fp"],
                    test_dr_5fp=test_metrics["dr_5fp"],
                    test_dr_10fp=test_metrics["dr_10fp"],
                    best_threshold=thresh,
                    dataset="",
                )
            )

        return results

    def benchmark_oneclass_svm(
        self,
        split: SplitData,
        nu_values: list[float] = None,
        kernel_values: list[str] = None,
    ) -> list[StrategyResult]:
        if nu_values is None:
            nu_values = [0.05, 0.1, 0.2, 0.3]
        if kernel_values is None:
            kernel_values = ["rbf", "linear"]

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)

        val_true = prepare_binary_labels(split.val_labels)
        test_true = prepare_binary_labels(split.test_labels)

        results = []
        for nu in nu_values:
            for kernel in kernel_values:
                try:
                    start_time = time.time()
                    novelty_val, novelty_test, train_time = self._compute_ocsvm_scores(
                        train_emb, val_emb, test_emb, nu, kernel
                    )
                    inference_time = time.time() - start_time

                    val_metrics = self.evaluate_on_split(
                        split.val_texts, val_true, novelty_val
                    )
                    test_metrics = self.evaluate_on_split(
                        split.test_texts, test_true, novelty_test
                    )

                    results.append(
                        StrategyResult(
                            strategy="oneclass_svm",
                            params={"nu": nu, "kernel": kernel},
                            train_time=train_time,
                            inference_time=inference_time,
                            val_auroc=val_metrics["auroc"],
                            val_auprc=val_metrics["auprc"],
                            val_dr_1fp=val_metrics["dr_1fp"],
                            val_dr_5fp=val_metrics["dr_5fp"],
                            val_dr_10fp=val_metrics["dr_10fp"],
                            test_auroc=test_metrics["auroc"],
                            test_auprc=test_metrics["auprc"],
                            test_dr_1fp=test_metrics["dr_1fp"],
                            test_dr_5fp=test_metrics["dr_5fp"],
                            test_dr_10fp=test_metrics["dr_10fp"],
                            best_threshold=0.0,
                            dataset="",
                        )
                    )
                except (ValueError, RuntimeError) as e:
                    print(f"    OneClassSVM failed: {e}")

        return results

    def _compute_ocsvm_scores(
        self,
        train_emb: np.ndarray,
        val_emb: np.ndarray,
        test_emb: np.ndarray,
        nu: float,
        kernel: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        from sklearn.svm import OneClassSVM

        start_time = time.time()
        ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")
        ocsvm.fit(train_emb)
        train_time = time.time() - start_time

        val_scores = -ocsvm.decision_function(val_emb)
        test_scores = -ocsvm.decision_function(test_emb)

        return val_scores, test_scores, train_time

    def benchmark_isolation_forest(
        self,
        split: SplitData,
        contamination_values: list[float] = None,
        n_estimators_list: list[int] = None,
    ) -> list[StrategyResult]:
        if contamination_values is None:
            contamination_values = [0.05, 0.1, 0.2, 0.3]
        if n_estimators_list is None:
            n_estimators_list = [50, 100, 200]

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)

        val_true = prepare_binary_labels(split.val_labels)
        test_true = prepare_binary_labels(split.test_labels)

        results = []
        for contamination in contamination_values:
            for n_estimators in n_estimators_list:
                try:
                    start_time = time.time()
                    novelty_val, novelty_test, train_time = self._compute_if_scores(
                        train_emb, val_emb, test_emb, contamination, n_estimators
                    )
                    inference_time = time.time() - start_time

                    val_metrics = self.evaluate_on_split(
                        split.val_texts, val_true, novelty_val
                    )
                    test_metrics = self.evaluate_on_split(
                        split.test_texts, test_true, novelty_test
                    )

                    results.append(
                        StrategyResult(
                            strategy="isolation_forest",
                            params={
                                "contamination": contamination,
                                "n_estimators": n_estimators,
                            },
                            train_time=train_time,
                            inference_time=inference_time,
                            val_auroc=val_metrics["auroc"],
                            val_auprc=val_metrics["auprc"],
                            val_dr_1fp=val_metrics["dr_1fp"],
                            val_dr_5fp=val_metrics["dr_5fp"],
                            val_dr_10fp=val_metrics["dr_10fp"],
                            test_auroc=test_metrics["auroc"],
                            test_auprc=test_metrics["auprc"],
                            test_dr_1fp=test_metrics["dr_1fp"],
                            test_dr_5fp=test_metrics["dr_5fp"],
                            test_dr_10fp=test_metrics["dr_10fp"],
                            best_threshold=0.0,
                            dataset="",
                        )
                    )
                except (ValueError, RuntimeError) as e:
                    print(f"    IsolationForest failed: {e}")

        return results

    def _compute_if_scores(
        self,
        train_emb: np.ndarray,
        val_emb: np.ndarray,
        test_emb: np.ndarray,
        contamination: float,
        n_estimators: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        from sklearn.ensemble import IsolationForest

        start_time = time.time()
        iso = IsolationForest(
            contamination=contamination, n_estimators=n_estimators, random_state=42
        )
        iso.fit(train_emb)
        train_time = time.time() - start_time

        val_scores = -iso.score_samples(val_emb)
        test_scores = -iso.score_samples(test_emb)

        return val_scores, test_scores, train_time

    def benchmark_setfit_novelty(
        self,
        split: SplitData,
        num_epochs_list: list[int] = None,
        margin_values: list[float] = None,
    ) -> list[StrategyResult]:
        if num_epochs_list is None:
            num_epochs_list = [1, 2, 4]
        if margin_values is None:
            margin_values = [0.5, 1.0, 2.0]

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)

        val_true = prepare_binary_labels(split.val_labels)
        test_true = prepare_binary_labels(split.test_labels)

        results = []
        for num_epochs in num_epochs_list:
            for margin in margin_values:
                try:
                    start_time = time.time()
                    novelty_val, novelty_test, train_time = self._compute_setfit_scores(
                        train_emb,
                        split.train_labels,
                        val_emb,
                        test_emb,
                        num_epochs,
                        margin,
                    )
                    inference_time = time.time() - start_time

                    val_metrics = self.evaluate_on_split(
                        split.val_texts, val_true, novelty_val
                    )
                    test_metrics = self.evaluate_on_split(
                        split.test_texts, test_true, novelty_test
                    )

                    results.append(
                        StrategyResult(
                            strategy="setfit_novelty",
                            params={"num_epochs": num_epochs, "margin": margin},
                            train_time=train_time,
                            inference_time=inference_time,
                            val_auroc=val_metrics["auroc"],
                            val_auprc=val_metrics["auprc"],
                            val_dr_1fp=val_metrics["dr_1fp"],
                            val_dr_5fp=val_metrics["dr_5fp"],
                            val_dr_10fp=val_metrics["dr_10fp"],
                            test_auroc=test_metrics["auroc"],
                            test_auprc=test_metrics["auprc"],
                            test_dr_1fp=test_metrics["dr_1fp"],
                            test_dr_5fp=test_metrics["dr_5fp"],
                            test_dr_10fp=test_metrics["dr_10fp"],
                            best_threshold=0.0,
                            dataset="",
                        )
                    )
                except (ValueError, RuntimeError) as e:
                    print(f"    SetFit novelty failed: {e}")

        return results

    def _compute_setfit_scores(
        self,
        train_emb: np.ndarray,
        train_labels: list[str],
        val_emb: np.ndarray,
        test_emb: np.ndarray,
        num_epochs: int,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics.pairwise import cosine_similarity

        start_time = time.time()

        unique_labels = list(set(train_labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}

        positive_pairs = []
        for i, emb in enumerate(train_emb):
            label = train_labels[i]
            for j, other_emb in enumerate(train_emb):
                if i != j and train_labels[j] == label:
                    positive_pairs.append((emb, other_emb))

        if len(positive_pairs) < 10:
            class_centroids = {}
            for i, label in enumerate(train_labels):
                if label not in class_centroids:
                    class_centroids[label] = []
                class_centroids[label].append(train_emb[i])
            centroids = np.array(
                [np.mean(class_centroids[l], axis=0) for l in unique_labels]
            )
            centroid_map = {l: centroids[i] for i, l in enumerate(unique_labels)}

            val_sim = np.array(
                [np.max(cosine_similarity([e], centroids)[0]) for e in val_emb]
            )
            test_sim = np.array(
                [np.max(cosine_similarity([e], centroids)[0]) for e in test_emb]
            )

            novelty_val = 1.0 - val_sim
            novelty_test = 1.0 - test_sim

            return novelty_val, novelty_test, time.time() - start_time

        pos_emb = np.array([p[0] for p in positive_pairs])
        neg_emb = np.array([p[1] for p in positive_pairs])

        for epoch in range(num_epochs):
            new_pos = []
            for emb in pos_emb:
                perturbations = emb + np.random.randn(5, emb.shape[0]) * 0.1
                new_pos.extend(perturbations)
            if len(new_pos) > 0:
                pos_emb = np.vstack([pos_emb, np.array(new_pos)])

        X_train = np.vstack([pos_emb, neg_emb])
        y_train = np.array([1] * len(pos_emb) + [0] * len(neg_emb))

        clf = LogisticRegression(max_iter=100, random_state=42)
        clf.fit(X_train, y_train)

        novelty_val = 1.0 - clf.predict_proba(val_emb)[:, 1]
        novelty_test = 1.0 - clf.predict_proba(test_emb)[:, 1]

        train_time = time.time() - start_time

        return novelty_val, novelty_test, train_time

    def benchmark_ensemble(
        self,
        split: SplitData,
        strategy_weights: dict[str, float] = None,
    ) -> list[StrategyResult]:
        if strategy_weights is None:
            strategy_weights = {
                "knn": 0.4,
                "mahalanobis": 0.3,
                "lof": 0.2,
                "pattern": 0.1,
            }

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)

        val_true = prepare_binary_labels(split.val_labels)
        test_true = prepare_binary_labels(split.test_labels)

        novelty_knn_val = self._compute_knn_novelty(train_emb, val_emb, k=5)
        novelty_knn_test = self._compute_knn_novelty(train_emb, test_emb, k=5)

        class_means = {}
        for label in set(split.train_labels):
            mask = np.array([l == label for l in split.train_labels])
            class_means[label] = train_emb[mask].mean(axis=0)
        global_cov = np.cov(train_emb, rowvar=False) + 1e-6 * np.eye(train_emb.shape[1])
        cov_inv = np.linalg.inv(global_cov)
        novelty_maha_val = self._compute_mahalanobis_novelty(
            val_emb, class_means, cov_inv
        )
        novelty_maha_test = self._compute_mahalanobis_novelty(
            test_emb, class_means, cov_inv
        )

        novelty_lof_val, novelty_lof_test = self._compute_lof_scores(
            train_emb, val_emb, test_emb, n_neighbors=20, contamination=0.1
        )

        from novelentitymatcher.novelty.strategies.pattern_impl import PatternScorer

        scorer = PatternScorer(split.train_texts)
        novelty_pattern_val = np.array(
            [1.0 - scorer.score_novelty(t) for t in split.val_texts]
        )
        novelty_pattern_test = np.array(
            [1.0 - scorer.score_novelty(t) for t in split.test_texts]
        )

        novelty_ensemble_val = (
            novelty_knn_val * strategy_weights["knn"]
            + novelty_maha_val * strategy_weights["mahalanobis"]
            + novelty_lof_val * strategy_weights["lof"]
            + novelty_pattern_val * strategy_weights["pattern"]
        )
        novelty_ensemble_test = (
            novelty_knn_test * strategy_weights["knn"]
            + novelty_maha_test * strategy_weights["mahalanobis"]
            + novelty_lof_test * strategy_weights["lof"]
            + novelty_pattern_test * strategy_weights["pattern"]
        )

        val_metrics = self.evaluate_on_split(
            split.val_texts, val_true, novelty_ensemble_val
        )
        test_metrics = self.evaluate_on_split(
            split.test_texts, test_true, novelty_ensemble_test
        )

        return [
            StrategyResult(
                strategy="ensemble",
                params=strategy_weights,
                train_time=0.0,
                inference_time=0.0,
                val_auroc=val_metrics["auroc"],
                val_auprc=val_metrics["auprc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                val_dr_10fp=val_metrics["dr_10fp"],
                test_auroc=test_metrics["auroc"],
                test_auprc=test_metrics["auprc"],
                test_dr_1fp=test_metrics["dr_1fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
                test_dr_10fp=test_metrics["dr_10fp"],
                best_threshold=0.0,
                dataset="",
            )
        ]

    def benchmark_signal_combiner(
        self,
        split: SplitData,
        combine_method: str = "weighted",
    ) -> list[StrategyResult]:
        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)

        val_true = prepare_binary_labels(split.val_labels)
        test_true = prepare_binary_labels(split.test_labels)

        strategy_outputs_val: dict[str, tuple[set[int], dict]] = {}
        strategy_outputs_test: dict[str, tuple[set[int], dict]] = {}

        # Confidence
        conf_config = ConfidenceConfig(threshold=0.5)
        conf_strategy = ConfidenceStrategy()
        conf_strategy.initialize(train_emb, split.train_labels, conf_config)
        flags_v, metrics_v = conf_strategy.detect(
            split.val_texts,
            val_emb,
            ["unknown"] * len(split.val_texts),
            np.ones(len(split.val_texts)) * 0.5,
        )
        flags_t, metrics_t = conf_strategy.detect(
            split.test_texts,
            test_emb,
            ["unknown"] * len(split.test_texts),
            np.ones(len(split.test_texts)) * 0.5,
        )
        strategy_outputs_val["confidence"] = (flags_v, metrics_v)
        strategy_outputs_test["confidence"] = (flags_t, metrics_t)

        # KNN
        knn_config = KNNConfig(k=5, distance_threshold=0.5)
        knn_strategy = KNNDistanceStrategy()
        knn_strategy.initialize(train_emb, split.train_labels, knn_config)
        flags_v, metrics_v = knn_strategy.detect(
            split.val_texts,
            val_emb,
            ["unknown"] * len(split.val_texts),
            np.ones(len(split.val_texts)) * 0.5,
        )
        flags_t, metrics_t = knn_strategy.detect(
            split.test_texts,
            test_emb,
            ["unknown"] * len(split.test_texts),
            np.ones(len(split.test_texts)) * 0.5,
        )
        strategy_outputs_val["knn_distance"] = (flags_v, metrics_v)
        strategy_outputs_test["knn_distance"] = (flags_t, metrics_t)

        # Pattern
        pattern_scorer = PatternScorer(split.train_texts)
        pattern_scores_val = np.array(
            [pattern_scorer.score_novelty(t) for t in split.val_texts]
        )
        pattern_scores_test = np.array(
            [pattern_scorer.score_novelty(t) for t in split.test_texts]
        )
        pattern_flags_v, pattern_metrics_v = {}, {}
        pattern_flags_t, pattern_metrics_t = {}, {}
        for i, s in enumerate(pattern_scores_val):
            is_novel = s < 0.5
            if is_novel:
                pattern_flags_v[i] = i
            pattern_metrics_v[i] = {
                "pattern_is_novel": is_novel,
                "pattern_score": 1.0 - s,
            }
        for i, s in enumerate(pattern_scores_test):
            is_novel = s < 0.5
            if is_novel:
                pattern_flags_t[i] = i
            pattern_metrics_t[i] = {
                "pattern_is_novel": is_novel,
                "pattern_score": 1.0 - s,
            }
        strategy_outputs_val["pattern"] = (
            set(pattern_flags_v.values()),
            pattern_metrics_v,
        )
        strategy_outputs_test["pattern"] = (
            set(pattern_flags_t.values()),
            pattern_metrics_t,
        )

        # SetFit Centroid
        sf_config = SetFitCentroidConfig()
        sf_strategy = SetFitCentroidStrategy()
        sf_strategy.initialize(train_emb, split.train_labels, sf_config)
        flags_v, metrics_v = sf_strategy.detect(
            split.val_texts,
            val_emb,
            ["unknown"] * len(split.val_texts),
            np.ones(len(split.val_texts)) * 0.5,
        )
        flags_t, metrics_t = sf_strategy.detect(
            split.test_texts,
            test_emb,
            ["unknown"] * len(split.test_texts),
            np.ones(len(split.test_texts)) * 0.5,
        )
        strategy_outputs_val["setfit_centroid"] = (flags_v, metrics_v)
        strategy_outputs_test["setfit_centroid"] = (flags_t, metrics_t)

        # Merge metrics
        all_metrics_val = {i: {} for i in range(len(split.val_texts))}
        all_metrics_test = {i: {} for i in range(len(split.test_texts))}
        for _, (_, m) in strategy_outputs_val.items():
            for idx, mm in m.items():
                all_metrics_val[idx].update(mm)
        for _, (_, m) in strategy_outputs_test.items():
            for idx, mm in m.items():
                all_metrics_test[idx].update(mm)

        weights = WeightConfig()
        det_config = DetectionConfig(combine_method=combine_method, weights=weights)
        combiner = SignalCombiner(det_config)

        if combine_method == "meta_learner":
            # Build training features from val set (using val labels as ground truth)
            feature_keys = combiner._feature_names
            train_features = []
            train_labels_arr = []
            for idx in sorted(all_metrics_val.keys()):
                feats = combiner._extract_features(idx, all_metrics_val)
                train_features.append(feats)
                train_labels_arr.append(val_true[idx])
            if len(set(train_labels_arr)) > 1:
                combiner.train_meta_learner(
                    np.array(train_features), np.array(train_labels_arr)
                )

        novel_indices_val, novelty_scores_val = combiner.combine(
            strategy_outputs_val, all_metrics_val
        )
        novel_indices_test, novelty_scores_test = combiner.combine(
            strategy_outputs_test, all_metrics_test
        )

        scores_val = np.array(
            [novelty_scores_val.get(i, 0.0) for i in range(len(split.val_texts))]
        )
        scores_test = np.array(
            [novelty_scores_test.get(i, 0.0) for i in range(len(split.test_texts))]
        )

        val_metrics = self.evaluate_on_split(split.val_texts, val_true, scores_val)
        test_metrics = self.evaluate_on_split(split.test_texts, test_true, scores_test)

        return [
            StrategyResult(
                strategy=f"signal_combiner_{combine_method}",
                params={"method": combine_method},
                train_time=0.0,
                inference_time=0.0,
                val_auroc=val_metrics["auroc"],
                val_auprc=val_metrics["auprc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                val_dr_10fp=val_metrics["dr_10fp"],
                test_auroc=test_metrics["auroc"],
                test_auprc=test_metrics["auprc"],
                test_dr_1fp=test_metrics["dr_1fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
                test_dr_10fp=test_metrics["dr_10fp"],
                best_threshold=0.0,
                dataset="",
            )
        ]

    def benchmark_adaptive_weights(
        self,
        split: SplitData,
    ) -> list[StrategyResult]:
        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)

        val_true = prepare_binary_labels(split.val_labels)
        test_true = prepare_binary_labels(split.test_labels)

        characteristics = compute_characteristics(train_emb, split.train_labels)
        adjusted_weights = adaptive_weights(characteristics)

        strategy_outputs_val: dict[str, tuple[set[int], dict]] = {}
        strategy_outputs_test: dict[str, tuple[set[int], dict]] = {}

        conf_config = ConfidenceConfig(threshold=0.5)
        conf_strategy = ConfidenceStrategy()
        conf_strategy.initialize(train_emb, split.train_labels, conf_config)
        flags_v, metrics_v = conf_strategy.detect(
            split.val_texts,
            val_emb,
            ["unknown"] * len(split.val_texts),
            np.ones(len(split.val_texts)) * 0.5,
        )
        flags_t, metrics_t = conf_strategy.detect(
            split.test_texts,
            test_emb,
            ["unknown"] * len(split.test_texts),
            np.ones(len(split.test_texts)) * 0.5,
        )
        strategy_outputs_val["confidence"] = (flags_v, metrics_v)
        strategy_outputs_test["confidence"] = (flags_t, metrics_t)

        knn_config = KNNConfig(k=5, distance_threshold=0.5)
        knn_strategy = KNNDistanceStrategy()
        knn_strategy.initialize(train_emb, split.train_labels, knn_config)
        flags_v, metrics_v = knn_strategy.detect(
            split.val_texts,
            val_emb,
            ["unknown"] * len(split.val_texts),
            np.ones(len(split.val_texts)) * 0.5,
        )
        flags_t, metrics_t = knn_strategy.detect(
            split.test_texts,
            test_emb,
            ["unknown"] * len(split.test_texts),
            np.ones(len(split.test_texts)) * 0.5,
        )
        strategy_outputs_val["knn_distance"] = (flags_v, metrics_v)
        strategy_outputs_test["knn_distance"] = (flags_t, metrics_t)

        pattern_scorer = PatternScorer(split.train_texts)
        pattern_scores_val = np.array(
            [pattern_scorer.score_novelty(t) for t in split.val_texts]
        )
        pattern_scores_test = np.array(
            [pattern_scorer.score_novelty(t) for t in split.test_texts]
        )
        pattern_flags_v, pattern_metrics_v = {}, {}
        pattern_flags_t, pattern_metrics_t = {}, {}
        for i, s in enumerate(pattern_scores_val):
            is_novel = s < 0.5
            if is_novel:
                pattern_flags_v[i] = i
            pattern_metrics_v[i] = {
                "pattern_is_novel": is_novel,
                "pattern_score": 1.0 - s,
            }
        for i, s in enumerate(pattern_scores_test):
            is_novel = s < 0.5
            if is_novel:
                pattern_flags_t[i] = i
            pattern_metrics_t[i] = {
                "pattern_is_novel": is_novel,
                "pattern_score": 1.0 - s,
            }
        strategy_outputs_val["pattern"] = (
            set(pattern_flags_v.values()),
            pattern_metrics_v,
        )
        strategy_outputs_test["pattern"] = (
            set(pattern_flags_t.values()),
            pattern_metrics_t,
        )

        sf_config = SetFitCentroidConfig()
        sf_strategy = SetFitCentroidStrategy()
        sf_strategy.initialize(train_emb, split.train_labels, sf_config)
        flags_v, metrics_v = sf_strategy.detect(
            split.val_texts,
            val_emb,
            ["unknown"] * len(split.val_texts),
            np.ones(len(split.val_texts)) * 0.5,
        )
        flags_t, metrics_t = sf_strategy.detect(
            split.test_texts,
            test_emb,
            ["unknown"] * len(split.test_texts),
            np.ones(len(split.test_texts)) * 0.5,
        )
        strategy_outputs_val["setfit_centroid"] = (flags_v, metrics_v)
        strategy_outputs_test["setfit_centroid"] = (flags_t, metrics_t)

        all_metrics_val = {i: {} for i in range(len(split.val_texts))}
        all_metrics_test = {i: {} for i in range(len(split.test_texts))}
        for _, (_, m) in strategy_outputs_val.items():
            for idx, mm in m.items():
                all_metrics_val[idx].update(mm)
        for _, (_, m) in strategy_outputs_test.items():
            for idx, mm in m.items():
                all_metrics_test[idx].update(mm)

        det_config = DetectionConfig(
            combine_method="weighted", weights=adjusted_weights
        )
        combiner = SignalCombiner(det_config)
        novel_indices_val, novelty_scores_val = combiner.combine(
            strategy_outputs_val, all_metrics_val
        )
        novel_indices_test, novelty_scores_test = combiner.combine(
            strategy_outputs_test, all_metrics_test
        )

        scores_val = np.array(
            [novelty_scores_val.get(i, 0.0) for i in range(len(split.val_texts))]
        )
        scores_test = np.array(
            [novelty_scores_test.get(i, 0.0) for i in range(len(split.test_texts))]
        )

        val_metrics = self.evaluate_on_split(split.val_texts, val_true, scores_val)
        test_metrics = self.evaluate_on_split(split.test_texts, test_true, scores_test)

        return [
            StrategyResult(
                strategy="signal_combiner_adaptive",
                params={
                    "method": "adaptive",
                    "adjusted_weights": {
                        "confidence": adjusted_weights.confidence,
                        "knn": adjusted_weights.knn,
                        "pattern": adjusted_weights.pattern,
                        "setfit_centroid": adjusted_weights.setfit_centroid,
                    },
                },
                train_time=0.0,
                inference_time=0.0,
                val_auroc=val_metrics["auroc"],
                val_auprc=val_metrics["auprc"],
                val_dr_1fp=val_metrics["dr_1fp"],
                val_dr_5fp=val_metrics["dr_5fp"],
                val_dr_10fp=val_metrics["dr_10fp"],
                test_auroc=test_metrics["auroc"],
                test_auprc=test_metrics["auprc"],
                test_dr_1fp=test_metrics["dr_1fp"],
                test_dr_5fp=test_metrics["dr_5fp"],
                test_dr_10fp=test_metrics["dr_10fp"],
                best_threshold=0.0,
                dataset="",
            )
        ]


def run_full_benchmark():
    print("=" * 100)
    print("NOVELTY DETECTION STRATEGIES BENCHMARK - Full Study")
    print("=" * 100)
    print(f"Model: {MODEL_NAME}")
    print(
        f"Train samples: {MAX_TRAIN}, Val samples: {MAX_VAL}, Test samples: {MAX_TEST}"
    )
    print(f"OOD ratio: {OOD_RATIO}")
    print()

    benchmark = NoveltyBenchmark(MODEL_NAME)
    all_results: list[StrategyResult] = []

    for ds_name in DATASETS:
        print(f"\n{'=' * 100}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 100}")

        split, full_df = load_and_split_data(ds_name)
        print(
            f"Train: {len(split.train_texts)}, Val: {len(split.val_texts)}, Test: {len(split.test_texts)}"
        )
        print(
            f"Known classes: {len(split.known_classes)}, OOD classes: {split.ood_classes}"
        )

        for result in split.train_texts[:3]:
            print(f"  Sample: {result[:60]}...")

        ds_results = []

        print("\n--- KNN Distance (zero-shot) ---")
        results = benchmark.benchmark_zero_shot_knn(split)
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        best = max(results, key=lambda x: x.val_auroc)
        print(
            f"  Best: k={best.params['k']}, thresh={best.params['distance_threshold']:.2f}"
        )
        print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
        print(f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}")

        print("\n--- Mahalanobis Distance (zero-shot) ---")
        results = benchmark.benchmark_mahalanobis(split)
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        best = max(results, key=lambda x: x.val_auroc)
        print(f"  Best: thresh={best.params['threshold']:.1f}")
        print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
        print(f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}")

        print("\n--- LOF (zero-shot) ---")
        results = benchmark.benchmark_lof(split)
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        if results:
            best = max(results, key=lambda x: x.val_auroc)
            print(
                f"  Best: n={best.params['n_neighbors']}, cont={best.params['contamination']:.2f}"
            )
            print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
            print(
                f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}"
            )

        print("\n--- Pattern (zero-shot) ---")
        results = benchmark.benchmark_pattern(split)
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        best = max(results, key=lambda x: x.val_auroc)
        print(f"  Best: thresh={best.params['threshold']:.2f}")
        print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
        print(f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}")

        print("\n--- One-Class SVM (trained) ---")
        results = benchmark.benchmark_oneclass_svm(split)
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        if results:
            best = max(results, key=lambda x: x.val_auroc)
            print(f"  Best: nu={best.params['nu']}, kernel={best.params['kernel']}")
            print(f"    Train time: {best.train_time:.2f}s")
            print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
            print(
                f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}"
            )

        print("\n--- Isolation Forest (trained) ---")
        results = benchmark.benchmark_isolation_forest(split)
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        if results:
            best = max(results, key=lambda x: x.val_auroc)
            print(
                f"  Best: cont={best.params['contamination']}, n={best.params['n_estimators']}"
            )
            print(f"    Train time: {best.train_time:.2f}s")
            print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
            print(
                f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}"
            )

        print("\n--- SetFit Novelty (trained - contrastive) ---")
        results = benchmark.benchmark_setfit_novelty(split)
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        if results:
            best = max(results, key=lambda x: x.val_auroc)
            print(
                f"  Best: epochs={best.params['num_epochs']}, margin={best.params['margin']}"
            )
            print(f"    Train time: {best.train_time:.2f}s")
            print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
            print(
                f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}"
            )

        print("\n--- Ensemble (KNN + Mahalanobis + LOF + Pattern) ---")
        results = benchmark.benchmark_ensemble(split)
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        best = results[0] if results else None
        if best:
            print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
            print(
                f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}"
            )

        print("\n--- Signal Combiner: Weighted ---")
        results = benchmark.benchmark_signal_combiner(split, combine_method="weighted")
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        if results:
            best = results[0]
            print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
            print(
                f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}"
            )

        print("\n--- Signal Combiner: Voting ---")
        results = benchmark.benchmark_signal_combiner(split, combine_method="voting")
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        if results:
            best = results[0]
            print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
            print(
                f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}"
            )

        print("\n--- Signal Combiner: Meta-Learner ---")
        results = benchmark.benchmark_signal_combiner(
            split, combine_method="meta_learner"
        )
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        if results:
            best = results[0]
            print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
            print(
                f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}"
            )

        print("\n--- Signal Combiner: Adaptive Weights ---")
        results = benchmark.benchmark_adaptive_weights(split)
        for r in results:
            r.dataset = ds_name
            ds_results.append(r)
        if results:
            best = results[0]
            print(f"    Val: AUROC={best.val_auroc:.3f}, DR@1%={best.val_dr_1fp:.3f}")
            print(
                f"    Test: AUROC={best.test_auroc:.3f}, DR@1%={best.test_dr_1fp:.3f}"
            )

        all_results.extend(ds_results)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY: Best Results by Strategy")
    print("=" * 100)

    records = []
    for ds_name in DATASETS:
        ds_results = [r for r in all_results if r.dataset == ds_name]
        print(f"\n--- {ds_name} ---")

        by_strategy = {}
        for r in ds_results:
            if r.strategy not in by_strategy:
                by_strategy[r.strategy] = []
            by_strategy[r.strategy].append(r)

        print(
            f"\n{'Strategy':<25} {'Val AUROC':>10} {'Test AUROC':>12} {'Test DR@1%':>12} {'Best Params':>30}"
        )
        print("-" * 90)

        for strategy, results in sorted(by_strategy.items()):
            best = max(results, key=lambda x: x.val_auroc)
            print(
                f"{strategy:<25} {best.val_auroc:>10.3f} {best.test_auroc:>12.3f} {best.test_dr_1fp:>12.3f} {str(best.params):>30}"
            )

        best_overall = max(ds_results, key=lambda x: x.test_auroc)
        records.append(
            {
                "dataset": ds_name,
                "best_strategy": best_overall.strategy,
                "best_params": str(best_overall.params),
                "val_auroc": best_overall.val_auroc,
                "test_auroc": best_overall.test_auroc,
                "test_dr_1fp": best_overall.test_dr_1fp,
            }
        )

    print("\n" + "=" * 100)
    print("OVERALL BEST BY DATASET")
    print("=" * 100)
    for r in records:
        print(f"\n{r['dataset']}:")
        print(f"  Best Strategy: {r['best_strategy']}")
        print(f"  Best Params: {r['best_params']}")
        print(f"  Val AUROC: {r['val_auroc']:.3f}")
        print(f"  Test AUROC: {r['test_auroc']:.3f}")
        print(f"  Test DR@1%: {r['test_dr_1fp']:.3f}")

    df = pd.DataFrame(
        [
            {
                "strategy": r.strategy,
                "dataset": r.dataset,
                "params": str(r.params),
                "val_auroc": r.val_auroc,
                "val_auprc": r.val_auprc,
                "val_dr_1fp": r.val_dr_1fp,
                "val_dr_5fp": r.val_dr_5fp,
                "val_dr_10fp": r.val_dr_10fp,
                "test_auroc": r.test_auroc,
                "test_auprc": r.test_auprc,
                "test_dr_1fp": r.test_dr_1fp,
                "test_dr_5fp": r.test_dr_5fp,
                "test_dr_10fp": r.test_dr_10fp,
                "train_time": r.train_time,
                "inference_time": r.inference_time,
                "best_threshold": r.best_threshold,
            }
            for r in all_results
        ]
    )

    output_path = "novelty_strategies_full_benchmark.csv"
    df.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    return df


if __name__ == "__main__":
    results_df = run_full_benchmark()
