"""Comprehensive benchmark for all novelty detection strategies."""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from novelentitymatcher.novelty.config.base import DetectionConfig
from novelentitymatcher.novelty.config.strategies import (
    ConfidenceConfig,
    KNNConfig,
    UncertaintyConfig,
    ClusteringConfig,
    SelfKnowledgeConfig,
    PatternConfig,
    OneClassConfig,
    PrototypicalConfig,
    MahalanobisConfig,
    LOFConfig,
    SetFitConfig,
)
from novelentitymatcher.novelty.strategies.base import NoveltyStrategy
from novelentitymatcher.novelty.strategies.confidence import ConfidenceStrategy
from novelentitymatcher.novelty.strategies.knn_distance import KNNDistanceStrategy
from novelentitymatcher.novelty.strategies.uncertainty import UncertaintyStrategy
from novelentitymatcher.novelty.strategies.clustering import ClusteringStrategy
from novelentitymatcher.novelty.strategies.self_knowledge import SelfKnowledgeStrategy
from novelentitymatcher.novelty.strategies.pattern_impl import PatternScorer
from novelentitymatcher.novelty.strategies.oneclass_impl import OneClassSVMDetector
from novelentitymatcher.novelty.strategies.prototypical_impl import PrototypicalDetector
from novelentitymatcher.novelty.strategies.mahalanobis import (
    MahalanobisDistanceStrategy,
)
from novelentitymatcher.novelty.strategies.lof import LOFStrategy
from novelentitymatcher.novelty.strategies.setfit_centroid import SetFitCentroidStrategy
from novelentitymatcher.novelty.config.strategies import SetFitCentroidConfig
from novelentitymatcher.novelty.core.signal_combiner import SignalCombiner
from novelentitymatcher.novelty.core.adaptive_weights import (
    compute_characteristics,
    adaptive_weights,
)
from sentence_transformers import SentenceTransformer

import warnings

warnings.filterwarnings("ignore")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATASETS = ["ag_news", "go_emotions"]
MAX_SAMPLES = 500
OOD_RATIO = 0.2


@dataclass
class BenchmarkResult:
    strategy: str
    params: dict[str, Any]
    auroc: float
    auprc: float
    detection_rate_1fp: float
    detection_rate_5fp: float
    detection_rate_10fp: float
    train_time: float
    inference_time: float
    dataset: str
    num_known: int
    num_ood: int


def load_dataset_data(hf_path: str, split: str = "test", max_samples: int = 500):
    ds = load_dataset(hf_path)
    if split not in ds:
        split = "train" if "train" in ds else list(ds.keys())[0]
    df = pd.DataFrame(ds[split][: min(max_samples * 10, len(ds[split]))])
    return df


def create_ood_split(
    df: pd.DataFrame,
    label_col: str,
    ood_ratio: float = 0.2,
    max_samples_per_split: int = 500,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    import random

    random.seed(random_seed)

    unique_labels = sorted(df[label_col].unique())
    num_ood = max(1, int(len(unique_labels) * ood_ratio))
    ood_labels = set(random.sample(unique_labels, num_ood))
    known_labels = set(unique_labels) - ood_labels

    known_data = df[df[label_col].isin(known_labels)].copy()
    ood_data = df[df[label_col].isin(ood_labels)].copy()

    known_data = known_data.head(max_samples_per_split)
    ood_data = ood_data.head(max_samples_per_split)

    return known_data, ood_data, list(ood_labels)


def compute_metrics(
    true_labels: np.ndarray, novelty_scores: np.ndarray
) -> dict[str, float]:
    auroc = (
        roc_auc_score(true_labels, novelty_scores)
        if len(np.unique(true_labels)) > 1
        else 0.0
    )
    auprc = (
        average_precision_score(true_labels, novelty_scores)
        if len(np.unique(true_labels)) > 1
        else 0.0
    )

    num_known = np.sum(true_labels == 0)
    num_ood = np.sum(true_labels == 1)

    detection_rates = {}
    for fp_rate in [0.01, 0.05, 0.10]:
        max_false_positives = int(fp_rate * num_known)
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
        "detection_rate_1fp": detection_rates[0.01],
        "detection_rate_5fp": detection_rates[0.05],
        "detection_rate_10fp": detection_rates[0.10],
    }


def encode_texts(model, texts: list[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False)


class StrategyBenchmark:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def benchmark_confidence(
        self,
        known_texts: list[str],
        ood_texts: list[str],
        known_labels: list[str],
        thresholds: list[float] = None,
    ) -> list[BenchmarkResult]:
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        all_texts = known_texts + ood_texts
        embeddings = encode_texts(self.model, all_texts)
        known_emb = embeddings[: len(known_texts)]
        ood_emb = embeddings[len(known_texts) :]

        results = []
        for threshold in thresholds:
            config = ConfidenceConfig(threshold=threshold)
            strategy = ConfidenceStrategy()
            strategy.initialize(known_emb, known_labels, config)

            start_time = time.time()
            flags, metrics = strategy.detect(
                all_texts,
                embeddings,
                ["unknown"] * len(all_texts),
                np.ones(len(all_texts)) * 0.5,
            )
            inference_time = time.time() - start_time

            novelty_scores = np.array(
                [1 - metrics[i]["confidence_score"] for i in range(len(all_texts))]
            )
            true_labels = np.array([0] * len(known_texts) + [1] * len(ood_texts))

            metrics_result = compute_metrics(true_labels, novelty_scores)

            results.append(
                BenchmarkResult(
                    strategy="confidence",
                    params={"threshold": threshold},
                    train_time=0.0,
                    inference_time=inference_time,
                    dataset="ag_news" if len(known_texts) > 0 else "goemotions",
                    num_known=len(known_texts),
                    num_ood=len(ood_texts),
                    **metrics_result,
                )
            )

        return results

    def benchmark_knn(
        self,
        known_texts: list[str],
        ood_texts: list[str],
        known_labels: list[str],
        k_values: list[int] = None,
        distance_thresholds: list[float] = None,
    ) -> list[BenchmarkResult]:
        if k_values is None:
            k_values = [3, 5, 10, 20, 30]
        if distance_thresholds is None:
            distance_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        all_texts = known_texts + ood_texts
        embeddings = encode_texts(self.model, all_texts)
        known_emb = embeddings[: len(known_texts)]
        ood_emb = embeddings[len(known_texts) :]

        results = []
        for k in k_values:
            for dist_thresh in distance_thresholds:
                config = KNNConfig(k=k, distance_threshold=dist_thresh)
                strategy = KNNDistanceStrategy()
                strategy.initialize(known_emb, known_labels, config)

                start_time = time.time()
                flags, metrics = strategy.detect(
                    all_texts,
                    embeddings,
                    ["unknown"] * len(all_texts),
                    np.ones(len(all_texts)) * 0.5,
                )
                inference_time = time.time() - start_time

                novelty_scores = np.array(
                    [metrics[i]["knn_novelty_score"] for i in range(len(all_texts))]
                )
                true_labels = np.array([0] * len(known_texts) + [1] * len(ood_texts))

                metrics_result = compute_metrics(true_labels, novelty_scores)

                results.append(
                    BenchmarkResult(
                        strategy="knn_distance",
                        params={"k": k, "distance_threshold": dist_thresh},
                        train_time=0.0,
                        inference_time=inference_time,
                        num_known=len(known_texts),
                        num_ood=len(ood_texts),
                        dataset="",
                        **metrics_result,
                    )
                )

        return results

    def benchmark_mahalanobis(
        self,
        known_texts: list[str],
        ood_texts: list[str],
        known_labels: list[str],
        thresholds: list[float] = None,
    ) -> list[BenchmarkResult]:
        if thresholds is None:
            thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

        all_texts = known_texts + ood_texts
        embeddings = encode_texts(self.model, all_texts)
        known_emb = embeddings[: len(known_texts)]
        ood_emb = embeddings[len(known_texts) :]

        results = []
        for threshold in thresholds:
            config = MahalanobisConfig(threshold=threshold, use_class_conditional=True)
            strategy = MahalanobisDistanceStrategy()
            strategy.initialize(known_emb, known_labels, config)

            start_time = time.time()
            flags, metrics = strategy.detect(
                all_texts,
                embeddings,
                ["unknown"] * len(all_texts),
                np.ones(len(all_texts)) * 0.5,
            )
            inference_time = time.time() - start_time

            novelty_scores = np.array(
                [metrics[i]["mahalanobis_distance"] for i in range(len(all_texts))]
            )
            true_labels = np.array([0] * len(known_texts) + [1] * len(ood_texts))

            metrics_result = compute_metrics(true_labels, novelty_scores)

            results.append(
                BenchmarkResult(
                    strategy="mahalanobis",
                    params={"threshold": threshold, "class_conditional": True},
                    train_time=0.0,
                    inference_time=inference_time,
                    num_known=len(known_texts),
                    num_ood=len(ood_texts),
                    dataset="",
                    **metrics_result,
                )
            )

        return results

    def benchmark_lof(
        self,
        known_texts: list[str],
        ood_texts: list[str],
        known_labels: list[str],
        n_neighbors_list: list[int] = None,
        contamination_list: list[float] = None,
    ) -> list[BenchmarkResult]:
        if n_neighbors_list is None:
            n_neighbors_list = [10, 20, 30, 50]
        if contamination_list is None:
            contamination_list = [0.05, 0.1, 0.15, 0.2]

        all_texts = known_texts + ood_texts
        embeddings = encode_texts(self.model, all_texts)
        known_emb = embeddings[: len(known_texts)]
        ood_emb = embeddings[len(known_texts) :]

        results = []
        for n_neighbors in n_neighbors_list:
            for contamination in contamination_list:
                config = LOFConfig(n_neighbors=n_neighbors, contamination=contamination)
                strategy = LOFStrategy()
                strategy.initialize(known_emb, known_labels, config)

                start_time = time.time()
                flags, metrics = strategy.detect(
                    all_texts,
                    embeddings,
                    ["unknown"] * len(all_texts),
                    np.ones(len(all_texts)) * 0.5,
                )
                inference_time = time.time() - start_time

                novelty_scores = np.array(
                    [metrics[i]["lof_novelty_score"] for i in range(len(all_texts))]
                )
                true_labels = np.array([0] * len(known_texts) + [1] * len(ood_texts))

                metrics_result = compute_metrics(true_labels, novelty_scores)

                results.append(
                    BenchmarkResult(
                        strategy="lof",
                        params={
                            "n_neighbors": n_neighbors,
                            "contamination": contamination,
                        },
                        train_time=0.0,
                        inference_time=inference_time,
                        num_known=len(known_texts),
                        num_ood=len(ood_texts),
                        dataset="",
                        **metrics_result,
                    )
                )

        return results

    def benchmark_setfit_centroid(
        self,
        known_texts: list[str],
        ood_texts: list[str],
        known_labels: list[str],
    ) -> list[BenchmarkResult]:
        all_texts = known_texts + ood_texts
        embeddings = encode_texts(self.model, all_texts)
        known_emb = embeddings[: len(known_texts)]

        config = SetFitCentroidConfig()
        strategy = SetFitCentroidStrategy()
        strategy.initialize(known_emb, known_labels, config)

        start_time = time.time()
        flags, metrics = strategy.detect(
            all_texts,
            embeddings,
            ["unknown"] * len(all_texts),
            np.ones(len(all_texts)) * 0.5,
        )
        inference_time = time.time() - start_time

        novelty_scores = np.array(
            [metrics[i].get("setfit_centroid_novelty_score", 0.0) for i in range(len(all_texts))]
        )
        true_labels = np.array([0] * len(known_texts) + [1] * len(ood_texts))

        metrics_result = compute_metrics(true_labels, novelty_scores)

        return [
            BenchmarkResult(
                strategy="setfit_centroid",
                params={"auto_threshold": True},
                train_time=0.0,
                inference_time=inference_time,
                num_known=len(known_texts),
                num_ood=len(ood_texts),
                dataset="",
                **metrics_result,
            )
        ]

    def benchmark_pattern(
        self,
        known_texts: list[str],
        ood_texts: list[str],
        known_labels: list[str],
        thresholds: list[float] = None,
    ) -> list[BenchmarkResult]:
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        all_texts = known_texts + ood_texts
        embeddings = encode_texts(self.model, all_texts)
        known_emb = embeddings[: len(known_texts)]
        ood_emb = embeddings[len(known_texts) :]

        scorer = PatternScorer(known_texts)

        results = []
        for threshold in thresholds:
            config = PatternConfig(threshold=threshold)

            start_time = time.time()
            novelty_scores = [scorer.score_novelty(t) for t in all_texts]
            inference_time = time.time() - start_time

            novelty_scores = np.array(novelty_scores)
            true_labels = np.array([0] * len(known_texts) + [1] * len(ood_texts))

            metrics_result = compute_metrics(true_labels, novelty_scores)

            results.append(
                BenchmarkResult(
                    strategy="pattern",
                    params={"threshold": threshold},
                    train_time=0.0,
                    inference_time=inference_time,
                    num_known=len(known_texts),
                    num_ood=len(ood_texts),
                    dataset="",
                    **metrics_result,
                )
            )

        return results

    def benchmark_ensemble_weighted(
        self,
        known_texts: list[str],
        ood_texts: list[str],
        known_labels: list[str],
    ) -> list[BenchmarkResult]:
        all_texts = known_texts + ood_texts
        embeddings = encode_texts(self.model, all_texts)
        known_emb = embeddings[: len(known_texts)]
        true_labels = np.array([0] * len(known_texts) + [1] * len(ood_texts))

        strategy_outputs = {}

        # Confidence
        conf_config = ConfidenceConfig(threshold=0.5)
        conf_strategy = ConfidenceStrategy()
        conf_strategy.initialize(known_emb, known_labels, conf_config)
        flags, metrics = conf_strategy.detect(
            all_texts, embeddings, ["unknown"] * len(all_texts), np.ones(len(all_texts)) * 0.5,
        )
        strategy_outputs["confidence"] = (flags, metrics)

        # KNN
        knn_config = KNNConfig(k=5, distance_threshold=0.5)
        knn_strategy = KNNDistanceStrategy()
        knn_strategy.initialize(known_emb, known_labels, knn_config)
        flags, metrics = knn_strategy.detect(
            all_texts, embeddings, ["unknown"] * len(all_texts), np.ones(len(all_texts)) * 0.5,
        )
        strategy_outputs["knn_distance"] = (flags, metrics)

        # Pattern
        pattern_scorer = PatternScorer(known_texts)
        pattern_scores = np.array([pattern_scorer.score_novelty(t) for t in all_texts])
        pattern_flags = set()
        pattern_metrics = {}
        for i, s in enumerate(pattern_scores):
            is_novel = s < 0.5
            if is_novel:
                pattern_flags.add(i)
            pattern_metrics[i] = {"pattern_is_novel": is_novel, "pattern_score": 1.0 - s}
        strategy_outputs["pattern"] = (pattern_flags, pattern_metrics)

        # SetFit Centroid
        sf_config = SetFitCentroidConfig()
        sf_strategy = SetFitCentroidStrategy()
        sf_strategy.initialize(known_emb, known_labels, sf_config)
        flags, metrics = sf_strategy.detect(
            all_texts, embeddings, ["unknown"] * len(all_texts), np.ones(len(all_texts)) * 0.5,
        )
        strategy_outputs["setfit_centroid"] = (flags, metrics)

        # Merge all metrics
        all_metrics = {}
        for i in range(len(all_texts)):
            all_metrics[i] = {}
        for _, (_, metrics) in strategy_outputs.items():
            for idx, m in metrics.items():
                all_metrics[idx].update(m)

        from novelentitymatcher.novelty.config.base import DetectionConfig
        det_config = DetectionConfig(combine_method="weighted")
        combiner = SignalCombiner(det_config)
        novel_indices, novelty_scores = combiner.combine(strategy_outputs, all_metrics)

        scores = np.array([novelty_scores.get(i, 0.0) for i in range(len(all_texts))])
        metrics_result = compute_metrics(true_labels, scores)

        return [
            BenchmarkResult(
                strategy="ensemble_weighted",
                params={"method": "weighted"},
                train_time=0.0,
                inference_time=0.0,
                num_known=len(known_texts),
                num_ood=len(ood_texts),
                dataset="",
                **metrics_result,
            )
        ]

    def benchmark_ensemble_voting(
        self,
        known_texts: list[str],
        ood_texts: list[str],
        known_labels: list[str],
    ) -> list[BenchmarkResult]:
        all_texts = known_texts + ood_texts
        embeddings = encode_texts(self.model, all_texts)
        known_emb = embeddings[: len(known_texts)]
        true_labels = np.array([0] * len(known_texts) + [1] * len(ood_texts))

        strategy_outputs = {}

        conf_config = ConfidenceConfig(threshold=0.5)
        conf_strategy = ConfidenceStrategy()
        conf_strategy.initialize(known_emb, known_labels, conf_config)
        flags, metrics = conf_strategy.detect(
            all_texts, embeddings, ["unknown"] * len(all_texts), np.ones(len(all_texts)) * 0.5,
        )
        strategy_outputs["confidence"] = (flags, metrics)

        knn_config = KNNConfig(k=5, distance_threshold=0.5)
        knn_strategy = KNNDistanceStrategy()
        knn_strategy.initialize(known_emb, known_labels, knn_config)
        flags, metrics = knn_strategy.detect(
            all_texts, embeddings, ["unknown"] * len(all_texts), np.ones(len(all_texts)) * 0.5,
        )
        strategy_outputs["knn_distance"] = (flags, metrics)

        pattern_scorer = PatternScorer(known_texts)
        pattern_scores = np.array([pattern_scorer.score_novelty(t) for t in all_texts])
        pattern_flags = set()
        pattern_metrics = {}
        for i, s in enumerate(pattern_scores):
            is_novel = s < 0.5
            if is_novel:
                pattern_flags.add(i)
            pattern_metrics[i] = {"pattern_is_novel": is_novel, "pattern_score": 1.0 - s}
        strategy_outputs["pattern"] = (pattern_flags, pattern_metrics)

        sf_config = SetFitCentroidConfig()
        sf_strategy = SetFitCentroidStrategy()
        sf_strategy.initialize(known_emb, known_labels, sf_config)
        flags, metrics = sf_strategy.detect(
            all_texts, embeddings, ["unknown"] * len(all_texts), np.ones(len(all_texts)) * 0.5,
        )
        strategy_outputs["setfit_centroid"] = (flags, metrics)

        all_metrics = {}
        for i in range(len(all_texts)):
            all_metrics[i] = {}
        for _, (_, metrics) in strategy_outputs.items():
            for idx, m in metrics.items():
                all_metrics[idx].update(m)

        from novelentitymatcher.novelty.config.base import DetectionConfig
        det_config = DetectionConfig(combine_method="voting")
        combiner = SignalCombiner(det_config)
        novel_indices, novelty_scores = combiner.combine(strategy_outputs, all_metrics)

        scores = np.array([novelty_scores.get(i, 0.0) for i in range(len(all_texts))])
        metrics_result = compute_metrics(true_labels, scores)

        return [
            BenchmarkResult(
                strategy="ensemble_voting",
                params={"method": "voting"},
                train_time=0.0,
                inference_time=0.0,
                num_known=len(known_texts),
                num_ood=len(ood_texts),
                dataset="",
                **metrics_result,
            )
        ]

    def benchmark_ensemble_adaptive(
        self,
        known_texts: list[str],
        ood_texts: list[str],
        known_labels: list[str],
    ) -> list[BenchmarkResult]:
        all_texts = known_texts + ood_texts
        embeddings = encode_texts(self.model, all_texts)
        known_emb = embeddings[: len(known_texts)]
        true_labels = np.array([0] * len(known_texts) + [1] * len(ood_texts))

        characteristics = compute_characteristics(known_emb, known_labels)
        adjusted_weights = adaptive_weights(characteristics)

        strategy_outputs = {}

        conf_config = ConfidenceConfig(threshold=0.5)
        conf_strategy = ConfidenceStrategy()
        conf_strategy.initialize(known_emb, known_labels, conf_config)
        flags, metrics = conf_strategy.detect(
            all_texts, embeddings, ["unknown"] * len(all_texts), np.ones(len(all_texts)) * 0.5,
        )
        strategy_outputs["confidence"] = (flags, metrics)

        knn_config = KNNConfig(k=5, distance_threshold=0.5)
        knn_strategy = KNNDistanceStrategy()
        knn_strategy.initialize(known_emb, known_labels, knn_config)
        flags, metrics = knn_strategy.detect(
            all_texts, embeddings, ["unknown"] * len(all_texts), np.ones(len(all_texts)) * 0.5,
        )
        strategy_outputs["knn_distance"] = (flags, metrics)

        pattern_scorer = PatternScorer(known_texts)
        pattern_scores = np.array([pattern_scorer.score_novelty(t) for t in all_texts])
        pattern_flags = set()
        pattern_metrics = {}
        for i, s in enumerate(pattern_scores):
            is_novel = s < 0.5
            if is_novel:
                pattern_flags.add(i)
            pattern_metrics[i] = {"pattern_is_novel": is_novel, "pattern_score": 1.0 - s}
        strategy_outputs["pattern"] = (pattern_flags, pattern_metrics)

        sf_config = SetFitCentroidConfig()
        sf_strategy = SetFitCentroidStrategy()
        sf_strategy.initialize(known_emb, known_labels, sf_config)
        flags, metrics = sf_strategy.detect(
            all_texts, embeddings, ["unknown"] * len(all_texts), np.ones(len(all_texts)) * 0.5,
        )
        strategy_outputs["setfit_centroid"] = (flags, metrics)

        all_metrics = {}
        for i in range(len(all_texts)):
            all_metrics[i] = {}
        for _, (_, metrics) in strategy_outputs.items():
            for idx, m in metrics.items():
                all_metrics[idx].update(m)

        from novelentitymatcher.novelty.config.base import DetectionConfig
        from novelentitymatcher.novelty.config.weights import WeightConfig
        det_config = DetectionConfig(combine_method="weighted", weights=adjusted_weights)
        combiner = SignalCombiner(det_config)
        novel_indices, novelty_scores = combiner.combine(strategy_outputs, all_metrics)

        scores = np.array([novelty_scores.get(i, 0.0) for i in range(len(all_texts))])
        metrics_result = compute_metrics(true_labels, scores)

        return [
            BenchmarkResult(
                strategy="ensemble_adaptive",
                params={"method": "adaptive", "adjusted_weights": {
                    "confidence": adjusted_weights.confidence,
                    "knn": adjusted_weights.knn,
                    "pattern": adjusted_weights.pattern,
                    "setfit_centroid": adjusted_weights.setfit_centroid,
                }},
                train_time=0.0,
                inference_time=0.0,
                num_known=len(known_texts),
                num_ood=len(ood_texts),
                dataset="",
                **metrics_result,
            )
        ]


def run_benchmark():
    print("=" * 80)
    print("NOVELTY DETECTION STRATEGIES BENCHMARK")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Max samples per class: {MAX_SAMPLES}")
    print(f"OOD ratio: {OOD_RATIO}")
    print()

    benchmark = StrategyBenchmark(MODEL_NAME)
    all_results: list[BenchmarkResult] = []

    for ds_name in DATASETS:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 80}")

        if ds_name == "ag_news":
            df = load_dataset_data("ag_news")
            label_col = "label"
            class_names = ["World", "Sports", "Business", "Sci/Tech"]
        else:
            df = load_dataset_data("go_emotions")
            df = df[df["labels"].apply(lambda x: isinstance(x, list) and len(x) == 1)]
            df["labels"] = df["labels"].apply(lambda x: x[0])
            label_col = "labels"
            class_names = None

        known_data, ood_data, ood_labels = create_ood_split(
            df, label_col, OOD_RATIO, MAX_SAMPLES
        )

        print(f"Known samples: {len(known_data)}, OOD samples: {len(ood_data)}")
        print(f"OOD labels: {ood_labels}")

        known_texts = (
            known_data["text"].tolist()
            if "text" in known_data.columns
            else known_data[known_data.columns[0]].tolist()
        )
        ood_texts = (
            ood_data["text"].tolist()
            if "text" in ood_data.columns
            else ood_data[ood_data.columns[0]].tolist()
        )
        known_labels = known_data[label_col].astype(str).tolist()

        # Confidence strategy
        print("\n--- Confidence Strategy ---")
        results = benchmark.benchmark_confidence(known_texts, ood_texts, known_labels)
        for r in results:
            r.dataset = ds_name
            all_results.append(r)
            print(
                f"  threshold={r.params['threshold']:.2f}: AUROC={r.auroc:.3f}, DR@1%={r.detection_rate_1fp:.3f}"
            )

        # KNN Distance strategy
        print("\n--- KNN Distance Strategy ---")
        results = benchmark.benchmark_knn(known_texts, ood_texts, known_labels)
        for r in results:
            r.dataset = ds_name
            all_results.append(r)
        best_knn = max(results, key=lambda x: x.auroc)
        print(
            f"  Best: k={best_knn.params['k']}, thresh={best_knn.params['distance_threshold']:.2f}: AUROC={best_knn.auroc:.3f}, DR@1%={best_knn.detection_rate_1fp:.3f}"
        )

        # Mahalanobis strategy
        print("\n--- Mahalanobis Strategy ---")
        try:
            results = benchmark.benchmark_mahalanobis(
                known_texts, ood_texts, known_labels
            )
            for r in results:
                r.dataset = ds_name
                all_results.append(r)
            best_maha = max(results, key=lambda x: x.auroc)
            print(
                f"  Best: thresh={best_maha.params['threshold']:.1f}: AUROC={best_maha.auroc:.3f}, DR@1%={best_maha.detection_rate_1fp:.3f}"
            )
        except Exception as e:
            print(f"  Failed: {e}")

        # LOF strategy
        print("\n--- LOF Strategy ---")
        try:
            results = benchmark.benchmark_lof(known_texts, ood_texts, known_labels)
            for r in results:
                r.dataset = ds_name
                all_results.append(r)
            best_lof = max(results, key=lambda x: x.auroc)
            print(
                f"  Best: n={best_lof.params['n_neighbors']}, cont={best_lof.params['contamination']:.2f}: AUROC={best_lof.auroc:.3f}, DR@1%={best_lof.detection_rate_1fp:.3f}"
            )
        except Exception as e:
            print(f"  Failed: {e}")

        # Pattern strategy
        print("\n--- Pattern Strategy ---")
        try:
            results = benchmark.benchmark_pattern(known_texts, ood_texts, known_labels)
            for r in results:
                r.dataset = ds_name
                all_results.append(r)
            best_pattern = max(results, key=lambda x: x.auroc)
            print(
                f"  Best: thresh={best_pattern.params['threshold']:.2f}: AUROC={best_pattern.auroc:.3f}, DR@1%={best_pattern.detection_rate_1fp:.3f}"
            )
        except Exception as e:
            print(f"  Failed: {e}")

        # SetFit Centroid strategy
        print("\n--- SetFit Centroid Strategy ---")
        try:
            results = benchmark.benchmark_setfit_centroid(known_texts, ood_texts, known_labels)
            for r in results:
                r.dataset = ds_name
                all_results.append(r)
            for r in results:
                print(
                    f"  AUROC={r.auroc:.3f}, DR@1%={r.detection_rate_1fp:.3f}"
                )
        except Exception as e:
            print(f"  Failed: {e}")

        # Ensemble: weighted combination
        print("\n--- Ensemble: Weighted ---")
        try:
            results = benchmark.benchmark_ensemble_weighted(known_texts, ood_texts, known_labels)
            for r in results:
                r.dataset = ds_name
                all_results.append(r)
            for r in results:
                print(
                    f"  AUROC={r.auroc:.3f}, DR@1%={r.detection_rate_1fp:.3f}"
                )
        except Exception as e:
            print(f"  Failed: {e}")

        # Ensemble: voting
        print("\n--- Ensemble: Voting ---")
        try:
            results = benchmark.benchmark_ensemble_voting(known_texts, ood_texts, known_labels)
            for r in results:
                r.dataset = ds_name
                all_results.append(r)
            for r in results:
                print(
                    f"  AUROC={r.auroc:.3f}, DR@1%={r.detection_rate_1fp:.3f}"
                )
        except Exception as e:
            print(f"  Failed: {e}")

        # Ensemble: adaptive weights
        print("\n--- Ensemble: Adaptive Weights ---")
        try:
            results = benchmark.benchmark_ensemble_adaptive(known_texts, ood_texts, known_labels)
            for r in results:
                r.dataset = ds_name
                all_results.append(r)
            for r in results:
                print(
                    f"  AUROC={r.auroc:.3f}, DR@1%={r.detection_rate_1fp:.3f}"
                )
        except Exception as e:
            print(f"  Failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Best Results by Strategy")
    print("=" * 80)

    df_results = pd.DataFrame(
        [
            {
                "strategy": r.strategy,
                "dataset": r.dataset,
                "best_auroc": r.auroc,
                "best_auprc": r.auprc,
                "dr@1%": r.detection_rate_1fp,
                "dr@5%": r.detection_rate_5fp,
                "dr@10%": r.detection_rate_10fp,
                "params": str(r.params),
                "train_time": r.train_time,
                "inference_time": r.inference_time,
            }
            for r in all_results
        ]
    )

    # Group by strategy and find best
    summary = (
        df_results.groupby("strategy")
        .agg(
            {
                "best_auroc": "max",
                "best_auprc": "max",
                "dr@1%": "max",
                "dr@5%": "max",
                "dr@10%": "max",
                "train_time": "mean",
                "inference_time": "mean",
            }
        )
        .round(4)
    )

    print("\n", summary.to_string())

    return df_results


if __name__ == "__main__":
    results_df = run_benchmark()
    print("\nFull results saved to novelty_strategies_benchmark.csv")
    results_df.to_csv("novelty_strategies_benchmark.csv", index=False)
