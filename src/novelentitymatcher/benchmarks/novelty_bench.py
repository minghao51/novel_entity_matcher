"""Merged novelty detection benchmark with depth levels.

Consolidates:
- benchmark_full_pipeline.py (Phase 2)
- benchmark_novelty_strategies.py
- benchmark_novelty_full.py

Depth levels:
- ``quick``: KNN, Mahalanobis, LOF, OneClassSVM, IsolationForest
- ``standard``: quick + Pattern, SetFit Centroid, weighted/voting/adaptive ensembles
- ``full``: standard + hyperparameter tuning + SignalCombiner + meta-learner + adaptive weights
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .shared import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OOD_RATIO,
    DEFAULT_RANDOM_SEED,
    SplitData,
    compute_ood_metrics,
    prepare_binary_labels,
)

DATASET_CONFIGS = {
    "ag_news": {
        "hf_path": "ag_news",
        "label_col": "label",
        "text_col": "text",
        "classes": ["World", "Sports", "Business", "Sci/Tech"],
    },
    "go_emotions": {
        "hf_path": "go_emotions",
        "label_col": "labels",
        "text_col": "text",
        "classes": [
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
        ],
    },
}


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
    dataset: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def load_and_split_data(
    ds_name: str,
    max_train: int = 200,
    max_val: int = 200,
    max_test: int = 500,
    ood_ratio: float = DEFAULT_OOD_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SplitData:
    import random

    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    random.seed(random_seed)
    np.random.seed(random_seed)

    cfg = DATASET_CONFIGS.get(ds_name)
    if cfg is None:
        raise ValueError(
            f"Unknown dataset: {ds_name}. Available: {list(DATASET_CONFIGS.keys())}"
        )

    ds = load_dataset(cfg["hf_path"])
    split_key = "test" if "test" in ds else "train"
    df = pd.DataFrame(ds[split_key])

    if ds_name == "go_emotions":
        df = df[
            df[cfg["label_col"]].apply(lambda x: isinstance(x, list) and len(x) == 1)
        ].copy()
        df[cfg["label_col"]] = df[cfg["label_col"]].apply(lambda x: int(x[0]))

    unique_labels = sorted(df[cfg["label_col"]].unique())
    num_ood = max(1, int(len(unique_labels) * ood_ratio))
    ood_labels = set(random.sample(unique_labels, num_ood))
    known_labels = sorted(set(unique_labels) - ood_labels)

    known_df = df[df[cfg["label_col"]].isin(known_labels)]
    ood_df = df[df[cfg["label_col"]].isin(ood_labels)]

    train_df, temp_known = train_test_split(
        known_df, test_size=0.4, random_state=random_seed
    )
    val_df, test_known_df = train_test_split(
        temp_known, test_size=0.5, random_state=random_seed
    )

    _train_ood_df, temp_ood = train_test_split(
        ood_df, test_size=0.5, random_state=random_seed
    )
    val_ood_df, test_ood_df = train_test_split(
        temp_ood, test_size=0.5, random_state=random_seed
    )

    classes = cfg["classes"]
    text_col = cfg["text_col"]
    label_col = cfg["label_col"]

    def _resolve_label(raw_label):
        if isinstance(raw_label, (int, np.integer)) and int(raw_label) < len(classes):
            return classes[int(raw_label)]
        return str(raw_label)

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

    known_class_names = [_resolve_label(lbl) for lbl in known_labels]

    return SplitData(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        known_classes=known_class_names,
        ood_classes=[_resolve_label(lbl) for lbl in ood_labels],
    )


def _compute_knn_novelty(
    train_emb: np.ndarray, query_emb: np.ndarray, k: int = 5
) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(query_emb, train_emb)
    k = min(k, len(train_emb))
    top_k_sims = np.sort(sims, axis=1)[:, -k:]
    return 1.0 - top_k_sims.mean(axis=1)


def _compute_mahalanobis_novelty(
    embeddings: np.ndarray,
    class_means: dict[str, np.ndarray],
    cov_inv: np.ndarray,
) -> np.ndarray:
    global_mean = np.mean(list(class_means.values()), axis=0)
    scores = []
    for emb in embeddings:
        diff = emb - global_mean
        scores.append(np.sqrt(np.abs(diff @ cov_inv @ diff)))
    return np.array(scores)


def _run_strategies_for_split(
    strategy_outputs: dict[str, tuple],
    all_metrics: dict[int, dict],
    true_labels: np.ndarray,
    combine_method: str,
    config: Any = None,
) -> dict[str, float]:
    from ..novelty.config.base import DetectionConfig
    from ..novelty.core.signal_combiner import SignalCombiner

    det_config = DetectionConfig(combine_method=combine_method, weights=config)
    combiner = SignalCombiner(det_config)
    _, novelty_scores = combiner.combine(strategy_outputs, all_metrics)
    scores = np.array([novelty_scores.get(i, 0.0) for i in range(len(true_labels))])
    return compute_ood_metrics(true_labels, scores)


def _build_strategy_outputs(
    model: SentenceTransformer,
    texts: list[str],
    train_texts: list[str],
    train_emb: np.ndarray,
    train_labels: list[str],
) -> tuple[dict[str, tuple], dict[int, dict]]:
    from ..novelty.config.strategies import (
        ConfidenceConfig,
        KNNConfig,
        SetFitCentroidConfig,
    )
    from ..novelty.strategies.confidence import ConfidenceStrategy
    from ..novelty.strategies.knn_distance import KNNDistanceStrategy
    from ..novelty.strategies.pattern_impl import PatternScorer
    from ..novelty.strategies.setfit_centroid import SetFitCentroidStrategy

    embeddings = np.asarray(model.encode(texts, show_progress_bar=False))
    strategy_outputs: dict[str, tuple] = {}

    conf_config = ConfidenceConfig(threshold=0.5)
    conf_strategy = ConfidenceStrategy()
    conf_strategy.initialize(train_emb, train_labels, conf_config)
    flags, metrics = conf_strategy.detect(
        texts, embeddings, ["unknown"] * len(texts), np.ones(len(texts)) * 0.5
    )
    strategy_outputs["confidence"] = (flags, metrics)

    knn_config = KNNConfig(k=5, distance_threshold=0.5)
    knn_strategy = KNNDistanceStrategy()
    knn_strategy.initialize(train_emb, train_labels, knn_config)
    flags, metrics = knn_strategy.detect(
        texts, embeddings, ["unknown"] * len(texts), np.ones(len(texts)) * 0.5
    )
    strategy_outputs["knn_distance"] = (flags, metrics)

    pattern_scorer = PatternScorer(train_texts)
    pattern_scores = np.array([pattern_scorer.score_novelty(t) for t in texts])
    pattern_flags: set[int] = set()
    pattern_metrics: dict[int, dict] = {}
    for i, s in enumerate(pattern_scores):
        is_novel = s < 0.5
        if is_novel:
            pattern_flags.add(i)
        pattern_metrics[i] = {"pattern_is_novel": is_novel, "pattern_score": 1.0 - s}
    strategy_outputs["pattern"] = (pattern_flags, pattern_metrics)

    sf_config = SetFitCentroidConfig()
    sf_strategy = SetFitCentroidStrategy()
    sf_strategy.initialize(train_emb, train_labels, sf_config)
    flags, metrics = sf_strategy.detect(
        texts, embeddings, ["unknown"] * len(texts), np.ones(len(texts)) * 0.5
    )
    strategy_outputs["setfit_centroid"] = (flags, metrics)

    all_metrics: dict[int, dict] = {i: {} for i in range(len(texts))}
    for _, (_, m) in strategy_outputs.items():
        for idx, mm in m.items():
            all_metrics[idx].update(mm)

    return strategy_outputs, all_metrics


class NoveltyBenchmark:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

    def _make_result(
        self,
        strategy: str,
        params: dict,
        val_m: dict,
        test_m: dict,
        train_time: float = 0.0,
        inference_time: float = 0.0,
    ) -> StrategyResult:
        return StrategyResult(
            strategy=strategy,
            params=params,
            train_time=train_time,
            inference_time=inference_time,
            val_auroc=val_m["auroc"],
            val_auprc=val_m["auprc"],
            val_dr_1fp=val_m["dr_1fp"],
            val_dr_5fp=val_m["dr_5fp"],
            val_dr_10fp=val_m["dr_10fp"],
            test_auroc=test_m["auroc"],
            test_auprc=test_m["auprc"],
            test_dr_1fp=test_m["dr_1fp"],
            test_dr_5fp=test_m["dr_5fp"],
            test_dr_10fp=test_m["dr_10fp"],
        )

    def benchmark_knn(
        self, split: SplitData, k_values: list[int] | None = None
    ) -> list[StrategyResult]:
        if k_values is None:
            k_values = [3, 5, 10, 20, 30]

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        results = []
        for k in k_values:
            val_novelty = _compute_knn_novelty(train_emb, val_emb, k=k)
            test_novelty = _compute_knn_novelty(train_emb, test_emb, k=k)
            val_m = compute_ood_metrics(val_true, val_novelty)
            test_m = compute_ood_metrics(test_true, test_novelty)
            results.append(self._make_result("knn_distance", {"k": k}, val_m, test_m))
        return results

    def benchmark_mahalanobis(self, split: SplitData) -> list[StrategyResult]:
        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        class_means = {}
        for lbl in set(split.train_labels):
            mask = np.array([s == lbl for s in split.train_labels])
            class_means[lbl] = train_emb[mask].mean(axis=0)
        global_cov = np.cov(train_emb, rowvar=False) + 1e-6 * np.eye(train_emb.shape[1])
        cov_inv = np.linalg.inv(global_cov)

        val_novelty = _compute_mahalanobis_novelty(val_emb, class_means, cov_inv)
        test_novelty = _compute_mahalanobis_novelty(test_emb, class_means, cov_inv)
        val_m = compute_ood_metrics(val_true, val_novelty)
        test_m = compute_ood_metrics(test_true, test_novelty)
        return [self._make_result("mahalanobis", {}, val_m, test_m)]

    def benchmark_lof(
        self, split: SplitData, n_neighbors_list: list[int] | None = None
    ) -> list[StrategyResult]:
        from sklearn.neighbors import LocalOutlierFactor

        if n_neighbors_list is None:
            n_neighbors_list = [10, 20, 30]
        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        results = []
        for n in n_neighbors_list:
            try:
                lof = LocalOutlierFactor(n_neighbors=n, contamination=0.1, novelty=True)
                lof.fit(train_emb)
                val_scores = -lof.score_samples(val_emb)
                test_scores = -lof.score_samples(test_emb)
                val_m = compute_ood_metrics(val_true, val_scores)
                test_m = compute_ood_metrics(test_true, test_scores)
                results.append(
                    self._make_result("lof", {"n_neighbors": n}, val_m, test_m)
                )
            except (ValueError, RuntimeError):
                pass
        return results

    def benchmark_oneclass_svm(
        self, split: SplitData, nu_values: list[float] | None = None
    ) -> list[StrategyResult]:
        from sklearn.svm import OneClassSVM

        if nu_values is None:
            nu_values = [0.05, 0.1, 0.2]
        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        results = []
        for nu in nu_values:
            try:
                start = time.perf_counter()
                ocsvm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
                ocsvm.fit(train_emb)
                train_time = time.perf_counter() - start

                val_scores = -ocsvm.decision_function(val_emb)
                test_scores = -ocsvm.decision_function(test_emb)
                val_m = compute_ood_metrics(val_true, val_scores)
                test_m = compute_ood_metrics(test_true, test_scores)
                results.append(
                    self._make_result(
                        "oneclass_svm", {"nu": nu}, val_m, test_m, train_time=train_time
                    )
                )
            except (ValueError, RuntimeError):
                pass
        return results

    def benchmark_isolation_forest(self, split: SplitData) -> list[StrategyResult]:
        from sklearn.ensemble import IsolationForest

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        results = []
        for n_est in [50, 100]:
            start = time.perf_counter()
            iso = IsolationForest(
                n_estimators=n_est, contamination=0.1, random_state=42
            )
            iso.fit(train_emb)
            train_time = time.perf_counter() - start

            val_scores = -iso.score_samples(val_emb)
            test_scores = -iso.score_samples(test_emb)
            val_m = compute_ood_metrics(val_true, val_scores)
            test_m = compute_ood_metrics(test_true, test_scores)
            results.append(
                self._make_result(
                    "isolation_forest",
                    {"n_estimators": n_est},
                    val_m,
                    test_m,
                    train_time=train_time,
                )
            )
        return results

    def benchmark_pattern(self, split: SplitData) -> list[StrategyResult]:
        from ..novelty.strategies.pattern_impl import PatternScorer

        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        scorer = PatternScorer(split.train_texts)
        val_scores = np.array([scorer.score_novelty(t) for t in split.val_texts])
        test_scores = np.array([scorer.score_novelty(t) for t in split.test_texts])
        val_m = compute_ood_metrics(val_true, val_scores)
        test_m = compute_ood_metrics(test_true, test_scores)
        return [self._make_result("pattern", {}, val_m, test_m)]

    def benchmark_setfit_centroid(
        self,
        split: SplitData,
        percentile_values: list[float] | None = None,
    ) -> list[StrategyResult]:
        from ..novelty.config.strategies import SetFitCentroidConfig
        from ..novelty.strategies.setfit_centroid import SetFitCentroidStrategy

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        results = []
        for pct in percentile_values or [95]:
            config = SetFitCentroidConfig()
            strategy = SetFitCentroidStrategy()
            strategy.initialize(train_emb, split.train_labels, config)

            _, val_metrics = strategy.detect(
                split.val_texts,
                val_emb,
                ["unknown"] * len(split.val_texts),
                np.ones(len(split.val_texts)) * 0.5,
            )
            _, test_metrics = strategy.detect(
                split.test_texts,
                test_emb,
                ["unknown"] * len(split.test_texts),
                np.ones(len(split.test_texts)) * 0.5,
            )

            val_novelty = np.array(
                [
                    val_metrics[i].get("setfit_centroid_novelty_score", 0.0)
                    for i in range(len(split.val_texts))
                ]
            )
            test_novelty = np.array(
                [
                    test_metrics[i].get("setfit_centroid_novelty_score", 0.0)
                    for i in range(len(split.test_texts))
                ]
            )
            val_m = compute_ood_metrics(val_true, val_novelty)
            test_m = compute_ood_metrics(test_true, test_novelty)
            results.append(
                self._make_result("setfit_centroid", {"percentile": pct}, val_m, test_m)
            )
        return results

    def benchmark_uncertainty(
        self,
        split: SplitData,
        margin_thresholds: list[float] | None = None,
        entropy_thresholds: list[float] | None = None,
    ) -> list[StrategyResult]:
        from ..novelty.config.strategies import UncertaintyConfig
        from ..novelty.strategies.uncertainty import UncertaintyStrategy

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        dummy_classes = ["unknown"] * max(len(split.val_texts), len(split.test_texts))
        dummy_conf = np.ones(max(len(split.val_texts), len(split.test_texts))) * 0.5

        results = []
        margins = margin_thresholds or [0.1, 0.2, 0.3, 0.4, 0.5]
        entropies = entropy_thresholds or [0.5, 1.0, 1.5, 2.0]

        for margin_t in margins:
            for entropy_t in entropies:
                config = UncertaintyConfig(
                    margin_threshold=margin_t, entropy_threshold=entropy_t
                )
                strategy = UncertaintyStrategy()
                strategy.initialize(train_emb, split.train_labels, config)

                _, val_metrics = strategy.detect(
                    split.val_texts,
                    val_emb,
                    dummy_classes[: len(split.val_texts)],
                    dummy_conf[: len(split.val_texts)],
                )
                _, test_metrics = strategy.detect(
                    split.test_texts,
                    test_emb,
                    dummy_classes[: len(split.test_texts)],
                    dummy_conf[: len(split.test_texts)],
                )

                val_novelty = np.array(
                    [
                        val_metrics[i].get("uncertainty_score", 0.0)
                        for i in range(len(split.val_texts))
                    ]
                )
                test_novelty = np.array(
                    [
                        test_metrics[i].get("uncertainty_score", 0.0)
                        for i in range(len(split.test_texts))
                    ]
                )
                val_m = compute_ood_metrics(val_true, val_novelty)
                test_m = compute_ood_metrics(test_true, test_novelty)
                results.append(
                    self._make_result(
                        "uncertainty",
                        {"margin_threshold": margin_t, "entropy_threshold": entropy_t},
                        val_m,
                        test_m,
                    )
                )
        return results

    def benchmark_self_knowledge(
        self,
        split: SplitData,
        hidden_dims: list[int] | None = None,
        epoch_values: list[int] | None = None,
    ) -> list[StrategyResult]:
        from ..novelty.config.strategies import SelfKnowledgeConfig
        from ..novelty.strategies.self_knowledge import SelfKnowledgeStrategy

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        dummy_classes = ["unknown"] * max(len(split.val_texts), len(split.test_texts))
        dummy_conf = np.ones(max(len(split.val_texts), len(split.test_texts))) * 0.5

        results = []
        dims = hidden_dims or [64, 128, 256]
        epochs = epoch_values or [50, 100, 200]

        for hd in dims:
            for ep in epochs:
                try:
                    config = SelfKnowledgeConfig(hidden_dim=hd, epochs=ep)
                    strategy = SelfKnowledgeStrategy()

                    start = time.perf_counter()
                    strategy.initialize(train_emb, split.train_labels, config)
                    train_time = time.perf_counter() - start

                    _, val_metrics = strategy.detect(
                        split.val_texts,
                        val_emb,
                        dummy_classes[: len(split.val_texts)],
                        dummy_conf[: len(split.val_texts)],
                    )
                    _, test_metrics = strategy.detect(
                        split.test_texts,
                        test_emb,
                        dummy_classes[: len(split.test_texts)],
                        dummy_conf[: len(split.test_texts)],
                    )

                    val_novelty = np.array(
                        [
                            val_metrics[i].get("self_knowledge_novelty_score", 0.0)
                            for i in range(len(split.val_texts))
                        ]
                    )
                    test_novelty = np.array(
                        [
                            test_metrics[i].get("self_knowledge_novelty_score", 0.0)
                            for i in range(len(split.test_texts))
                        ]
                    )
                    val_m = compute_ood_metrics(val_true, val_novelty)
                    test_m = compute_ood_metrics(test_true, test_novelty)
                    results.append(
                        self._make_result(
                            "self_knowledge",
                            {"hidden_dim": hd, "epochs": ep},
                            val_m,
                            test_m,
                            train_time=train_time,
                        )
                    )
                except (ValueError, RuntimeError) as e:
                    print(f"  self_knowledge hidden_dim={hd} epochs={ep} failed: {e}")
        return results

    def benchmark_prototypical(
        self,
        split: SplitData,
        distance_thresholds: list[float] | None = None,
    ) -> list[StrategyResult]:
        from ..novelty.config.strategies import PrototypicalConfig
        from ..novelty.strategies.prototypical import PrototypicalStrategy

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        results = []
        thresholds = distance_thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]

        for dt in thresholds:
            try:
                config = PrototypicalConfig(
                    distance_threshold=dt, model_name=self.model_name
                )
                strategy = PrototypicalStrategy()

                start = time.perf_counter()
                strategy.initialize(train_emb, split.train_labels, config)
                train_time = time.perf_counter() - start

                _, val_metrics = strategy.detect(
                    split.val_texts,
                    val_emb,
                    ["unknown"] * len(split.val_texts),
                    np.ones(len(split.val_texts)) * 0.5,
                )
                _, test_metrics = strategy.detect(
                    split.test_texts,
                    test_emb,
                    ["unknown"] * len(split.test_texts),
                    np.ones(len(split.test_texts)) * 0.5,
                )

                val_novelty = np.array(
                    [
                        val_metrics[i].get("prototypical_novelty_score", 0.0)
                        for i in range(len(split.val_texts))
                    ]
                )
                test_novelty = np.array(
                    [
                        test_metrics[i].get("prototypical_novelty_score", 0.0)
                        for i in range(len(split.test_texts))
                    ]
                )
                val_m = compute_ood_metrics(val_true, val_novelty)
                test_m = compute_ood_metrics(test_true, test_novelty)
                results.append(
                    self._make_result(
                        "prototypical",
                        {"distance_threshold": dt},
                        val_m,
                        test_m,
                        train_time=train_time,
                    )
                )
            except (ValueError, RuntimeError) as e:
                print(f"  prototypical threshold={dt} failed: {e}")
        return results

    def benchmark_setfit(
        self,
        split: SplitData,
        margin_values: list[float] | None = None,
    ) -> list[StrategyResult]:
        from ..novelty.config.strategies import SetFitConfig
        from ..novelty.strategies.setfit import SetFitStrategy

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        results = []
        margins = margin_values or [0.3, 0.5, 0.7, 1.0]

        for margin in margins:
            try:
                config = SetFitConfig(margin=margin, model_name=self.model_name)
                strategy = SetFitStrategy()

                start = time.perf_counter()
                strategy.initialize(train_emb, split.train_labels, config)
                train_time = time.perf_counter() - start

                _, val_metrics = strategy.detect(
                    split.val_texts,
                    val_emb,
                    ["unknown"] * len(split.val_texts),
                    np.ones(len(split.val_texts)) * 0.5,
                )
                _, test_metrics = strategy.detect(
                    split.test_texts,
                    test_emb,
                    ["unknown"] * len(split.test_texts),
                    np.ones(len(split.test_texts)) * 0.5,
                )

                val_novelty = np.array(
                    [
                        val_metrics[i].get("setfit_novelty_score", 0.0)
                        for i in range(len(split.val_texts))
                    ]
                )
                test_novelty = np.array(
                    [
                        test_metrics[i].get("setfit_novelty_score", 0.0)
                        for i in range(len(split.test_texts))
                    ]
                )
                val_m = compute_ood_metrics(val_true, val_novelty)
                test_m = compute_ood_metrics(test_true, test_novelty)
                results.append(
                    self._make_result(
                        "setfit",
                        {"margin": margin},
                        val_m,
                        test_m,
                        train_time=train_time,
                    )
                )
            except (ValueError, RuntimeError) as e:
                print(f"  setfit margin={margin} failed: {e}")
        return results

    def benchmark_mahalanobis_conformal(
        self,
        split: SplitData,
        calibration_methods: list[str] | None = None,
        alpha_values: list[float] | None = None,
    ) -> list[StrategyResult]:
        from ..novelty.config.strategies import MahalanobisConfig
        from ..novelty.strategies.mahalanobis import MahalanobisDistanceStrategy

        train_emb = self.encode_texts(split.train_texts)
        val_emb = self.encode_texts(split.val_texts)
        test_emb = self.encode_texts(split.test_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        results = []
        methods = calibration_methods or ["split", "mondrian"]
        alphas = alpha_values or [0.05, 0.1, 0.15, 0.2]

        for method in methods:
            for alpha in alphas:
                try:
                    config = MahalanobisConfig(
                        calibration_mode="conformal",
                        calibration_method=method,  # type: ignore[arg-type]
                        calibration_alpha=alpha,
                    )
                    strategy = MahalanobisDistanceStrategy()

                    start = time.perf_counter()
                    strategy.initialize(train_emb, split.train_labels, config)
                    train_time = time.perf_counter() - start

                    _, val_metrics = strategy.detect(
                        split.val_texts,
                        val_emb,
                        ["unknown"] * len(split.val_texts),
                        np.ones(len(split.val_texts)) * 0.5,
                    )
                    _, test_metrics = strategy.detect(
                        split.test_texts,
                        test_emb,
                        ["unknown"] * len(split.test_texts),
                        np.ones(len(split.test_texts)) * 0.5,
                    )

                    val_scores = np.array(
                        [
                            val_metrics[i].get(
                                "p_value",
                                val_metrics[i].get("mahalanobis_novelty_score", 0.0),
                            )
                            for i in range(len(split.val_texts))
                        ]
                    )
                    test_scores = np.array(
                        [
                            test_metrics[i].get(
                                "p_value",
                                test_metrics[i].get("mahalanobis_novelty_score", 0.0),
                            )
                            for i in range(len(split.test_texts))
                        ]
                    )
                    val_m = compute_ood_metrics(val_true, val_scores)
                    test_m = compute_ood_metrics(test_true, test_scores)
                    results.append(
                        self._make_result(
                            "mahalanobis_conformal",
                            {"method": method, "alpha": alpha},
                            val_m,
                            test_m,
                            train_time=train_time,
                        )
                    )
                except (ValueError, RuntimeError) as e:
                    print(
                        f"  mahalanobis_conformal method={method} alpha={alpha} failed: {e}"
                    )
        return results

    def benchmark_ensemble(self, split: SplitData) -> list[StrategyResult]:
        train_emb = self.encode_texts(split.train_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        val_outputs, val_all_metrics = _build_strategy_outputs(
            self.model,
            split.val_texts,
            split.train_texts,
            train_emb,
            split.train_labels,
        )
        test_outputs, test_all_metrics = _build_strategy_outputs(
            self.model,
            split.test_texts,
            split.train_texts,
            train_emb,
            split.train_labels,
        )

        results = []
        for method in ["weighted", "voting"]:
            val_m = _run_strategies_for_split(
                val_outputs, val_all_metrics, val_true, method
            )
            test_m = _run_strategies_for_split(
                test_outputs, test_all_metrics, test_true, method
            )
            results.append(
                self._make_result(
                    f"ensemble_{method}", {"method": method}, val_m, test_m
                )
            )

        return results

    def benchmark_ensemble_adaptive(self, split: SplitData) -> list[StrategyResult]:
        from ..novelty.core.adaptive_weights import (
            adaptive_weights,
            compute_characteristics,
        )

        train_emb = self.encode_texts(split.train_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        characteristics = compute_characteristics(train_emb, split.train_labels)
        adjusted_weights = adaptive_weights(characteristics)

        val_outputs, val_all_metrics = _build_strategy_outputs(
            self.model,
            split.val_texts,
            split.train_texts,
            train_emb,
            split.train_labels,
        )
        test_outputs, test_all_metrics = _build_strategy_outputs(
            self.model,
            split.test_texts,
            split.train_texts,
            train_emb,
            split.train_labels,
        )

        val_m = _run_strategies_for_split(
            val_outputs, val_all_metrics, val_true, "weighted", config=adjusted_weights
        )
        test_m = _run_strategies_for_split(
            test_outputs,
            test_all_metrics,
            test_true,
            "weighted",
            config=adjusted_weights,
        )
        return [
            self._make_result(
                "ensemble_adaptive",
                {
                    "confidence": adjusted_weights.confidence,
                    "knn": adjusted_weights.knn,
                    "pattern": adjusted_weights.pattern,
                    "setfit_centroid": adjusted_weights.setfit_centroid,
                },
                val_m,
                test_m,
            )
        ]

    def benchmark_signal_combiner(
        self, split: SplitData, method: str = "weighted"
    ) -> list[StrategyResult]:
        from ..novelty.config.base import DetectionConfig
        from ..novelty.config.weights import WeightConfig
        from ..novelty.core.signal_combiner import SignalCombiner

        train_emb = self.encode_texts(split.train_texts)
        val_true = prepare_binary_labels(split.val_labels, "__OOD__")
        test_true = prepare_binary_labels(split.test_labels, "__OOD__")

        val_outputs, val_all_metrics = _build_strategy_outputs(
            self.model,
            split.val_texts,
            split.train_texts,
            train_emb,
            split.train_labels,
        )
        test_outputs, test_all_metrics = _build_strategy_outputs(
            self.model,
            split.test_texts,
            split.train_texts,
            train_emb,
            split.train_labels,
        )

        weights = WeightConfig()
        det_config = DetectionConfig(combine_method=method, weights=weights)
        combiner = SignalCombiner(det_config)

        if method == "meta_learner":
            train_features = []
            train_labels_arr = []
            for idx in sorted(val_all_metrics.keys()):
                feats = combiner._extract_features(idx, val_all_metrics)
                train_features.append(feats)
                train_labels_arr.append(val_true[idx])
            if len(set(train_labels_arr)) > 1:
                combiner.train_meta_learner(
                    np.array(train_features), np.array(train_labels_arr)
                )

        _, val_scores = combiner.combine(val_outputs, val_all_metrics)
        _, test_scores = combiner.combine(test_outputs, test_all_metrics)
        val_novelty = np.array(
            [val_scores.get(i, 0.0) for i in range(len(split.val_texts))]
        )
        test_novelty = np.array(
            [test_scores.get(i, 0.0) for i in range(len(split.test_texts))]
        )

        val_m = compute_ood_metrics(val_true, val_novelty)
        test_m = compute_ood_metrics(test_true, test_novelty)
        return [
            self._make_result(
                f"signal_combiner_{method}", {"method": method}, val_m, test_m
            )
        ]

    def run_depth(
        self,
        split: SplitData,
        depth: str = "standard",
        ds_name: str = "",
    ) -> list[StrategyResult]:
        all_results: list[StrategyResult] = []

        print("\n--- KNN Distance ---")
        results = self.benchmark_knn(split)
        for r in results:
            r.dataset = ds_name
            all_results.append(r)
        if results:
            best = max(results, key=lambda x: x.val_auroc)
            print(
                f"  Best k={best.params['k']}: Val AUROC={best.val_auroc:.3f}, Test AUROC={best.test_auroc:.3f}"
            )

        print("\n--- Mahalanobis ---")
        results = self.benchmark_mahalanobis(split)
        for r in results:
            r.dataset = ds_name
            all_results.append(r)
        if results:
            print(
                f"  Val AUROC={results[0].val_auroc:.3f}, Test AUROC={results[0].test_auroc:.3f}"
            )

        print("\n--- LOF ---")
        results = self.benchmark_lof(split)
        for r in results:
            r.dataset = ds_name
            all_results.append(r)

        print("\n--- One-Class SVM ---")
        results = self.benchmark_oneclass_svm(split)
        for r in results:
            r.dataset = ds_name
            all_results.append(r)

        print("\n--- Isolation Forest ---")
        results = self.benchmark_isolation_forest(split)
        for r in results:
            r.dataset = ds_name
            all_results.append(r)

        print("\n--- Uncertainty ---")
        try:
            results = self.benchmark_uncertainty(split)
            for r in results:
                r.dataset = ds_name
                all_results.append(r)
            if results:
                best = max(results, key=lambda x: x.val_auroc)
                print(
                    f"  Best: margin={best.params.get('margin_threshold')}, entropy={best.params.get('entropy_threshold')}, "
                    f"Val AUROC={best.val_auroc:.3f}, Test AUROC={best.test_auroc:.3f}"
                )
        except (ValueError, RuntimeError) as e:
            print(f"  Failed: {e}")

        print("\n--- Self-Knowledge ---")
        try:
            results = self.benchmark_self_knowledge(split)
            for r in results:
                r.dataset = ds_name
                all_results.append(r)
            if results:
                best = max(results, key=lambda x: x.val_auroc)
                print(
                    f"  Best: hidden_dim={best.params.get('hidden_dim')}, epochs={best.params.get('epochs')}, "
                    f"Val AUROC={best.val_auroc:.3f}, Test AUROC={best.test_auroc:.3f}"
                )
        except (ValueError, RuntimeError) as e:
            print(f"  Failed: {e}")

        if depth in ("standard", "full"):
            print("\n--- Pattern ---")
            results = self.benchmark_pattern(split)
            for r in results:
                r.dataset = ds_name
                all_results.append(r)

            print("\n--- SetFit Centroid ---")
            try:
                results = self.benchmark_setfit_centroid(split)
                for r in results:
                    r.dataset = ds_name
                    all_results.append(r)
            except (ValueError, RuntimeError) as e:
                print(f"  Failed: {e}")

            print("\n--- Prototypical ---")
            try:
                results = self.benchmark_prototypical(split)
                for r in results:
                    r.dataset = ds_name
                    all_results.append(r)
                if results:
                    best = max(results, key=lambda x: x.val_auroc)
                    print(
                        f"  Best threshold={best.params.get('distance_threshold')}: "
                        f"Val AUROC={best.val_auroc:.3f}, Test AUROC={best.test_auroc:.3f}"
                    )
            except (ValueError, RuntimeError) as e:
                print(f"  Failed: {e}")

            print("\n--- SetFit Contrastive ---")
            try:
                results = self.benchmark_setfit(split)
                for r in results:
                    r.dataset = ds_name
                    all_results.append(r)
                if results:
                    best = max(results, key=lambda x: x.val_auroc)
                    print(
                        f"  Best margin={best.params.get('margin')}: "
                        f"Val AUROC={best.val_auroc:.3f}, Test AUROC={best.test_auroc:.3f}"
                    )
            except (ValueError, RuntimeError) as e:
                print(f"  Failed: {e}")

            print("\n--- Mahalanobis (Conformal) ---")
            try:
                results = self.benchmark_mahalanobis_conformal(split)
                for r in results:
                    r.dataset = ds_name
                    all_results.append(r)
                if results:
                    best = max(results, key=lambda x: x.val_auroc)
                    print(
                        f"  Best: method={best.params.get('method')}, alpha={best.params.get('alpha')}, "
                        f"Val AUROC={best.val_auroc:.3f}, Test AUROC={best.test_auroc:.3f}"
                    )
            except (ValueError, RuntimeError) as e:
                print(f"  Failed: {e}")

            print("\n--- Ensemble: Weighted & Voting ---")
            results = self.benchmark_ensemble(split)
            for r in results:
                r.dataset = ds_name
                all_results.append(r)

            print("\n--- Ensemble: Adaptive ---")
            try:
                results = self.benchmark_ensemble_adaptive(split)
                for r in results:
                    r.dataset = ds_name
                    all_results.append(r)
            except (ValueError, RuntimeError) as e:
                print(f"  Failed: {e}")

        if depth == "full":
            for method in ["weighted", "voting", "meta_learner"]:
                print(f"\n--- Signal Combiner: {method} ---")
                try:
                    results = self.benchmark_signal_combiner(split, method=method)
                    for r in results:
                        r.dataset = ds_name
                        all_results.append(r)
                except (ValueError, RuntimeError) as e:
                    print(f"  Failed: {e}")

        return all_results


def run_benchmark(
    depth: str = "standard",
    datasets: list[str] | None = None,
    max_train: int = 200,
    max_val: int = 200,
    max_test: int = 500,
    model_name: str = DEFAULT_MODEL_NAME,
    models: list[str] | None = None,
    output: str | None = None,
) -> pd.DataFrame:
    model_list = models or [model_name]

    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())

    print("=" * 80)
    print(f"NOVELTY DETECTION BENCHMARK (depth={depth})")
    print(f"Models: {model_list}")
    print(f"Datasets: {datasets}")
    print("=" * 80)

    all_results: list[StrategyResult] = []

    for mdl in model_list:
        print(f"\n{'#' * 80}")
        print(f"Model: {mdl}")
        print(f"{'#' * 80}")

        benchmark = NoveltyBenchmark(mdl)

        for ds_name in datasets:
            print(f"\n{'=' * 80}")
            print(f"Dataset: {ds_name}")
            print(f"{'=' * 80}")

            split = load_and_split_data(ds_name, max_train, max_val, max_test)
            print(
                f"Train: {len(split.train_texts)}, Val: {len(split.val_texts)}, Test: {len(split.test_texts)}"
            )
            print(f"Known: {split.known_classes}, OOD: {split.ood_classes}")

            results = benchmark.run_depth(split, depth=depth, ds_name=ds_name)
            for r in results:
                r.extra["model"] = mdl
            all_results.extend(results)

    records = [
        {
            "strategy": r.strategy,
            "dataset": r.dataset,
            "model": r.extra.get("model", model_name),
            "params": str(r.params),
            "val_auroc": r.val_auroc,
            "val_auprc": r.val_auprc,
            "val_dr_1fp": r.val_dr_1fp,
            "test_auroc": r.test_auroc,
            "test_auprc": r.test_auprc,
            "test_dr_1fp": r.test_dr_1fp,
            "test_dr_5fp": r.test_dr_5fp,
            "test_dr_10fp": r.test_dr_10fp,
            "train_time": r.train_time,
        }
        for r in all_results
    ]
    df = pd.DataFrame(records)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Strategy':<25} {'Model':<20} {'Val AUROC':>10} {'Test AUROC':>12} {'Test DR@1%':>12}"
    )
    print("-" * 80)
    by_key: dict[str, list[StrategyResult]] = {}
    for r in all_results:
        key = f"{r.strategy}|{r.extra.get('model', model_name)}"
        by_key.setdefault(key, []).append(r)
    for key, results in sorted(by_key.items()):
        best = max(results, key=lambda x: x.val_auroc)
        strategy, model = key.split("|", 1)
        print(
            f"{strategy:<25} {model[-20:]:<20} {best.val_auroc:>10.3f} {best.test_auroc:>12.3f} {best.test_dr_1fp:>12.3f}"
        )

    if output:
        df.to_csv(output, index=False)
        print(f"\nResults saved to {output}")

    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Novelty detection benchmark")
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "full"],
        default="standard",
        help="Benchmark depth: quick (basic), standard (+ ensembles), full (+ signal combiner + meta-learner)",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME, help="Single model to benchmark"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Multiple models to benchmark (overrides --model)",
    )
    parser.add_argument("--output", default=None, help="Output CSV path")

    args = parser.parse_args(argv)
    run_benchmark(
        depth=args.depth,
        datasets=args.datasets,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        model_name=args.model,
        models=args.models,
        output=args.output,
    )
    return 0
