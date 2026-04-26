"""Bayesian optimization of ensemble weights using Optuna.

Searches for optimal strategy weights and thresholds that maximize
AUROC on validation data. Compares weighted/voting/meta_learner
combination methods.

Usage:
    novelentitymatcher-bench bench-weights --trials 200 --dataset ag_news
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from .novelty_bench import (
    DATASET_CONFIGS,
    StrategyResult,
    load_and_split_data,
)
from .shared import (
    DEFAULT_MODEL_NAME,
    compute_ood_metrics,
    prepare_binary_labels,
)


WEIGHT_FIELDS = [
    "confidence",
    "uncertainty",
    "knn",
    "cluster",
    "self_knowledge",
    "pattern",
    "oneclass",
    "prototypical",
    "setfit",
    "setfit_centroid",
    "mahalanobis",
    "lof",
]


def _run_all_strategies(
    model: Any,
    split: Any,
    train_emb: np.ndarray,
) -> tuple[dict[str, tuple], dict[int, dict], dict[str, tuple], dict[int, dict]]:
    from ..novelty.config.strategies import (
        ConfidenceConfig,
        KNNConfig,
        SetFitCentroidConfig,
        UncertaintyConfig,
        SelfKnowledgeConfig,
        MahalanobisConfig,
        LOFConfig,
        OneClassConfig,
        PatternConfig,
    )
    from ..novelty.strategies.confidence import ConfidenceStrategy
    from ..novelty.strategies.knn_distance import KNNDistanceStrategy
    from ..novelty.strategies.setfit_centroid import SetFitCentroidStrategy
    from ..novelty.strategies.uncertainty import UncertaintyStrategy
    from ..novelty.strategies.self_knowledge import SelfKnowledgeStrategy
    from ..novelty.strategies.mahalanobis import MahalanobisDistanceStrategy
    from ..novelty.strategies.lof import LOFStrategy
    from ..novelty.strategies.oneclass import OneClassStrategy
    from ..novelty.strategies.pattern import PatternStrategy

    strategies_configs = [
        ("confidence", ConfidenceStrategy, ConfidenceConfig()),
        ("knn_distance", KNNDistanceStrategy, KNNConfig()),
        ("setfit_centroid", SetFitCentroidStrategy, SetFitCentroidConfig()),
        ("uncertainty", UncertaintyStrategy, UncertaintyConfig()),
        ("self_knowledge", SelfKnowledgeStrategy, SelfKnowledgeConfig()),
        ("mahalanobis", MahalanobisDistanceStrategy, MahalanobisConfig()),
        ("lof", LOFStrategy, LOFConfig()),
        ("oneclass", OneClassStrategy, OneClassConfig()),
        ("pattern", PatternStrategy, PatternConfig()),
    ]

    trained = {}
    for name, cls, config in strategies_configs:
        try:
            s = cls()
            s.initialize(train_emb, split.train_labels, config)
            trained[name] = s
        except (ValueError, RuntimeError):
            pass

    results: dict[str, dict[str, tuple]] = {"val": {}, "test": {}}
    all_metrics: dict[str, dict[int, dict]] = {"val": {}, "test": {}}

    for split_name, texts, emb in [
        ("val", split.val_texts, None),
        ("test", split.test_texts, None),
    ]:
        from sentence_transformers import SentenceTransformer

        emb_actual = model.encode(texts, show_progress_bar=False)
        dummy_classes = ["unknown"] * len(texts)
        dummy_conf = np.ones(len(texts)) * 0.5

        per_idx_metrics: dict[int, dict] = {i: {} for i in range(len(texts))}

        for name, strategy in trained.items():
            try:
                flags, metrics = strategy.detect(texts, emb_actual, dummy_classes, dummy_conf)
                results[split_name][name] = (flags, metrics)
                for idx, m in metrics.items():
                    per_idx_metrics[idx].update(m)
            except (ValueError, RuntimeError):
                pass

        all_metrics[split_name] = per_idx_metrics

    return results["val"], all_metrics["val"], results["test"], all_metrics["test"]


def _compute_combined_auroc(
    strategy_outputs: dict[str, tuple],
    all_metrics: dict[int, dict],
    true_labels: np.ndarray,
    combine_method: str,
    weights: dict[str, float],
    novelty_threshold: float,
) -> float:
    from ..novelty.config.base import DetectionConfig
    from ..novelty.config.weights import WeightConfig
    from ..novelty.core.signal_combiner import SignalCombiner

    wc = WeightConfig(
        **{k: v for k, v in weights.items() if k in WEIGHT_FIELDS},  # type: ignore[arg-type]
        novelty_threshold=novelty_threshold,
    )
    det_config = DetectionConfig(combine_method=combine_method, weights=wc)
    combiner = SignalCombiner(det_config)

    _, novelty_scores = combiner.combine(strategy_outputs, all_metrics)
    scores = np.array([novelty_scores.get(i, 0.0) for i in range(len(true_labels))])

    try:
        m = compute_ood_metrics(true_labels, scores)
        return m["auroc"]
    except (ValueError, RuntimeError):
        return 0.0


def run_weight_optimization(
    dataset: str = "ag_news",
    model_name: str = DEFAULT_MODEL_NAME,
    n_trials: int = 200,
    max_train: int = 200,
    max_val: int = 200,
    max_test: int = 500,
    output: str | None = None,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    print(f"Loading data: {dataset}")
    split = load_and_split_data(dataset, max_train, max_val, max_test)
    print(f"Train: {len(split.train_texts)}, Val: {len(split.val_texts)}, Test: {len(split.test_texts)}")

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding training data...")
    train_emb = model.encode(split.train_texts, show_progress_bar=False)

    print("Running all strategies...")
    val_outputs, val_all_metrics, test_outputs, test_all_metrics = _run_all_strategies(
        model, split, train_emb,
    )

    val_true = prepare_binary_labels(split.val_labels, "__OOD__")
    test_true = prepare_binary_labels(split.test_labels, "__OOD__")

    print(f"Strategies available: {list(val_outputs.keys())}")
    print(f"Starting Optuna optimization ({n_trials} trials)...")

    strategy_name_to_weight_key = {
        "confidence": "confidence",
        "knn_distance": "knn",
        "setfit_centroid": "setfit_centroid",
        "uncertainty": "uncertainty",
        "self_knowledge": "self_knowledge",
        "mahalanobis": "mahalanobis",
        "lof": "lof",
        "oneclass": "oneclass",
        "pattern": "pattern",
    }

    available_strategies = [n for n in val_outputs if n in strategy_name_to_weight_key]

    def objective(trial: optuna.Trial) -> float:
        weights = {}
        for strat_name in available_strategies:
            wkey = strategy_name_to_weight_key[strat_name]
            weights[wkey] = trial.suggest_float(f"w_{wkey}", 0.0, 1.0)

        for wkey in WEIGHT_FIELDS:
            if wkey not in weights:
                weights[wkey] = 0.0

        novelty_threshold = trial.suggest_float("novelty_threshold", 0.3, 0.8)
        combine_method = trial.suggest_categorical("combine_method", ["weighted", "voting", "meta_learner"])

        return _compute_combined_auroc(
            val_outputs, val_all_metrics, val_true,
            combine_method, weights, novelty_threshold,
        )

    study = optuna.create_study(direction="maximize", study_name="novelty_weights")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_val_auroc = study.best_value

    best_weights = {}
    for strat_name in available_strategies:
        wkey = strategy_name_to_weight_key[strat_name]
        best_weights[wkey] = best[f"w_{wkey}"]
    for wkey in WEIGHT_FIELDS:
        if wkey not in best_weights:
            best_weights[wkey] = 0.0

    best_weights["novelty_threshold"] = best["novelty_threshold"]
    best_method = best["combine_method"]

    test_auroc = _compute_combined_auroc(
        test_outputs, test_all_metrics, test_true,
        best_method, best_weights, best["novelty_threshold"],
    )

    print(f"\n{'=' * 60}")
    print("OPTIMAL WEIGHTS (Bayesian Optimization)")
    print(f"{'=' * 60}")
    print(f"Combine method: {best_method}")
    print(f"Novelty threshold: {best['novelty_threshold']:.4f}")
    print(f"Val AUROC: {best_val_auroc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"\nOptimal weights:")
    for wkey in WEIGHT_FIELDS:
        print(f"  {wkey}: {best_weights[wkey]:.4f}")

    top5 = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]
    print(f"\nTop 5 trials:")
    for i, trial in enumerate(top5):
        print(f"  #{i + 1}: AUROC={trial.value:.4f}, method={trial.params.get('combine_method')}, "
              f"threshold={trial.params.get('novelty_threshold'):.3f}")

    result = {
        "best_val_auroc": best_val_auroc,
        "best_test_auroc": test_auroc,
        "best_method": best_method,
        "best_novelty_threshold": best["novelty_threshold"],
        "best_weights": {k: v for k, v in best_weights.items() if k != "novelty_threshold"},
        "top5_trials": [
            {"value": t.value, "params": t.params}
            for t in top5
        ],
        "n_trials": n_trials,
        "dataset": dataset,
        "model": model_name,
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {output}")

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bayesian optimization of ensemble weights")
    parser.add_argument("--dataset", default="ag_news", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--output", default=None, help="Output JSON path")

    args = parser.parse_args(argv)
    run_weight_optimization(
        dataset=args.dataset,
        model_name=args.model,
        n_trials=args.trials,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        output=args.output,
    )
    return 0
