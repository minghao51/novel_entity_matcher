"""Merged classifier benchmark: BERT vs SetFit comparison and multi-model sweep.

Consolidates:
- benchmark_bert.py (head-to-head BERT vs SetFit)
- benchmark_bert_models.py (multi-model BERT sweep)

Modes:
- ``compare``: BERT vs SetFit head-to-head
- ``sweep-models``: benchmark multiple BERT-family classifiers
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .shared import (
    DEFAULT_MODEL_NAME,
    benchmark_inference,
    benchmark_training,
    generate_synthetic_data,
)

MODEL_ALIASES = {
    "distilbert": "distilbert-base-uncased",
    "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    "roberta-base": "roberta-base",
    "deberta-v3": "microsoft/deberta-v3-base",
    "bert-multilingual": "bert-base-multilingual-cased",
}


def run_compare(
    num_entities: int = 10,
    samples_per_entity: int = 50,
    num_epochs: int = 3,
) -> Dict[str, Dict[str, Any]]:
    from ..core.bert_classifier import BERTClassifier
    from ..core.classifier import SetFitClassifier

    training_data, test_data = generate_synthetic_data(num_entities, samples_per_entity)
    labels = sorted(set(item["label"] for item in training_data))

    print(f"\n{'=' * 60}")
    print("BERT vs SetFit Comparison")
    print(f"  Entities: {num_entities}, Samples/entity: {samples_per_entity}")
    print(f"  Train: {len(training_data)}, Test: {len(test_data)}, Epochs: {num_epochs}")
    print(f"{'=' * 60}\n")

    results: Dict[str, Dict[str, Any]] = {}

    for name, cls, model in [
        ("setfit", SetFitClassifier, DEFAULT_MODEL_NAME),
        ("bert", BERTClassifier, "distilbert-base-uncased"),
    ]:
        print(f"Benchmarking {name}...")
        try:
            train_m = benchmark_training(
                cls, training_data, labels, num_epochs=num_epochs, model_name=model
            )
            clf = cls(labels=labels, model_name=model)
            clf.train(training_data, num_epochs=num_epochs, show_progress=False)
            infer_m = benchmark_inference(clf, test_data)
            results[name] = {**train_m, **infer_m}
            print(f"  Train: {train_m['training_time']:.2f}s, Mem: {train_m['memory_peak_mb']:.2f}MB")
            print(f"  Infer: {infer_m['inference_time']:.2f}s, Thru: {infer_m['throughput_samples_per_sec']:.2f}/s")
            print(f"  Acc: {infer_m['accuracy']:.2%}")
        except (ValueError, RuntimeError) as e:
            print(f"  FAILED: {e}")
            results[name] = None  # type: ignore[assignment]

    if results.get("setfit") and results.get("bert"):
        s, b = results["setfit"], results["bert"]
        print(f"\n{'=' * 60}")
        print(f"{'Metric':<30} {'SetFit':<15} {'BERT':<15} {'Ratio':<10}")
        print(f"{'-' * 60}")
        for label, key in [
            ("Training Time (s)", "training_time"),
            ("Peak Memory (MB)", "memory_peak_mb"),
            ("Inference Time (s)", "inference_time"),
            ("Throughput (samples/s)", "throughput_samples_per_sec"),
            ("Accuracy", "accuracy"),
        ]:
            sv, bv = s[key], b[key]
            ratio = (
                f"{sv / bv:.2f}x"
                if key in ["training_time", "inference_time", "memory_peak_mb"]
                else f"{bv / sv:.2f}x"
            )
            print(f"{label:<30} {sv:<15.2f} {bv:<15.2f} {ratio:<10}")
        print(f"{'=' * 60}\n")

    return results


def run_sweep(
    model_names: List[str] | None = None,
    num_entities: int = 20,
    samples_per_entity: int = 100,
    num_epochs: int = 5,
) -> Dict[str, Dict[str, Any]]:
    from ..core.bert_classifier import BERTClassifier

    if model_names is None:
        model_names = [
            "distilbert-base-uncased",
            "huawei-noah/TinyBERT_General_4L_312D",
            "roberta-base",
        ]

    training_data, test_data = generate_synthetic_data(num_entities, samples_per_entity)
    labels = sorted(set(item["label"] for item in training_data))

    print(f"\n{'=' * 60}")
    print(f"BERT Model Sweep ({len(model_names)} models)")
    print(f"  Entities: {num_entities}, Samples/entity: {samples_per_entity}")
    print(f"{'=' * 60}\n")

    results: Dict[str, Dict[str, Any]] = {}

    for model_name in model_names:
        print(f"Benchmarking {model_name}...")
        try:
            train_m = benchmark_training(
                BERTClassifier,
                training_data,
                labels,
                num_epochs=num_epochs,
                model_name=model_name,
            )
            clf = BERTClassifier(labels=labels, model_name=model_name)
            clf.train(training_data, num_epochs=num_epochs, show_progress=False)
            infer_m = benchmark_inference(clf, test_data)

            results[model_name] = {
                **train_m,
                **infer_m,
                "model_name": model_name,
                "status": "ok",
            }
            print(f"  Train: {train_m['training_time']:.2f}s, Mem: {train_m['memory_peak_mb']:.2f}MB")
            print(f"  Infer: {infer_m['inference_time']:.2f}s, Acc: {infer_m['accuracy']:.2%}")
        except (ValueError, RuntimeError) as e:
            print(f"  FAILED: {e}")
            results[model_name] = {"model_name": model_name, "status": "failed", "error": str(e)}

    successful = {k: v for k, v in results.items() if v.get("status") == "ok"}
    if successful:
        print(f"\n{'=' * 80}")
        print(f"{'Model':<25} {'Train(s)':<10} {'Mem(MB)':<10} {'Infer(s)':<10} {'Thru(/s)':<12} {'Acc':<8}")
        print(f"{'-' * 80}")
        for name, m in sorted(successful.items()):
            print(
                f"{name:<25} {m['training_time']:<10.2f} {m['memory_peak_mb']:<10.2f} "
                f"{m['inference_time']:<10.2f} {m['throughput_samples_per_sec']:<12.2f} {m['accuracy']:<8.2%}"
            )
        print(f"{'=' * 80}\n")

    return results


SETFIT_MODEL_ALIASES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "nomic": "nomic-ai/nomic-embed-text-v1",
}


def run_setfit_sweep(
    model_names: list[str] | None = None,
    num_entities: int = 10,
    samples_per_entity: int = 50,
    num_epochs: int = 3,
) -> dict[str, dict[str, Any]]:
    from ..core.classifier import SetFitClassifier

    if model_names is None:
        model_names = list(SETFIT_MODEL_ALIASES.values())

    training_data, test_data = generate_synthetic_data(num_entities, samples_per_entity)
    labels = sorted(set(item["label"] for item in training_data))

    print(f"\n{'=' * 60}")
    print(f"SetFit Model Sweep ({len(model_names)} models)")
    print(f"  Entities: {num_entities}, Samples/entity: {samples_per_entity}")
    print(f"{'=' * 60}\n")

    results: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        print(f"Benchmarking {model_name}...")
        try:
            train_m = benchmark_training(
                SetFitClassifier, training_data, labels,
                num_epochs=num_epochs, model_name=model_name,
            )
            clf = SetFitClassifier(labels=labels, model_name=model_name)
            clf.train(training_data, num_epochs=num_epochs, show_progress=False)
            infer_m = benchmark_inference(clf, test_data)

            results[model_name] = {
                **train_m, **infer_m,
                "model_name": model_name, "status": "ok",
            }
            print(f"  Train: {train_m['training_time']:.2f}s, Acc: {infer_m['accuracy']:.2%}")
        except (ValueError, RuntimeError) as e:
            print(f"  FAILED: {e}")
            results[model_name] = {"model_name": model_name, "status": "failed", "error": str(e)}

    successful = {k: v for k, v in results.items() if v.get("status") == "ok"}
    if successful:
        print(f"\n{'=' * 80}")
        print(f"{'Model':<40} {'Train(s)':<10} {'Acc':<10}")
        print(f"{'-' * 60}")
        for name, m in sorted(successful.items()):
            print(f"{name:<40} {m['training_time']:<10.2f} {m['accuracy']:<10.2%}")
        print(f"{'=' * 80}\n")

    return results


def run_scale_test(
    sample_counts: list[int] | None = None,
    num_entities: int = 10,
    num_epochs: int = 3,
) -> dict[str, dict[str, Any]]:
    from ..core.classifier import SetFitClassifier

    if sample_counts is None:
        sample_counts = [50, 100, 200, 500, 1000]

    results: dict[str, dict[str, Any]] = {}

    print(f"\n{'=' * 60}")
    print(f"Scaling Benchmark (SetFit, {num_entities} entities)")
    print(f"  Sample counts: {sample_counts}")
    print(f"{'=' * 60}\n")

    for n_samples in sample_counts:
        samples_per_entity = max(1, n_samples // num_entities)
        training_data, test_data = generate_synthetic_data(num_entities, samples_per_entity)
        labels = sorted(set(item["label"] for item in training_data))

        try:
            train_m = benchmark_training(
                SetFitClassifier, training_data, labels,
                num_epochs=num_epochs, model_name=DEFAULT_MODEL_NAME,
            )
            clf = SetFitClassifier(labels=labels, model_name=DEFAULT_MODEL_NAME)
            clf.train(training_data, num_epochs=num_epochs, show_progress=False)
            infer_m = benchmark_inference(clf, test_data)

            results[str(n_samples)] = {
                "n_train": len(training_data),
                "samples_per_entity": samples_per_entity,
                **train_m, **infer_m,
                "status": "ok",
            }
            print(f"  n={n_samples}: Train={train_m['training_time']:.2f}s, Acc={infer_m['accuracy']:.2%}")
        except (ValueError, RuntimeError) as e:
            print(f"  n={n_samples}: FAILED - {e}")
            results[str(n_samples)] = {"n_train": n_samples, "status": "failed", "error": str(e)}

    return results


def run_mode_comparison(
    num_entities: int = 10,
    samples_per_entity: int = 50,
    num_epochs: int = 3,
) -> dict[str, dict[str, Any]]:
    from ..core.matcher import Matcher

    training_data, test_data = generate_synthetic_data(num_entities, samples_per_entity)
    entities = sorted(set(item["label"] for item in training_data))

    print(f"\n{'=' * 60}")
    print(f"Mode Comparison ({len(entities)} entities)")
    print(f"{'=' * 60}\n")

    results: dict[str, dict[str, Any]] = {}

    for mode in ["zero-shot", "head-only", "full", "bert"]:
        print(f"Benchmarking mode: {mode}...")
        try:
            matcher = Matcher(entities=entities, mode=mode, model=DEFAULT_MODEL_NAME)

            if mode != "zero-shot":
                matcher.fit(training_data, num_epochs=num_epochs)

            correct = 0
            start = time.perf_counter()
            for item in test_data:
                result = matcher.match(item["text"])
                if result and result.get("entity") == item["label"]:
                    correct += 1
            elapsed = time.perf_counter() - start

            accuracy = correct / len(test_data) if test_data else 0.0
            results[mode] = {
                "mode": mode,
                "accuracy": accuracy,
                "inference_time_s": elapsed,
                "qps": len(test_data) / elapsed if elapsed > 0 else 0,
                "status": "ok",
            }
            print(f"  Acc: {accuracy:.2%}, Time: {elapsed:.2f}s, QPS: {len(test_data) / elapsed:.0f}")
        except (ValueError, RuntimeError) as e:
            print(f"  FAILED: {e}")
            results[mode] = {"mode": mode, "status": "failed", "error": str(e)}

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classifier benchmarks")
    parser.add_argument(
        "--mode",
        choices=["compare", "sweep-models", "sweep-setfit", "scale-test", "sweep-modes"],
        default="compare",
        help="'compare' BERT vs SetFit, 'sweep-models' BERT sweep, "
             "'sweep-setfit' SetFit model sweep, 'scale-test' sample scaling, "
             "'sweep-modes' mode comparison",
    )
    parser.add_argument("--num-entities", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["distilbert", "tinybert", "roberta-base"],
        help="Model names or aliases",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args(argv)

    if args.mode == "compare":
        results = run_compare(
            num_entities=args.num_entities,
            samples_per_entity=args.num_samples,
            num_epochs=args.num_epochs,
        )
    elif args.mode == "sweep-models":
        models = [MODEL_ALIASES.get(m, m) for m in args.models]
        results = run_sweep(
            model_names=models,
            num_entities=args.num_entities,
            samples_per_entity=args.num_samples,
            num_epochs=args.num_epochs,
        )
    elif args.mode == "sweep-setfit":
        models = [SETFIT_MODEL_ALIASES.get(m, m) for m in args.models]
        results = run_setfit_sweep(
            model_names=models,
            num_entities=args.num_entities,
            samples_per_entity=args.num_samples,
            num_epochs=args.num_epochs,
        )
    elif args.mode == "scale-test":
        results = run_scale_test(
            num_entities=args.num_entities,
            num_epochs=args.num_epochs,
        )
    elif args.mode == "sweep-modes":
        results = run_mode_comparison(
            num_entities=args.num_entities,
            samples_per_entity=args.num_samples,
            num_epochs=args.num_epochs,
        )
    else:
        results = {}

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results saved to {output_path}")

    return 0
