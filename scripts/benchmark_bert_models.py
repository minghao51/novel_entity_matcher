#!/usr/bin/env python
"""Enhanced benchmark script for multiple BERT models.

This script benchmarks multiple BERT-family classifiers on various metrics:
- Training time
- Inference speed
- Accuracy
- Memory usage

Usage:
    python scripts/benchmark_bert_models.py [--models MODEL...] [--output PATH]
"""

import argparse
import json
import time
import tracemalloc
from typing import Dict, List, Tuple
from pathlib import Path


def generate_synthetic_data(
    num_entities: int = 10,
    samples_per_entity: int = 50,
) -> Tuple[List[dict], List[dict]]:
    """Generate synthetic training and test data.

    Args:
        num_entities: Number of unique entities (labels)
        samples_per_entity: Number of training samples per entity

    Returns:
        Tuple of (training_data, test_data)
    """
    entities = [f"ENTITY_{i}" for i in range(num_entities)]

    # Generate training data
    training_data = []
    for entity in entities:
        for i in range(samples_per_entity):
            training_data.append(
                {
                    "text": f"{entity} text variant {i}",
                    "label": entity,
                }
            )

    # Generate test data (10% of training size)
    test_data = []
    for entity in entities:
        for i in range(samples_per_entity // 10):
            test_data.append(
                {
                    "text": f"{entity} test variant {i}",
                    "label": entity,
                }
            )

    return training_data, test_data


def benchmark_training(
    classifier_class,
    training_data: List[dict],
    labels: List[str],
    num_epochs: int = 3,
    **classifier_kwargs,
) -> Dict[str, float]:
    """Benchmark classifier training.

    Args:
        classifier_class: BERTClassifier
        training_data: Training examples
        labels: List of class labels
        num_epochs: Number of training epochs
        **classifier_kwargs: Additional arguments for classifier initialization

    Returns:
        Dict with training_time and memory_usage metrics
    """
    # Initialize classifier
    clf = classifier_class(labels=labels, **classifier_kwargs)

    # Measure memory and time
    tracemalloc.start()
    start_time = time.time()

    clf.train(training_data, num_epochs=num_epochs, show_progress=False)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "training_time": end_time - start_time,
        "memory_peak_mb": peak / 1024 / 1024,
    }


def benchmark_inference(
    classifier,
    test_data: List[dict],
) -> Dict[str, float]:
    """Benchmark classifier inference.

    Args:
        classifier: Trained classifier instance
        test_data: Test examples

    Returns:
        Dict with inference_time, throughput, and accuracy metrics
    """
    texts = [item["text"] for item in test_data]
    true_labels = [item["label"] for item in test_data]

    # Measure inference time
    start_time = time.time()
    predictions = classifier.predict(texts)
    end_time = time.time()

    inference_time = end_time - start_time
    throughput = len(texts) / inference_time

    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    accuracy = correct / len(true_labels)

    return {
        "inference_time": inference_time,
        "throughput_samples_per_sec": throughput,
        "accuracy": accuracy,
    }


def run_benchmark(
    model_names: List[str],
    num_entities: int = 20,
    samples_per_entity: int = 100,
    num_epochs: int = 5,
) -> Dict[str, Dict]:
    """Run complete benchmark for multiple BERT models.

    Args:
        model_names: List of BERT model names to benchmark
        num_entities: Number of unique entities
        samples_per_entity: Training samples per entity
        num_epochs: Number of training epochs

    Returns:
        Dict with benchmark results for each model
    """
    from semanticmatcher.core.bert_classifier import BERTClassifier

    # Generate data
    training_data, test_data = generate_synthetic_data(num_entities, samples_per_entity)
    labels = list(set(item["label"] for item in training_data))

    print(f"\n{'=' * 60}")
    print("BERT Model Benchmark Configuration:")
    print(f"  Entities: {num_entities}")
    print(f"  Samples per entity: {samples_per_entity}")
    print(f"  Total training samples: {len(training_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Models to benchmark: {len(model_names)}")
    print(f"{'=' * 60}\n")

    results = {}

    for model_name in model_names:
        print(f"Benchmarking {model_name}...")
        try:
            # Benchmark training
            train_metrics = benchmark_training(
                BERTClassifier,
                training_data,
                labels,
                num_epochs=num_epochs,
                model_name=model_name,
            )

            # Train fresh classifier for inference benchmark
            clf = BERTClassifier(
                labels=labels,
                model_name=model_name,
            )
            clf.train(training_data, num_epochs=num_epochs, show_progress=False)
            infer_metrics = benchmark_inference(clf, test_data)

            results[model_name] = {
                **train_metrics,
                **infer_metrics,
                "model_name": model_name,
                "status": "ok",
            }

            print(f"  Training time: {train_metrics['training_time']:.2f}s")
            print(f"  Peak memory: {train_metrics['memory_peak_mb']:.2f} MB")
            print(f"  Inference time: {infer_metrics['inference_time']:.2f}s")
            print(f"  Throughput: {infer_metrics['throughput_samples_per_sec']:.2f} samples/s")
            print(f"  Accuracy: {infer_metrics['accuracy']:.2%}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results[model_name] = {
                "model_name": model_name,
                "status": "failed",
                "error": str(e),
            }

        print()

    return results


def print_comparison(results: Dict[str, Dict]):
    """Print comparison table of results.

    Args:
        results: Dict with benchmark results for each model
    """
    # Filter out failed benchmarks
    successful = {k: v for k, v in results.items() if v.get("status") == "ok"}

    if not successful:
        print("\nNo successful benchmarks to compare.")
        return

    print(f"\n{'=' * 80}")
    print("BERT Model Comparison Table:")
    print(f"{'=' * 80}")
    print(f"{'Model':<25} {'Train(s)':<10} {'Mem(MB)':<10} {'Infer(s)':<10} {'Thru(/s)':<12} {'Acc':<8}")
    print(f"{'-' * 80}")

    for model_name, metrics in sorted(successful.items()):
        print(
            f"{model_name:<25} "
            f"{metrics['training_time']:<10.2f} "
            f"{metrics['memory_peak_mb']:<10.2f} "
            f"{metrics['inference_time']:<10.2f} "
            f"{metrics['throughput_samples_per_sec']:<12.2f} "
            f"{metrics['accuracy']:<8.2%}"
        )

    print(f"{'=' * 80}\n")

    # Find best model for each metric
    print("Best Models by Metric:")
    best_train = min(successful.items(), key=lambda x: x[1]["training_time"])
    print(f"  Fastest Training: {best_train[0]} ({best_train[1]['training_time']:.2f}s)")

    best_mem = min(successful.items(), key=lambda x: x[1]["memory_peak_mb"])
    print(f"  Lowest Memory: {best_mem[0]} ({best_mem[1]['memory_peak_mb']:.2f} MB)")

    best_infer = min(successful.items(), key=lambda x: x[1]["inference_time"])
    print(f"  Fastest Inference: {best_infer[0]} ({best_infer[1]['inference_time']:.2f}s)")

    best_thru = max(successful.items(), key=lambda x: x[1]["throughput_samples_per_sec"])
    print(f"  Highest Throughput: {best_thru[0]} ({best_thru[1]['throughput_samples_per_sec']:.2f} samples/s)")

    best_acc = max(successful.items(), key=lambda x: x[1]["accuracy"])
    print(f"  Highest Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']:.2%})")
    print()


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark multiple BERT models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["distilbert-base-uncased", "huawei-noah/TinyBERT_General_4L_312D", "roberta-base"],
        help="BERT models to benchmark (default: distilbert tinybert roberta-base)",
    )
    parser.add_argument(
        "--num-entities",
        type=int,
        default=20,
        help="Number of unique entities (default: 20)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Training samples per entity (default: 100)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: print only)",
    )

    args = parser.parse_args()

    # Map model aliases to full names
    model_mapping = {
        "distilbert": "distilbert-base-uncased",
        "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
        "roberta-base": "roberta-base",
        "deberta-v3": "microsoft/deberta-v3-base",
        "bert-multilingual": "bert-base-multilingual-cased",
    }

    # Resolve model names
    models = [model_mapping.get(m, m) for m in args.models]

    # Run benchmark
    results = run_benchmark(
        model_names=models,
        num_entities=args.num_entities,
        samples_per_entity=args.num_samples,
        num_epochs=args.num_epochs,
    )

    # Print comparison
    print_comparison(results)

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}\n")


if __name__ == "__main__":
    main()
