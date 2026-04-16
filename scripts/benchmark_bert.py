#!/usr/bin/env python
"""Benchmark script comparing BERT vs SetFit classifiers.

This script benchmarks the performance differences between BERTClassifier
and SetFitClassifier on various metrics:
- Training time
- Inference speed
- Accuracy
- Memory usage

Usage:
    python scripts/benchmark_bert.py [--num-samples N] [--num-epochs E]
"""

import argparse
import time
import tracemalloc
from typing import Dict, List, Tuple


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
        classifier_class: Either BERTClassifier or SetFitClassifier
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
    num_entities: int = 10,
    samples_per_entity: int = 50,
    num_epochs: int = 3,
) -> Dict[str, Dict]:
    """Run complete benchmark comparing BERT vs SetFit.

    Args:
        num_entities: Number of unique entities
        samples_per_entity: Training samples per entity
        num_epochs: Number of training epochs

    Returns:
        Dict with benchmark results for both classifiers
    """
    from novelentitymatcher.core.classifier import SetFitClassifier
    from novelentitymatcher.core.bert_classifier import BERTClassifier

    # Generate data
    training_data, test_data = generate_synthetic_data(num_entities, samples_per_entity)
    labels = list(set(item["label"] for item in training_data))

    print(f"\n{'=' * 60}")
    print("Benchmark Configuration:")
    print(f"  Entities: {num_entities}")
    print(f"  Samples per entity: {samples_per_entity}")
    print(f"  Total training samples: {len(training_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Epochs: {num_epochs}")
    print(f"{'=' * 60}\n")

    results = {}

    # Benchmark SetFitClassifier
    print("Benchmarking SetFitClassifier...")
    try:
        setfit_train = benchmark_training(
            SetFitClassifier,
            training_data,
            labels,
            num_epochs=num_epochs,
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster for benchmarking
        )

        # Train fresh classifier for inference benchmark
        setfit_clf = SetFitClassifier(
            labels=labels,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        setfit_clf.train(training_data, num_epochs=num_epochs, show_progress=False)
        setfit_infer = benchmark_inference(setfit_clf, test_data)

        results["setfit"] = {**setfit_train, **setfit_infer}
        print(f"  Training time: {setfit_train['training_time']:.2f}s")
        print(f"  Peak memory: {setfit_train['memory_peak_mb']:.2f} MB")
        print(f"  Inference time: {setfit_infer['inference_time']:.2f}s")
        print(
            f"  Throughput: {setfit_infer['throughput_samples_per_sec']:.2f} samples/s"
        )
        print(f"  Accuracy: {setfit_infer['accuracy']:.2%}")
    except (ValueError, RuntimeError) as e:
        print(f"  SetFit benchmark failed: {e}")
        results["setfit"] = None

    print()

    # Benchmark BERTClassifier
    print("Benchmarking BERTClassifier...")
    try:
        bert_train = benchmark_training(
            BERTClassifier,
            training_data,
            labels,
            num_epochs=num_epochs,
            model_name="distilbert-base-uncased",
        )

        # Train fresh classifier for inference benchmark
        bert_clf = BERTClassifier(
            labels=labels,
            model_name="distilbert-base-uncased",
        )
        bert_clf.train(training_data, num_epochs=num_epochs, show_progress=False)
        bert_infer = benchmark_inference(bert_clf, test_data)

        results["bert"] = {**bert_train, **bert_infer}
        print(f"  Training time: {bert_train['training_time']:.2f}s")
        print(f"  Peak memory: {bert_train['memory_peak_mb']:.2f} MB")
        print(f"  Inference time: {bert_infer['inference_time']:.2f}s")
        print(f"  Throughput: {bert_infer['throughput_samples_per_sec']:.2f} samples/s")
        print(f"  Accuracy: {bert_infer['accuracy']:.2%}")
    except (ValueError, RuntimeError) as e:
        print(f"  BERT benchmark failed: {e}")
        results["bert"] = None

    return results


def print_comparison(results: Dict[str, Dict]):
    """Print comparison table of results.

    Args:
        results: Dict with benchmark results
    """
    if results.get("setfit") is None or results.get("bert") is None:
        print("\nCannot print comparison - one or both benchmarks failed.")
        return

    setfit = results["setfit"]
    bert = results["bert"]

    print(f"\n{'=' * 60}")
    print("Comparison Table:")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30} {'SetFit':<15} {'BERT':<15} {'Ratio':<10}")
    print(f"{'-' * 60}")

    metrics = [
        ("Training Time (s)", "training_time"),
        ("Peak Memory (MB)", "memory_peak_mb"),
        ("Inference Time (s)", "inference_time"),
        ("Throughput (samples/s)", "throughput_samples_per_sec"),
        ("Accuracy", "accuracy"),
    ]

    for label, key in metrics:
        setfit_val = setfit[key]
        bert_val = bert[key]

        if key in ["training_time", "inference_time", "memory_peak_mb"]:
            # Lower is better
            ratio = f"{setfit_val / bert_val:.2f}x"
        else:
            # Higher is better
            ratio = f"{bert_val / setfit_val:.2f}x"

        print(f"{label:<30} {setfit_val:<15.2f} {bert_val:<15.2f} {ratio:<10}")

    print(f"{'=' * 60}\n")

    # Print summary
    print("Summary:")
    print(
        f"  - BERT is {setfit['training_time'] / bert['training_time']:.2f}x faster to train"
    )
    print(
        f"  - BERT uses {bert['memory_peak_mb'] / setfit['memory_peak_mb']:.2f}x more memory"
    )
    print(
        f"  - BERT is {setfit['throughput_samples_per_sec'] / bert['throughput_samples_per_sec']:.2f}x slower at inference"
    )
    print(
        f"  - BERT accuracy is {(bert['accuracy'] - setfit['accuracy']) * 100:+.2f} percentage points different"
    )
    print()


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark BERT vs SetFit classifiers")
    parser.add_argument(
        "--num-entities",
        type=int,
        default=10,
        help="Number of unique entities (default: 10)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Training samples per entity (default: 50)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        num_entities=args.num_entities,
        samples_per_entity=args.num_samples,
        num_epochs=args.num_epochs,
    )

    # Print comparison
    print_comparison(results)


if __name__ == "__main__":
    main()
