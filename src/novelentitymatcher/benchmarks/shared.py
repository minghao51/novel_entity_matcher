"""Shared utilities for benchmark scripts.

Consolidates duplicated code from:
- benchmark_bert.py / benchmark_bert_models.py (generate_synthetic_data, benchmark_training, benchmark_inference)
- benchmark_full_pipeline.py / benchmark_novelty_strategies.py / benchmark_novelty_full.py (compute_ood_metrics, SplitData, OOD splitting)
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RANDOM_SEED = 42
DEFAULT_OOD_RATIO = 0.2


@dataclass
class SplitData:
    train_texts: list[str]
    train_labels: list[str]
    val_texts: list[str]
    val_labels: list[str]
    test_texts: list[str]
    test_labels: list[str]
    entities: list[dict] = field(default_factory=list)
    known_classes: list[str] = field(default_factory=list)
    ood_classes: list[str] = field(default_factory=list)


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    result: dict[str, float] = {}
    start = time.perf_counter()
    yield result
    result["elapsed"] = time.perf_counter() - start


def compute_ood_metrics(
    true_labels: list[str] | np.ndarray,
    novelty_scores: np.ndarray,
    novel_marker: str = "__NOVEL__",
) -> dict[str, float]:
    if isinstance(true_labels, np.ndarray):
        true_binary = true_labels.astype(int)
    else:
        true_binary = np.array(
            [1 if label == novel_marker else 0 for label in true_labels]
        )

    if len(np.unique(true_binary)) < 2:
        return {
            "auroc": 0.5 if len(np.unique(true_binary)) < 2 else 0.0,
            "auprc": 0.5 if len(np.unique(true_binary)) < 2 else 0.0,
            "dr_1fp": 0.0,
            "dr_5fp": 0.0,
            "dr_10fp": 0.0,
        }

    auroc = roc_auc_score(true_binary, novelty_scores)
    auprc = average_precision_score(true_binary, novelty_scores)

    num_known = int(np.sum(true_binary == 0))
    num_ood = int(np.sum(true_binary == 1))

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
        detection_rates[fp_rate] = detected / num_ood if num_ood > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "dr_1fp": detection_rates[0.01],
        "dr_5fp": detection_rates[0.05],
        "dr_10fp": detection_rates[0.10],
    }


def prepare_binary_labels(
    labels: list[str], novel_marker: str = "__OOD__"
) -> np.ndarray:
    return np.array([1 if label == novel_marker else 0 for label in labels])


def generate_synthetic_data(
    num_entities: int = 10,
    samples_per_entity: int = 50,
) -> tuple[list[dict], list[dict]]:
    entities = [f"ENTITY_{i}" for i in range(num_entities)]

    training_data = []
    for entity in entities:
        for i in range(samples_per_entity):
            training_data.append(
                {"text": f"{entity} text variant {i}", "label": entity}
            )

    test_data = []
    for entity in entities:
        for i in range(samples_per_entity // 10):
            test_data.append({"text": f"{entity} test variant {i}", "label": entity})

    return training_data, test_data


def benchmark_training(
    classifier_class,
    training_data: list[dict],
    labels: list[str],
    num_epochs: int = 3,
    **classifier_kwargs,
) -> dict[str, float]:
    clf = classifier_class(labels=labels, **classifier_kwargs)

    tracemalloc.start()
    start_time = time.perf_counter()

    clf.train(training_data, num_epochs=num_epochs, show_progress=False)

    elapsed = time.perf_counter() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "training_time": elapsed,
        "memory_peak_mb": peak / 1024 / 1024,
    }


def benchmark_inference(
    classifier,
    test_data: list[dict],
) -> dict[str, float]:
    texts = [item["text"] for item in test_data]
    true_labels = [item["label"] for item in test_data]

    start_time = time.perf_counter()
    predictions = classifier.predict(texts)
    elapsed = time.perf_counter() - start_time

    throughput = len(texts) / elapsed
    correct = sum(1 for pred, true in zip(predictions, true_labels, strict=False) if pred == true)
    accuracy = correct / len(true_labels)

    return {
        "inference_time": elapsed,
        "throughput_samples_per_sec": throughput,
        "accuracy": accuracy,
    }
