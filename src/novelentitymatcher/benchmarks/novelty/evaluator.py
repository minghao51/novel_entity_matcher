"""Novelty detection benchmark evaluator."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from ..base import BaseEvaluator, EvaluationResult


@dataclass
class NoveltySample:
    text: str
    label: int
    is_known: bool = True


class NoveltyEvaluator(BaseEvaluator[tuple[pd.DataFrame, pd.DataFrame]]):
    def __init__(self):
        super().__init__("Novelty Detection")

    def get_default_metrics(self) -> list[str]:
        return [
            "auroc",
            "auprc",
            "detection_rate_at_1fp",
            "detection_rate_at_5fp",
            "detection_rate_at_10fp",
        ]

    def create_ood_split(
        self,
        data: pd.DataFrame,
        label_col: str = "label",
        ood_ratio: float = 0.2,
        random_seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        random.seed(random_seed)
        unique_labels = sorted(data[label_col].unique())
        num_ood = max(1, int(len(unique_labels) * ood_ratio))
        ood_labels = set(random.sample(unique_labels, num_ood))
        known_labels = set(unique_labels) - ood_labels

        known_data = data[data[label_col].isin(known_labels)].copy()
        ood_data = data[data[label_col].isin(ood_labels)].copy()
        known_data["is_ood"] = 0
        ood_data["is_ood"] = 1

        return known_data, ood_data

    def evaluate(
        self,
        data: tuple[pd.DataFrame, pd.DataFrame],
        detector_fn: Callable[[str], tuple[bool, float]] | None = None,
        novelty_score_fn: Callable[[str], float] | None = None,
        text_col: str = "text",
        **kwargs,
    ) -> EvaluationResult:
        known_data, ood_data = data

        if len(known_data) == 0 or len(ood_data) == 0:
            return EvaluationResult(
                metrics=dict.fromkeys(self.get_default_metrics(), 0.0),
                details={"error": "Empty known or OOD dataset"},
            )

        if novelty_score_fn is None and detector_fn is None:
            raise ValueError("Either novelty_score_fn or detector_fn is required")

        novelty_scores = []
        true_labels = []

        for _, row in known_data.iterrows():
            if novelty_score_fn:
                score = novelty_score_fn(str(row[text_col]))
            else:
                assert detector_fn is not None
                is_novel, score = detector_fn(str(row[text_col]))
                score = 1.0 - score if is_novel else score
            novelty_scores.append(score)
            true_labels.append(0)

        for _, row in ood_data.iterrows():
            if novelty_score_fn:
                score = novelty_score_fn(str(row[text_col]))
            else:
                assert detector_fn is not None
                is_novel, score = detector_fn(str(row[text_col]))
                score = 1.0 - score if is_novel else score
            novelty_scores.append(score)
            true_labels.append(1)

        novelty_scores_arr = np.array(novelty_scores)
        true_labels_arr = np.array(true_labels)

        auroc = -1.0
        try:
            auroc = roc_auc_score(true_labels_arr, novelty_scores_arr)
        except ValueError:
            pass

        auprc = -1.0
        try:
            auprc = average_precision_score(true_labels_arr, novelty_scores_arr)
        except ValueError:
            pass

        detection_rates = self._compute_detection_rates(
            true_labels_arr, novelty_scores_arr
        )

        all_data = pd.concat([known_data, ood_data])
        per_sample_results = pd.DataFrame(
            {
                "text": all_data[text_col],
                "true_label": all_data["is_ood"],
                "novelty_score": novelty_scores,
            }
        )

        return EvaluationResult(
            metrics={
                "auroc": auroc,
                "auprc": auprc,
                "detection_rate_at_1fp": detection_rates[0.01],
                "detection_rate_at_5fp": detection_rates[0.05],
                "detection_rate_at_10fp": detection_rates[0.10],
            },
            details={
                "num_known": len(known_data),
                "num_ood": len(ood_data),
                "ood_ratio": len(ood_data) / (len(known_data) + len(ood_data)),
            },
            dataframe=per_sample_results,
        )

    def _compute_detection_rates(
        self,
        true_labels: np.ndarray,
        novelty_scores: np.ndarray,
        fp_rates: list[float] | None = None,
    ) -> dict[float, float]:
        if fp_rates is None:
            fp_rates = [0.01, 0.05, 0.10]

        results = {}
        num_known = np.sum(true_labels == 0)
        num_ood = np.sum(true_labels == 1)

        if num_ood == 0:
            return dict.fromkeys(fp_rates, 0.0)

        sorted_indices = np.argsort(novelty_scores)[::-1]
        sorted_labels = true_labels[sorted_indices]

        for fp_rate in fp_rates:
            max_false_positives = int(fp_rate * num_known)
            fp_count = 0
            detected_ood = 0

            for label in sorted_labels:
                if label == 0:
                    fp_count += 1
                    if fp_count > max_false_positives:
                        break
                else:
                    detected_ood += 1

            detection_rate = detected_ood / num_ood if num_ood > 0 else 0.0
            results[fp_rate] = detection_rate

        return results


def sweep_knn_params(
    detector_fn: Callable[[str], tuple[bool, float]],
    known_data: pd.DataFrame,
    ood_data: pd.DataFrame,
    k_values: list[int] | None = None,
    distance_thresholds: list[float] | None = None,
    text_col: str = "text",
) -> dict[tuple[int, float], dict[str, float]]:
    if k_values is None:
        k_values = [3, 5, 10, 20]
    if distance_thresholds is None:
        distance_thresholds = [0.4, 0.5, 0.6, 0.7]

    evaluator = NoveltyEvaluator()
    combined_data = (known_data, ood_data)

    results = {}
    for k in k_values:
        for dist_thresh in distance_thresholds:
            result = evaluator.evaluate(
                combined_data,
                novelty_score_fn=lambda x: 0.5,
                text_col=text_col,
            )
            results[(k, dist_thresh)] = result.metrics

    return results


def find_optimal_knn_params(
    detector_fn: Callable[[str], tuple[bool, float]],
    known_data: pd.DataFrame,
    ood_data: pd.DataFrame,
    k_values: list[int] | None = None,
    distance_thresholds: list[float] | None = None,
    metric: str = "auroc",
    text_col: str = "text",
) -> tuple[tuple[int, float], dict[str, float]]:
    sweep_results = sweep_knn_params(
        detector_fn,
        known_data,
        ood_data,
        k_values,
        distance_thresholds,
        text_col,
    )
    best_params = max(sweep_results, key=lambda p: sweep_results[p][metric])
    return best_params, sweep_results[best_params]
