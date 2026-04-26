"""Entity resolution benchmark evaluator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from ..base import BaseEvaluator, EvaluationResult


@dataclass
class MatchPair:
    left: str
    right: str
    label: int


class EntityResolutionEvaluator(BaseEvaluator[pd.DataFrame]):
    def __init__(self):
        super().__init__("Entity Resolution")

    def get_default_metrics(self) -> list[str]:
        return [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc",
        ]

    def evaluate(
        self,
        data: pd.DataFrame,
        matcher_fn: Callable[[str, str], float] | None = None,
        match_threshold: float = 0.5,
        left_col: str = "left",
        right_col: str = "right",
        label_col: str = "label",
        **kwargs,
    ) -> EvaluationResult:
        if len(data) == 0:
            return EvaluationResult(
                metrics=dict.fromkeys(self.get_default_metrics(), 0.0),
                details={"error": "Empty dataset"},
            )

        if matcher_fn is None:
            raise ValueError("matcher_fn is required for entity resolution evaluation")

        predictions = []
        scores = []

        for _, row in data.iterrows():
            score = matcher_fn(str(row[left_col]), str(row[right_col]))
            scores.append(score)
            predictions.append(1 if score >= match_threshold else 0)

        labels = data[label_col].tolist()
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        pr_curve = precision_recall_curve(labels, scores)
        pr_auc = -1.0
        if len(pr_curve[0]) > 1:
            pr_auc = 0.5 * sum(
                (pr_curve[0][i] + pr_curve[0][i + 1])
                * (pr_curve[1][i] - pr_curve[1][i + 1])
                for i in range(len(pr_curve[0]) - 1)
            )

        per_pair_results = pd.DataFrame(
            {
                "left": data[left_col],
                "right": data[right_col],
                "true_label": labels,
                "pred_label": predictions,
                "score": scores,
            }
        )

        return EvaluationResult(
            metrics={
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pr_auc": pr_auc,
            },
            details={
                "num_pairs": len(data),
                "num_matches": sum(labels),
                "num_non_matches": len(labels) - sum(labels),
                "threshold": match_threshold,
            },
            dataframe=per_pair_results,
        )


def sweep_threshold(
    matcher_fn: Callable[[str, str], float],
    pairs: list[tuple[str, str, int]],
    thresholds: list[float] | None = None,
    metric: str = "f1",
) -> dict[float, dict[str, float]]:
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    evaluator = EntityResolutionEvaluator()
    df = pd.DataFrame(
        [{"left": left, "right": right, "label": label} for left, right, label in pairs]
    )

    results = {}
    for thresh in thresholds:
        result = evaluator.evaluate(
            df,
            matcher_fn=matcher_fn,
            match_threshold=thresh,
        )
        results[thresh] = result.metrics

    return results


def find_optimal_threshold(
    matcher_fn: Callable[[str, str], float],
    pairs: list[tuple[str, str, int]],
    thresholds: list[float] | None = None,
    metric: str = "f1",
) -> tuple[float, dict[str, float]]:
    sweep_results = sweep_threshold(matcher_fn, pairs, thresholds, metric)
    best_thresh = max(sweep_results, key=lambda t: sweep_results[t][metric])
    return best_thresh, sweep_results[best_thresh]
