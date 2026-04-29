"""Classification benchmark evaluator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ..base import BaseEvaluator, EvaluationResult


@dataclass
class ClassificationSample:
    text: str
    label: int


class ClassificationEvaluator(BaseEvaluator[pd.DataFrame]):
    def __init__(self):
        super().__init__("Classification")

    def get_default_metrics(self) -> list[str]:
        return [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "per_class_f1",
        ]

    def evaluate(
        self,
        data: pd.DataFrame,
        matcher_fn: Callable[[str], tuple[str, float]] | None = None,
        threshold: float = 0.5,
        text_col: str = "text",
        label_col: str = "label",
        classes: list[str] | None = None,
        **kwargs,
    ) -> EvaluationResult:
        if len(data) == 0:
            return EvaluationResult(
                metrics=dict.fromkeys(self.get_default_metrics(), 0.0),
                details={"error": "Empty dataset"},
            )

        if matcher_fn is None:
            raise ValueError("matcher_fn is required for classification evaluation")

        predictions = []
        confidences = []
        true_labels_raw = data[label_col].tolist()

        for _, row in data.iterrows():
            result = matcher_fn(str(row[text_col]))
            if result is None:
                predicted_label = None
                confidence = 0.0
            elif isinstance(result, tuple):
                predicted_label, confidence = result
            else:
                predicted_label = None
                confidence = 0.0
            predictions.append(predicted_label)
            confidences.append(confidence)

        true_labels = [str(label) for label in true_labels_raw]
        pred_labels = [
            "__no_match__" if label is None else str(label) for label in predictions
        ]

        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(
            true_labels, pred_labels, average="weighted", zero_division=0
        )

        unique_labels = sorted(set(true_labels + pred_labels))
        per_class_f1 = f1_score(
            true_labels,
            pred_labels,
            labels=unique_labels,
            average=None,
            zero_division=0,
        )
        per_class_f1_dict = {
            str(label): float(f1)
            for label, f1 in zip(unique_labels, per_class_f1, strict=False)
        }

        class_names = (
            classes
            if classes and len(classes) == len(unique_labels)
            else [str(label) for label in unique_labels]
        )
        report = classification_report(
            true_labels,
            pred_labels,
            labels=unique_labels,
            target_names=class_names[: len(unique_labels)],
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

        per_sample_results = pd.DataFrame(
            {
                "text": data[text_col],
                "true_label": true_labels,
                "pred_label": pred_labels,
                "confidence": confidences,
            }
        )

        return EvaluationResult(
            metrics={
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            },
            details={
                "num_samples": len(data),
                "num_classes": len(unique_labels),
                "classes": class_names,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "per_class_f1": per_class_f1_dict,
            },
            dataframe=per_sample_results,
        )


def sweep_num_classes(
    matcher_fn: Callable[[str], tuple[str, float]],
    dataset: pd.DataFrame,
    label_col: str = "label",
    text_col: str = "text",
    max_classes: int | None = None,
) -> dict[int, dict[str, float]]:
    unique_classes = sorted(dataset[label_col].unique())
    if max_classes:
        unique_classes = unique_classes[:max_classes]

    filtered = dataset[dataset[label_col].isin(unique_classes)]

    evaluator = ClassificationEvaluator()
    result = evaluator.evaluate(
        filtered,
        matcher_fn=matcher_fn,
        label_col=label_col,
        text_col=text_col,
    )

    return {len(unique_classes): result.metrics}


def evaluate_by_class_count(
    matcher_fn: Callable[[str], tuple[str, float]],
    dataset: pd.DataFrame,
    label_col: str = "label",
    text_col: str = "text",
    class_counts: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    if class_counts is None:
        class_counts = [2, 4, 8, 10, 16, 28]

    unique_classes = sorted(dataset[label_col].unique())
    results = {}

    for count in class_counts:
        if count > len(unique_classes):
            continue
        subset_classes = unique_classes[:count]
        filtered = dataset[dataset[label_col].isin(subset_classes)]
        evaluator = ClassificationEvaluator()
        result = evaluator.evaluate(
            filtered,
            matcher_fn=matcher_fn,
            label_col=label_col,
            text_col=text_col,
        )
        results[count] = result.metrics

    return results
