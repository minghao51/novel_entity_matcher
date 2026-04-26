"""
Metric computations for novelty detection evaluation.

Provides functions for computing AUROC, AUPRC, detection rates,
precision, recall, F1, and confusion matrices.
"""

import numpy as np


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        scores: Predicted novelty scores (higher = more novel)
        labels: Ground truth labels (True = novel)

    Returns:
        AUROC score (0-1, 0.5 = random)
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return 0.5

    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def compute_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    Args:
        scores: Predicted novelty scores (higher = more novel)
        labels: Ground truth labels (True = novel)

    Returns:
        AUPRC score (0-1)
    """
    from sklearn.metrics import auc, precision_recall_curve

    if len(np.unique(labels)) < 2:
        return 0.0

    try:
        prec, rec, _ = precision_recall_curve(labels, scores)
        return float(auc(rec, prec))
    except ValueError:
        return 0.0


def compute_detection_rates(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_thresholds: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict[str, float]:
    """
    Compute detection rates at specific false positive rates.

    Args:
        scores: Predicted novelty scores (higher = more novel)
        labels: Ground truth labels (True = novel)
        fpr_thresholds: FPR values to compute detection rates for

    Returns:
        Dict mapping fpr_percentage -> detection_rate
        (e.g., "detection_rate_1" -> 0.95 for 1% FPR)
    """
    results = {}

    for fpr in fpr_thresholds:
        non_novel_scores = scores[~labels]
        if len(non_novel_scores) == 0:
            detection_rate = 1.0 if np.all(labels) else 0.0
        else:
            threshold = np.percentile(non_novel_scores, (1 - fpr) * 100)
            detected = np.sum((scores >= threshold) & labels)
            total_novel = np.sum(labels)
            detection_rate = detected / total_novel if total_novel > 0 else 0.0

        percentage = int(fpr * 100)
        results[f"detection_rate_{percentage}"] = float(detection_rate)

    return results


def compute_precision_recall_f1(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        scores: Predicted novelty scores (higher = more novel)
        labels: Ground truth labels (True = novel)
        threshold: Decision threshold (if None, finds optimal)

    Returns:
        Dict with precision, recall, f1, and threshold
    """
    if threshold is None:
        threshold = find_optimal_threshold(scores, labels)

    predictions = scores >= threshold

    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Find threshold that maximizes F1 score.

    Args:
        scores: Predicted novelty scores (higher = more novel)
        labels: Ground truth labels (True = novel)

    Returns:
        Optimal threshold value
    """
    thresholds: np.ndarray = np.percentile(scores, np.arange(5, 100, 5))
    best_f1 = 0.0
    best_thresh = 0.5

    for thresh in thresholds:
        predictions = scores >= thresh
        tp = np.sum(predictions & labels)
        fp = np.sum(predictions & ~labels)
        fn = np.sum(~predictions & labels)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return float(best_thresh)


def compute_confusion_matrix(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """
    Compute confusion matrix components.

    Args:
        scores: Predicted novelty scores (higher = more novel)
        labels: Ground truth labels (True = novel)
        threshold: Decision threshold

    Returns:
        Dict with tp, tn, fp, fn counts
    """
    predictions = scores >= threshold

    tp = int(np.sum(predictions & labels))
    tn = int(np.sum(~predictions & ~labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def sweep_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Sweep across thresholds and compute metrics at each.

    Args:
        scores: Predicted novelty scores (higher = more novel)
        labels: Ground truth labels (True = novel)
        thresholds: Array of thresholds to sweep (default: 0-100)

    Returns:
        Dict with arrays for thresholds, precision, recall, f1, tp, fp, tn, fn
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    precision = []
    recall = []
    f1 = []
    tp = []
    fp = []
    tn = []
    fn = []

    for thresh in thresholds:
        preds = scores >= thresh

        tp_i = np.sum(preds & labels)
        fp_i = np.sum(preds & ~labels)
        tn_i = np.sum(~preds & ~labels)
        fn_i = np.sum(~preds & labels)

        prec = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
        rec = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
        f1_i = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precision.append(float(prec))
        recall.append(float(rec))
        f1.append(float(f1_i))
        tp.append(int(tp_i))
        fp.append(int(fp_i))
        tn.append(int(tn_i))
        fn.append(int(fn_i))

    return {
        "thresholds": thresholds,
        "precision": np.array(precision),
        "recall": np.array(recall),
        "f1": np.array(f1),
        "tp": np.array(tp),
        "fp": np.array(fp),
        "tn": np.array(tn),
        "fn": np.array(fn),
    }
