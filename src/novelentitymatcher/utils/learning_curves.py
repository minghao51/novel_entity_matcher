"""Learning curve analysis for overfitting detection."""

from typing import Any


def analyze_overfitting(
    train_acc: float, val_acc: float, test_acc: float
) -> dict[str, Any]:
    """Analyze train/val/test accuracy for overfitting patterns.

    Args:
        train_acc: Training set accuracy (0-1)
        val_acc: Validation set accuracy (0-1)
        test_acc: Test set accuracy (0-1)

    Returns:
        Dict with gap metrics and diagnosis.
    """
    train_val_gap = train_acc - val_acc
    train_test_gap = train_acc - test_acc
    val_test_gap = val_acc - test_acc

    if train_val_gap > 0.3:
        diagnosis = "severe_overfitting"
    elif train_val_gap > 0.15:
        diagnosis = "moderate_overfitting"
    elif train_acc < 0.6 and val_acc < 0.6:
        diagnosis = "underfitting"
    else:
        diagnosis = "healthy"

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_val_gap": train_val_gap,
        "train_test_gap": train_test_gap,
        "val_test_gap": val_test_gap,
        "diagnosis": diagnosis,
    }
