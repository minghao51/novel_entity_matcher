"""
Unified evaluation for novelty detection.

This module consolidates benchmark and research evaluation into a single
system that supports both use cases.
"""

from .evaluator import NoveltyEvaluator
from .metrics import (
    compute_auprc,
    compute_auroc,
    compute_detection_rates,
    compute_precision_recall_f1,
)
from .splitters import GradualNoveltySplitter, OODSplitter

__all__ = [
    "GradualNoveltySplitter",
    "NoveltyEvaluator",
    "OODSplitter",
    "compute_auprc",
    "compute_auroc",
    "compute_detection_rates",
    "compute_precision_recall_f1",
]
