"""
Report dataclasses for novelty detection.

This module re-exports the main report classes for convenience.
"""

from .results import (
    DetectionReport,
    EvaluationReport,
    NovelSampleReport,
)

__all__ = [
    "DetectionReport",
    "EvaluationReport",
    "NovelSampleReport",
]
