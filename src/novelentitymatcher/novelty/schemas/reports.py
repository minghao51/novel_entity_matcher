"""
Report dataclasses for novelty detection.

This module re-exports the main report classes for convenience.
"""

from .results import (
    DetectionReport,
    EvaluationReport,
)

__all__ = [
    "DetectionReport",
    "EvaluationReport",
]
