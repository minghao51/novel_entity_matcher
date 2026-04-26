"""
Utility functions for novelty detection.

This module contains shared utility functions used across
the novelty detection subsystem.
"""

from .scoring import compute_similarity, normalize_score

__all__ = [
    "compute_similarity",
    "normalize_score",
]
