"""
Clustering functionality for novelty detection.

This module contains clustering algorithms and validation logic
used for detecting novel samples.
"""

from .scalable import ScalableClusterer
from .validation import ClusterValidator

__all__ = [
    "ClusterValidator",
    "ScalableClusterer",
]
