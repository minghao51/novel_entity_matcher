"""
Configuration for novelty detection.

This module contains all configuration classes for novelty detection,
including the main DetectionConfig and per-strategy configurations.
"""

from .base import DetectionConfig
from .strategies import (
    ClusteringConfig,
    ConfidenceConfig,
    KNNConfig,
    OneClassConfig,
    PatternConfig,
    PrototypicalConfig,
    SelfKnowledgeConfig,
    SetFitConfig,
)
from .weights import WeightConfig

__all__ = [
    "ClusteringConfig",
    "ConfidenceConfig",
    "DetectionConfig",
    "KNNConfig",
    "OneClassConfig",
    "PatternConfig",
    "PrototypicalConfig",
    "SelfKnowledgeConfig",
    "SetFitConfig",
    "WeightConfig",
]
