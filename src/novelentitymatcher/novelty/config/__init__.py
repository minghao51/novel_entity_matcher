"""
Configuration for novelty detection.

This module contains all configuration classes for novelty detection,
including the main DetectionConfig and per-strategy configurations.
"""

from .base import DetectionConfig
from .strategies import (
    ClusteringConfig,
    ConfidenceConfig,
    EnergyConfig,
    KNNConfig,
    LOFConfig,
    MahalanobisConfig,
    MixtureGaussianConfig,
    OneClassConfig,
    PatternConfig,
    PrototypicalConfig,
    ReActConfig,
    SelfKnowledgeConfig,
    SetFitCentroidConfig,
    SetFitConfig,
    UncertaintyConfig,
)
from .weights import WeightConfig

__all__ = [
    "ClusteringConfig",
    "ConfidenceConfig",
    "DetectionConfig",
    "EnergyConfig",
    "KNNConfig",
    "LOFConfig",
    "MahalanobisConfig",
    "MixtureGaussianConfig",
    "OneClassConfig",
    "PatternConfig",
    "PrototypicalConfig",
    "ReActConfig",
    "SelfKnowledgeConfig",
    "SetFitCentroidConfig",
    "SetFitConfig",
    "UncertaintyConfig",
    "WeightConfig",
]
