"""
Novelty detection strategies.

This module contains the base strategy protocol and all concrete
strategy implementations for detecting novel entities.

Core Strategies:
- confidence: Confidence threshold
- knn_distance: kNN distance-based
- uncertainty: Margin/entropy uncertainty
- clustering: Clustering-based

Advanced Strategies:
- self_knowledge: Sparse autoencoder
- pattern: Pattern-based
- oneclass: One-Class SVM
- prototypical: Prototypical networks
- setfit: SetFit contrastive learning

Usage:
    from novelentitymatcher.novelty import StrategyRegistry, DetectionConfig

    # List available strategies
    strategies = StrategyRegistry.list_strategies()

    # Use in configuration
    config = DetectionConfig(
        strategies=["confidence", "knn_distance", "clustering"],
    )

Importing Strategies:
    # Import specific strategies directly
    from novelentitymatcher.novelty.strategies.confidence import ConfidenceStrategy
    from novelentitymatcher.novelty.strategies.knn_distance import KNNDistanceStrategy
"""

# Base protocol
from .base import NoveltyStrategy
from .oneclass_impl import OneClassSVMDetector

# Import low-level strategy helpers that are still useful directly.
from .pattern_impl import PatternScorer, score_batch_novelty
from .prototypical_impl import PrototypicalDetector
from .self_knowledge_impl import SelfKnowledgeDetector, SparseAutoencoder
from .setfit_impl import SetFitDetector

__all__ = [
    # Base
    "NoveltyStrategy",
    "OneClassSVMDetector",
    # Low-level strategy helpers
    "PatternScorer",
    "PrototypicalDetector",
    "SelfKnowledgeDetector",
    "SetFitDetector",
    "SparseAutoencoder",
    "score_batch_novelty",
]


def _register_all() -> None:
    """Import all strategy modules to trigger @StrategyRegistry.register decorators.

    Called once from novelty/__init__.py after core modules are fully initialized.
    """
    from . import (
        clustering,  # noqa: F401
        confidence,  # noqa: F401
        knn_distance,  # noqa: F401
        lof,  # noqa: F401
        mahalanobis,  # noqa: F401
        oneclass,  # noqa: F401
        pattern,  # noqa: F401
        prototypical,  # noqa: F401
        self_knowledge,  # noqa: F401
        setfit,  # noqa: F401
        setfit_centroid,  # noqa: F401
        uncertainty,  # noqa: F401
    )
