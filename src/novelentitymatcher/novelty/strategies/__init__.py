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

# Import low-level strategy helpers that are still useful directly.
from .pattern_impl import PatternScorer, score_batch_novelty
from .oneclass_impl import OneClassSVMDetector
from .prototypical_impl import PrototypicalDetector
from .setfit_impl import SetFitDetector
from .self_knowledge_impl import SelfKnowledgeDetector, SparseAutoencoder

__all__ = [
    # Base
    "NoveltyStrategy",
    # Low-level strategy helpers
    "PatternScorer",
    "score_batch_novelty",
    "OneClassSVMDetector",
    "PrototypicalDetector",
    "SetFitDetector",
    "SelfKnowledgeDetector",
    "SparseAutoencoder",
]


def _register_all() -> None:
    """Import all strategy modules to trigger @StrategyRegistry.register decorators.

    Called once from novelty/__init__.py after core modules are fully initialized.
    """
    from . import confidence  # noqa: F401
    from . import knn_distance  # noqa: F401
    from . import uncertainty  # noqa: F401
    from . import clustering  # noqa: F401
    from . import pattern  # noqa: F401
    from . import oneclass  # noqa: F401
    from . import prototypical  # noqa: F401
    from . import setfit  # noqa: F401
    from . import self_knowledge  # noqa: F401
    from . import lof  # noqa: F401
    from . import mahalanobis  # noqa: F401
    from . import setfit_centroid  # noqa: F401
