"""
Main configuration for novelty detection.

The DetectionConfig class is the primary configuration object for the
NoveltyDetector, containing strategy selection, per-strategy configs,
and signal combination settings.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field

from .strategies import (
    ConfidenceConfig,
    KNNConfig,
    UncertaintyConfig,
    ClusteringConfig,
    SelfKnowledgeConfig,
    PatternConfig,
    OneClassConfig,
    PrototypicalConfig,
    SetFitConfig,
    SetFitCentroidConfig,
)
from .weights import WeightConfig


class DetectionConfig(BaseModel):
    """
    Main configuration for novelty detection.

    This config specifies which strategies to use, their individual
    configurations, and how to combine their signals.
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    # Strategy selection
    strategies: List[str] = Field(
        default_factory=lambda: ["confidence", "knn_distance"]
    )
    """
    List of strategy IDs to use for novelty detection.

    Available strategies:
    - confidence: Confidence threshold
    - knn_distance: kNN distance-based
    - uncertainty: Margin/entropy uncertainty
    - clustering: Clustering-based
    - self_knowledge: Sparse autoencoder
    - pattern: Pattern-based
    - oneclass: One-Class SVM
    - prototypical: Prototypical networks
    - setfit: SetFit contrastive
    """

    # Signal combination method
    combine_method: str = Field(default="weighted")
    """
    Method for combining strategy signals.

    Options:
    - weighted: Weighted fusion of scores
    - union: Flag if any strategy flags
    - intersection: Flag if all strategies flag
    - voting: Flag if majority of strategies flag
    - meta_learner: Logistic regression meta-learner (requires training)
    """

    # Strategy-specific configurations
    confidence: Optional[ConfidenceConfig] = None
    """Configuration for confidence strategy."""

    knn_distance: Optional[KNNConfig] = None
    """Configuration for kNN distance strategy."""

    uncertainty: Optional[UncertaintyConfig] = None
    """Configuration for uncertainty strategy."""

    clustering: Optional[ClusteringConfig] = None
    """Configuration for clustering strategy."""

    self_knowledge: Optional[SelfKnowledgeConfig] = None
    """Configuration for self-knowledge strategy."""

    pattern: Optional[PatternConfig] = None
    """Configuration for pattern strategy."""

    oneclass: Optional[OneClassConfig] = None
    """Configuration for One-Class SVM strategy."""

    prototypical: Optional[PrototypicalConfig] = None
    """Configuration for prototypical strategy."""

    setfit: Optional[SetFitConfig] = None
    """Configuration for SetFit strategy."""

    setfit_centroid: Optional[SetFitCentroidConfig] = None
    """Configuration for SetFit centroid distance strategy."""

    # Signal combination weights
    weights: Optional[WeightConfig] = None
    """Weights for signal combination."""

    # Global settings
    enable_lazy_initialization: bool = Field(default=True)
    """Whether to lazily initialize strategies (only when first used)."""

    debug_mode: bool = Field(default=False)
    """Enable debug mode for verbose logging."""

    candidate_top_k: int = Field(default=5, ge=1)
    """How many matcher candidates to request when collecting metadata."""

    def get_strategy_config(self, strategy_id: str) -> Any:
        """
        Get configuration for a specific strategy.

        Returns the strategy-specific config if it exists, otherwise
        returns a default config for that strategy.

        Args:
            strategy_id: The strategy identifier

        Returns:
            Strategy-specific configuration object
        """
        config_map = {
            "confidence": self.confidence or ConfidenceConfig(),
            "knn_distance": self.knn_distance or KNNConfig(),
            "uncertainty": self.uncertainty or UncertaintyConfig(),
            "clustering": self.clustering or ClusteringConfig(),
            "self_knowledge": self.self_knowledge or SelfKnowledgeConfig(),
            "pattern": self.pattern or PatternConfig(),
            "oneclass": self.oneclass or OneClassConfig(),
            "prototypical": self.prototypical or PrototypicalConfig(),
            "setfit": self.setfit or SetFitConfig(),
        }

        return config_map.get(strategy_id)

    def get_weight_config(self) -> WeightConfig:
        """
        Get the weight configuration, with defaults if not set.

        Returns:
            WeightConfig instance
        """
        if self.weights is None:
            return WeightConfig()
        return self.weights

    def validate_strategies(self) -> None:
        """
        Validate that all configured strategies are available.

        Strategies are registered at module load time via decorators.
        This method only validates — it does not trigger imports.

        Raises:
            ValueError: If an unknown strategy is configured
        """
        from ..core.strategies import StrategyRegistry

        for strategy_id in self.strategies:
            if not StrategyRegistry.is_registered(strategy_id):
                available = ", ".join(StrategyRegistry.list_strategies())
                raise ValueError(
                    f"Unknown strategy: '{strategy_id}'. Available: {available}"
                )
