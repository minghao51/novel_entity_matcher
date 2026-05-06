"""
Main configuration for novelty detection.

The DetectionConfig class is the primary configuration object for the
NoveltyDetector, containing strategy selection, per-strategy configs,
and signal combination settings.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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


class DetectionConfig(BaseModel):
    """
    Main configuration for novelty detection.

    This config specifies which strategies to use, their individual
    configurations, and how to combine their signals.
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    # Strategy selection
    strategies: list[str] = Field(
        default_factory=lambda: ["confidence", "knn_distance", "setfit_centroid"]
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
    confidence: ConfidenceConfig | None = None
    """Configuration for confidence strategy."""

    knn_distance: KNNConfig | None = None
    """Configuration for kNN distance strategy."""

    uncertainty: UncertaintyConfig | None = None
    """Configuration for uncertainty strategy."""

    clustering: ClusteringConfig | None = None
    """Configuration for clustering strategy."""

    self_knowledge: SelfKnowledgeConfig | None = None
    """Configuration for self-knowledge strategy."""

    pattern: PatternConfig | None = None
    """Configuration for pattern strategy."""

    oneclass: OneClassConfig | None = None
    """Configuration for One-Class SVM strategy."""

    prototypical: PrototypicalConfig | None = None
    """Configuration for prototypical strategy."""

    setfit: SetFitConfig | None = None
    """Configuration for SetFit strategy."""

    setfit_centroid: SetFitCentroidConfig | None = None
    """Configuration for SetFit centroid distance strategy."""

    mahalanobis: MahalanobisConfig | None = None
    """Configuration for Mahalanobis distance strategy."""

    lof: LOFConfig | None = None
    """Configuration for Local Outlier Factor strategy."""

    energy_ood: EnergyConfig | None = None
    """Configuration for energy-based OOD strategy."""

    mixture_gaussian: MixtureGaussianConfig | None = None
    """Configuration for mixture Gaussian OOD strategy."""

    react_energy: ReActConfig | None = None
    """Configuration for ReAct + energy OOD strategy."""

    # Signal combination weights
    weights: WeightConfig | None = None
    """Weights for signal combination."""

    # Global settings
    enable_lazy_initialization: bool = Field(default=True)
    """Whether to lazily initialize strategies (only when first used)."""

    debug_mode: bool = Field(default=False)
    """Enable debug mode for verbose logging."""

    candidate_top_k: int = Field(default=5, ge=1)
    """How many matcher candidates to request when collecting metadata."""

    allowed_maturities: list[str] = Field(
        default_factory=lambda: ["production", "experimental", "internal"]
    )
    """Allowed strategy maturity levels. Strategies outside these levels are rejected during validation."""

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
            "setfit_centroid": self.setfit_centroid or SetFitCentroidConfig(),
            "mahalanobis": self.mahalanobis or MahalanobisConfig(),
            "lof": self.lof or LOFConfig(),
            "energy_ood": self.energy_ood or EnergyConfig(),
            "mixture_gaussian": self.mixture_gaussian or MixtureGaussianConfig(),
            "react_energy": self.react_energy or ReActConfig(),
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
        Validate that all configured strategies are available and allowed by maturity.

        Strategies are registered at module load time via decorators.
        This method only validates — it does not trigger imports.

        Raises:
            ValueError: If an unknown strategy is configured or maturity not allowed
        """
        from ..core.strategies import StrategyRegistry

        for strategy_id in self.strategies:
            if not StrategyRegistry.is_registered(strategy_id):
                available = ", ".join(StrategyRegistry.list_strategies())
                raise ValueError(
                    f"Unknown strategy: '{strategy_id}'. Available: {available}"
                )
            strategy_cls = StrategyRegistry.get(strategy_id)
            strategy_maturity = getattr(strategy_cls, "maturity", "experimental")
            if strategy_maturity not in self.allowed_maturities:
                raise ValueError(
                    f"Strategy '{strategy_id}' has maturity '{strategy_maturity}' "
                    f"which is not in allowed_maturities={self.allowed_maturities}"
                )
