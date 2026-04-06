"""
Signal combination weight configuration.

Defines weights for combining signals from different strategies
and thresholds for novelty decision making.
"""

from pydantic import BaseModel, Field


class WeightConfig(BaseModel):
    """
    Weights for signal combination from different strategies.

    Each strategy's contribution to the final novelty score is weighted.
    Weights should sum to approximately 1.0, but this is not enforced
    as normalization is applied during combination.
    """

    # Core strategy weights
    confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    """Weight for confidence threshold strategy."""

    uncertainty: float = Field(default=0.35, ge=0.0, le=1.0)
    """Weight for uncertainty-based strategy."""

    knn: float = Field(default=0.45, ge=0.0, le=1.0)
    """Weight for kNN distance-based strategy."""

    cluster: float = Field(default=0.2, ge=0.0, le=1.0)
    """Weight for clustering-based strategy."""

    # New strategy weights (lower weights as they're experimental)
    self_knowledge: float = Field(default=0.15, ge=0.0, le=1.0)
    """Weight for sparse autoencoder strategy."""

    pattern: float = Field(default=0.2, ge=0.0, le=1.0)
    """Weight for pattern-based strategy."""

    oneclass: float = Field(default=0.1, ge=0.0, le=1.0)
    """Weight for One-Class SVM strategy."""

    prototypical: float = Field(default=0.1, ge=0.0, le=1.0)
    """Weight for prototypical networks strategy."""

    setfit: float = Field(default=0.1, ge=0.0, le=1.0)
    """Weight for SetFit contrastive strategy."""

    setfit_centroid: float = Field(default=0.45, ge=0.0, le=1.0)
    """Weight for SetFit centroid distance strategy (recommended, highest weight)."""

    # Adaptive weight control
    adaptive: bool = Field(default=False)
    """Enable adaptive weight computation based on dataset characteristics."""

    # Decision thresholds
    novelty_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    """Final novelty score threshold for flagging samples."""

    knn_gate_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    """kNN gate threshold - samples above this are always considered novel."""

    strong_uncertainty_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    """Strong uncertainty threshold - samples above this are always novel."""

    strong_knn_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    """Strong kNN threshold - samples above this are always novel."""

    def normalize_weights(self) -> "WeightConfig":
        """
        Normalize weights to sum to 1.0.

        Returns:
            A new WeightConfig with normalized weights
        """
        strategy_weights = [
            self.uncertainty,
            self.knn,
            self.cluster,
            self.self_knowledge,
            self.pattern,
            self.oneclass,
            self.prototypical,
            self.setfit,
            self.setfit_centroid,
        ]

        total = sum(strategy_weights)
        if total == 0:
            return self

        factor = 1.0 / total

        return WeightConfig(
            uncertainty=self.uncertainty * factor,
            knn=self.knn * factor,
            cluster=self.cluster * factor,
            self_knowledge=self.self_knowledge * factor,
            pattern=self.pattern * factor,
            oneclass=self.oneclass * factor,
            prototypical=self.prototypical * factor,
            setfit=self.setfit * factor,
            setfit_centroid=self.setfit_centroid * factor,
            # Keep thresholds unchanged
            novelty_threshold=self.novelty_threshold,
            knn_gate_threshold=self.knn_gate_threshold,
            strong_uncertainty_threshold=self.strong_uncertainty_threshold,
            strong_knn_threshold=self.strong_knn_threshold,
        )
