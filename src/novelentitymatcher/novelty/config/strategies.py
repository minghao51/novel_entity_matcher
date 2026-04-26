"""
Strategy-specific configuration classes.

Each strategy has its own configuration with sensible defaults.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ConfidenceConfig(BaseModel):
    """Configuration for confidence threshold strategy."""

    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    """Minimum confidence threshold. Samples below this are flagged as novel."""


class KNNConfig(BaseModel):
    """Configuration for kNN distance-based strategy."""

    k: int = Field(default=20, ge=1, le=100)
    """Number of nearest neighbors to consider."""

    distance_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    """Threshold for kNN distance score. Samples above this are flagged."""

    strong_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    """Strong novelty threshold for high-confidence detection."""

    metric: str = Field(default="cosine")
    """Distance metric to use ('cosine', 'euclidean', etc.)."""


class UncertaintyConfig(BaseModel):
    """Configuration for uncertainty-based strategy."""

    margin_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    """Margin between top predictions. Small margin = high uncertainty."""

    entropy_threshold: float = Field(default=1.5, ge=0.0)
    """Entropy threshold for uncertainty detection."""


class ClusteringConfig(BaseModel):
    """Configuration for clustering-based strategy."""

    min_cluster_size: int = Field(default=5, ge=1)
    """Minimum cluster size to be considered valid."""

    persistence_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    """Persistence threshold for cluster stability."""

    cohesion_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    """Cohesion threshold for cluster compactness."""

    hdbscan_min_cluster_size: int = Field(default=5, ge=1)
    """min_cluster_size parameter for HDBSCAN."""

    hdbscan_min_samples: int = Field(default=1, ge=1)
    """min_samples parameter for HDBSCAN."""

    cluster_selection_epsilon: float = Field(default=0.0, ge=0.0)
    """cluster_selection_epsilon for HDBSCAN."""


class SelfKnowledgeConfig(BaseModel):
    """Configuration for sparse autoencoder strategy."""

    hidden_dim: int = Field(default=128, ge=1)
    """Hidden dimension for the autoencoder."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Reconstruction error threshold for novelty detection."""

    epochs: int = Field(default=100, ge=1)
    """Number of training epochs."""

    batch_size: int = Field(default=32, ge=1)
    """Training batch size."""

    learning_rate: float = Field(default=0.001, gt=0.0)
    """Learning rate for training."""


class PatternConfig(BaseModel):
    """Configuration for pattern-based strategy."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Novelty score threshold for pattern-based detection."""

    char_ngram_n: int = Field(default=3, ge=1, le=5)
    """Character n-gram size for pattern extraction."""

    char_4gram_n: int = Field(default=4, ge=1, le=5)
    """Character 4-gram size."""

    prefix_suffix_n: int = Field(default=3, ge=1, le=5)
    """Prefix/suffix length for distribution analysis."""


class OneClassConfig(BaseModel):
    """Configuration for One-Class SVM strategy."""

    nu: float = Field(default=0.1, ge=0.0, le=1.0)
    """Expected outlier fraction. Lower = stricter boundary."""

    kernel: str = Field(default="rbf")
    """SVM kernel type ('rbf', 'linear', 'poly', 'sigmoid')."""

    gamma: str = Field(default="scale")
    """Kernel coefficient ('scale', 'auto', or float)."""

    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    """Sentence transformer model name for embeddings."""


class PrototypicalConfig(BaseModel):
    """Configuration for prototypical networks strategy."""

    distance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Distance threshold for novelty detection."""

    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    """Sentence transformer model name for embeddings."""

    support_samples_per_class: int = Field(default=5, ge=1)
    """Number of support samples per class for prototype computation."""


class MahalanobisConfig(BaseModel):
    """Configuration for Mahalanobis distance-based strategy."""

    threshold: float = Field(default=3.0, gt=0.0)
    """Mahalanobis distance threshold. Samples above this are flagged as novel."""

    regularization: float = Field(default=1e-4, gt=0.0)
    """Covariance matrix regularization (ridge) for numerical stability."""

    use_class_conditional: bool = Field(default=True)
    """Whether to use per-class distributions (True) or a single global distribution (False)."""

    calibration_mode: Literal["none", "conformal"] = Field(default="none")
    """Calibration mode: 'none' for raw threshold, 'conformal' for p-value calibration."""

    calibration_alpha: float = Field(default=0.1, gt=0.0, le=1.0)
    """Significance level for conformal prediction. Lower = stricter."""

    calibration_method: Literal["split", "mondrian"] = Field(default="split")
    """Conformal calibration method: 'split' or 'mondrian' (class-conditional)."""

    calibration_set_fraction: float = Field(default=0.2, gt=0.0, le=0.5)
    """Fraction of reference data held out for conformal calibration."""


class LOFConfig(BaseModel):
    """Configuration for Local Outlier Factor (LOF) strategy."""

    n_neighbors: int = Field(default=20, ge=2)
    """Number of neighbors to use for LOF."""

    contamination: float = Field(default=0.1, gt=0.0, le=0.5)
    """Expected proportion of outliers in the reference set."""

    metric: str = Field(default="cosine")
    """Distance metric to use ('cosine', 'euclidean', 'manhattan', etc.)."""

    score_threshold: float = Field(default=0.0)
    """LOF score threshold. Samples below this are flagged as novel."""


class SetFitConfig(BaseModel):
    """Configuration for SetFit contrastive strategy."""

    margin: float = Field(default=0.5, ge=0.0)
    """Contrastive loss margin."""

    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    """Sentence transformer model name."""

    epochs: int = Field(default=10, ge=1)
    """Number of training epochs."""

    batch_size: int = Field(default=16, ge=1)
    """Training batch size."""

    learning_rate: float = Field(default=2e-5, gt=0.0)
    """Learning rate for fine-tuning."""

    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    """Similarity threshold for novelty detection."""


class SetFitCentroidConfig(BaseModel):
    """Configuration for SetFit centroid distance strategy."""

    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    """
    Cosine distance threshold for novelty detection.
    Samples with min centroid distance above this are flagged as novel.
    If None, threshold is auto-calibrated from reference set (95th percentile).
    """
