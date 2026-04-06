"""Unified pipeline configuration for the discovery pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


_VALID_OOD_STRATEGIES = {
    "confidence",
    "knn_distance",
    "uncertainty",
    "clustering",
    "self_knowledge",
    "pattern",
    "oneclass",
    "prototypical",
    "setfit",
}

_VALID_CLUSTERING_BACKENDS = {"auto", "hdbscan", "soptics", "umap_hdbscan"}


class PipelineConfig(BaseModel):
    """Unified configuration driving stage selection and optional capabilities."""

    model_config = ConfigDict(validate_assignment=True)

    # --- Matching stage ---
    match_enabled: bool = Field(default=True)
    top_k: int = Field(default=5, ge=1)

    # --- OOD detection ---
    ood_enabled: bool = Field(default=True)
    ood_strategies: List[str] = Field(
        default_factory=lambda: ["confidence", "knn_distance"]
    )
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # --- Clustering ---
    clustering_enabled: bool = Field(default=True)
    clustering_backend: str = Field(default="auto")
    min_cluster_size: int = Field(default=5, ge=1)
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # --- Evidence extraction ---
    evidence_enabled: bool = Field(default=True)
    max_keywords: int = Field(default=8, ge=1)
    max_examples: int = Field(default=4, ge=1)
    token_budget: int = Field(default=256, ge=1)
    use_tfidf: bool = Field(default=True)

    # --- Proposal generation ---
    proposal_enabled: bool = Field(default=True)
    llm_provider: Optional[str] = Field(default=None)
    llm_model: Optional[str] = Field(default=None)
    max_retries: int = Field(default=2, ge=0)
    prefer_cluster_level: bool = Field(default=True)

    # --- HITL ---
    auto_create_review_records: bool = Field(default=True)
    review_storage_path: str = Field(default="./proposals/review_records.json")

    # --- General ---
    output_dir: str = Field(default="./proposals")
    auto_save: bool = Field(default=True)

    # --- Validation ---

    @field_validator("ood_strategies")
    @classmethod
    def validate_ood_strategies(cls, v: List[str]) -> List[str]:
        for strategy in v:
            if strategy not in _VALID_OOD_STRATEGIES:
                raise ValueError(
                    f"Unknown OOD strategy: '{strategy}'. "
                    f"Valid strategies: {sorted(_VALID_OOD_STRATEGIES)}"
                )
        return v

    @field_validator("clustering_backend")
    @classmethod
    def validate_clustering_backend(cls, v: str) -> str:
        if v not in _VALID_CLUSTERING_BACKENDS:
            raise ValueError(
                f"Unknown clustering backend: '{v}'. "
                f"Valid backends: {sorted(_VALID_CLUSTERING_BACKENDS)}"
            )
        return v

    # --- Convenience methods ---

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Construct a PipelineConfig from a plain dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.model_fields})

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return self.model_dump()

    def stages(self) -> List[str]:
        """Return an ordered list of enabled stage names."""
        enabled: List[str] = []
        if self.match_enabled:
            enabled.append("match")
        if self.ood_enabled:
            enabled.append("ood")
        if self.clustering_enabled:
            enabled.append("cluster")
        if self.evidence_enabled:
            enabled.append("evidence")
        if self.proposal_enabled:
            enabled.append("proposal")
        return enabled
