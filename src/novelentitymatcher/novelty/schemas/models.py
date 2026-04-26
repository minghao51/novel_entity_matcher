"""Canonical Pydantic models for novelty detection and discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class NovelSampleMetadata(BaseModel):
    """Metadata for a single sample flagged as novel."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    index: int
    confidence: float
    predicted_class: str
    novelty_score: float | None = None
    margin_score: float | None = None
    entropy_score: float | None = None
    uncertainty_score: float | None = None
    knn_novelty_score: float | None = None
    knn_mean_distance: float | None = None
    knn_max_distance: float | None = None
    cluster_id: int | None = None
    cluster_support_score: float | None = None
    signals: dict[str, bool] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class NovelSampleReport(BaseModel):
    """Novel samples found during a detection run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    novel_samples: list[NovelSampleMetadata] = Field(default_factory=list)
    detection_strategies: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    signal_counts: dict[str, int] = Field(default_factory=dict)


class ClusterEvidence(BaseModel):
    """Compact statistical evidence extracted for a cluster."""

    keywords: list[str] = Field(default_factory=list)
    rake_keywords: list[str] = Field(default_factory=list)
    representative_examples: list[str] = Field(default_factory=list)
    sample_indices: list[int] = Field(default_factory=list)
    predicted_classes: list[str] = Field(default_factory=list)
    confidence_summary: dict[str, float] = Field(default_factory=dict)
    token_budget: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiscoveryCluster(BaseModel):
    """Community of likely novel samples discovered in a batch."""

    cluster_id: int
    sample_indices: list[int] = Field(default_factory=list)
    sample_count: int = Field(ge=0)
    example_texts: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    evidence: ClusterEvidence | None = None
    mean_novelty_score: float | None = None
    mean_confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassProposal(BaseModel):
    """A proposed class for a cluster of novel samples."""

    name: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)
    example_samples: list[str]
    justification: str
    suggested_parent: str | None = None
    source_cluster_ids: list[int] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    discovered_attributes: list[DiscoveredAttribute] | None = None
    attribute_schema: dict[str, Any] | None = None


class DiscoveredAttribute(BaseModel):
    """A discovered attribute/field for a proposed class."""

    name: str
    description: str
    value_type: Literal["string", "number", "boolean", "enum", "date"]
    enum_values: list[str] | None = None
    example_values: list[str] = Field(default_factory=list)


class NovelClassAnalysis(BaseModel):
    """Class proposals generated from a novelty discovery run."""

    proposed_classes: list[ClassProposal] = Field(default_factory=list)
    rejected_as_noise: list[str] = Field(default_factory=list)
    analysis_summary: str
    cluster_count: int = Field(ge=0)
    model_used: str
    validation_errors: list[str] = Field(default_factory=list)
    proposal_metadata: dict[str, Any] = Field(default_factory=dict)


ReviewState = Literal["pending_review", "approved", "rejected", "promoted"]


class ProposalReviewRecord(BaseModel):
    """Lifecycle-aware review record for a proposed class."""

    review_id: str
    discovery_id: str
    proposal_index: int = Field(ge=0)
    proposal_name: str
    state: ReviewState = "pending_review"
    proposal: ClassProposal
    provenance: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    reviewed_at: datetime | None = None
    promoted_at: datetime | None = None


class NovelClassDiscoveryReport(BaseModel):
    """End-to-end report for novelty detection and optional proposal generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    discovery_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    matcher_config: dict[str, Any] = Field(default_factory=dict)
    detection_config: dict[str, Any] = Field(default_factory=dict)
    novel_sample_report: NovelSampleReport
    discovery_clusters: list[DiscoveryCluster] = Field(default_factory=list)
    class_proposals: NovelClassAnalysis | None = None
    review_records: list[ProposalReviewRecord] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    output_file: str | None = None


@dataclass
class PromotionResult:
    """Captures what happened during a promotion."""

    review_record: ProposalReviewRecord
    entities_added: list[dict[str, Any]] = field(default_factory=list)
    index_updated: bool = False
    retrain_required: bool = False

    @property
    def state(self) -> str:
        """Backward-compatible alias for review_record.state."""
        return self.review_record.state

    @property
    def promoted_at(self):
        """Backward-compatible alias for review_record.promoted_at."""
        return self.review_record.promoted_at
