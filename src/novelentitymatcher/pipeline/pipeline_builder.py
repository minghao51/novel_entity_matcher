"""Pipeline builder that consolidates 5-stage discovery pipeline construction."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .adapters import (
    ClusterEvidenceStage,
    CommunityDetectionStage,
    MatcherMetadataStage,
    OODDetectionStage,
    ProposalStage,
)
from .orchestrator import PipelineOrchestrator


@dataclass
class PipelineStageConfig:
    """Configuration for a single pipeline stage."""

    match_enabled: bool = True
    collect_sync: Callable[[list[str]], tuple[Any, dict]] | None = None
    collect_async: Callable[[list[str]], Awaitable[tuple[Any, dict]]] | None = None
    detector: Any = None
    clusterer: Any = None
    llm_proposer: Any = None
    use_novelty_detector: bool = True
    clustering_enabled: bool = True
    clustering_backend: str = "auto"
    similarity_threshold: float = 0.75
    min_cluster_size: int = 5
    clustering_metric: str = "cosine"
    clustering_min_samples: int | None = None
    clustering_cluster_selection_epsilon: float = 0.0
    evidence_enabled: bool = True
    evidence_method: str = "tfidf"
    max_keywords: int = 8
    max_examples: int = 4
    token_budget: int = 256
    use_tfidf: bool | None = None
    run_llm_proposal: bool = True
    existing_classes_resolver: Callable[[], list[str]] | None = None
    context_text: str | None = None
    max_retries: int = 2
    prefer_cluster_level: bool = True
    ood_strategies: list[str] | None = None
    ood_calibration_mode: str = "none"
    ood_calibration_alpha: float = 0.1
    ood_mahalanobis_mode: str = "class_conditional"
    proposal_mode: str = "cluster"
    proposal_schema_discovery: bool = False
    proposal_schema_max_attributes: int = 10
    proposal_hierarchical: bool = True


class PipelineBuilder:
    """Builds a 5-stage discovery pipeline orchestrator.

    Consolidates pipeline construction logic that was previously duplicated
    between DiscoveryPipeline and NovelEntityMatcher.
    """

    def __init__(self, config: PipelineStageConfig | None = None, **kwargs: Any):
        if config is not None:
            self._cfg = config
        else:
            self._cfg = self._from_kwargs(kwargs)

    def _from_kwargs(self, kwargs: Any) -> PipelineStageConfig:
        detection_config = kwargs.get("detection_config")
        clustering_cfg = detection_config.clustering if detection_config else None
        min_cluster_size = (
            clustering_cfg.min_cluster_size
            if clustering_cfg
            else kwargs.get("min_cluster_size", 5)
        )

        return PipelineStageConfig(
            match_enabled=kwargs.get("match_enabled", True),
            collect_sync=kwargs.get("collect_sync"),
            collect_async=kwargs.get("collect_async"),
            detector=kwargs.get("detector"),
            clusterer=kwargs.get("clusterer"),
            llm_proposer=kwargs.get("llm_proposer"),
            use_novelty_detector=kwargs.get("use_novelty_detector", True),
            clustering_enabled=kwargs.get("clustering_enabled", True),
            clustering_backend=kwargs.get("clustering_backend", "auto"),
            similarity_threshold=kwargs.get("similarity_threshold", 0.75),
            min_cluster_size=min_cluster_size,
            clustering_metric=kwargs.get("clustering_metric", "cosine"),
            clustering_min_samples=kwargs.get("clustering_min_samples"),
            clustering_cluster_selection_epsilon=kwargs.get(
                "clustering_cluster_selection_epsilon", 0.0
            ),
            evidence_enabled=kwargs.get("evidence_enabled", True),
            evidence_method=kwargs.get("evidence_method", "tfidf"),
            max_keywords=kwargs.get("max_keywords", 8),
            max_examples=kwargs.get("max_examples", 4),
            token_budget=kwargs.get("token_budget", 256),
            use_tfidf=kwargs.get("use_tfidf"),
            run_llm_proposal=kwargs.get("run_llm_proposal", True),
            existing_classes_resolver=kwargs.get("existing_classes_resolver"),
            context_text=kwargs.get("context"),
            max_retries=kwargs.get("max_retries", 2),
            prefer_cluster_level=kwargs.get("prefer_cluster_level", True),
            ood_strategies=kwargs.get("ood_strategies"),
            ood_calibration_mode=kwargs.get("ood_calibration_mode", "none"),
            ood_calibration_alpha=kwargs.get("ood_calibration_alpha", 0.1),
            ood_mahalanobis_mode=kwargs.get(
                "ood_mahalanobis_mode", "class_conditional"
            ),
            proposal_mode=kwargs.get("proposal_mode", "cluster"),
            proposal_schema_discovery=kwargs.get("proposal_schema_discovery", False),
            proposal_schema_max_attributes=kwargs.get(
                "proposal_schema_max_attributes", 10
            ),
            proposal_hierarchical=kwargs.get("proposal_hierarchical", True),
        )

    def build(
        self,
        *,
        existing_classes: list[str] | None = None,
        context: str | None = None,
        run_llm_proposal: bool | None = None,
    ) -> PipelineOrchestrator:
        """Build the 5-stage pipeline orchestrator."""
        cfg = self._cfg
        if not cfg.match_enabled:
            raise ValueError(
                "match_enabled=False is not supported because downstream stages "
                "require matcher metadata artifacts."
            )

        if run_llm_proposal is None:
            run_llm_proposal = cfg.run_llm_proposal

        clusterer = cfg.clusterer
        if clusterer is None and cfg.clustering_enabled:
            try:
                from ..novelty.clustering.scalable import ScalableClusterer

                clusterer = ScalableClusterer(
                    backend=cfg.clustering_backend,
                    min_cluster_size=cfg.min_cluster_size,
                    min_samples=cfg.clustering_min_samples or cfg.min_cluster_size,
                    cluster_selection_epsilon=cfg.clustering_cluster_selection_epsilon,
                    umap_metric=cfg.clustering_metric,
                )
            except ImportError:
                from ..utils.logging_config import get_logger

                get_logger(__name__).warning(
                    "ScalableClusterer not available; disabling clustering"
                )
                clusterer = None

        stages = [
            MatcherMetadataStage(
                collect_sync=cfg.collect_sync,
                collect_async=cfg.collect_async,
            ),
            OODDetectionStage(
                detector=cfg.detector,
                enabled=cfg.use_novelty_detector,
                ood_strategies=cfg.ood_strategies,
                ood_calibration_mode=cfg.ood_calibration_mode,
                ood_calibration_alpha=cfg.ood_calibration_alpha,
                ood_mahalanobis_mode=cfg.ood_mahalanobis_mode,
            ),
            CommunityDetectionStage(
                clusterer=clusterer,
                enabled=cfg.clustering_enabled,
                similarity_threshold=cfg.similarity_threshold,
                min_cluster_size=max(2, cfg.min_cluster_size),
                clustering_metric=cfg.clustering_metric,
            ),
            ClusterEvidenceStage(
                enabled=cfg.evidence_enabled,
                max_keywords=cfg.max_keywords,
                max_examples=cfg.max_examples,
                token_budget=cfg.token_budget,
                evidence_method=cfg.evidence_method,
                use_tfidf=cfg.use_tfidf,
            ),
            ProposalStage(
                proposer=cfg.llm_proposer,
                existing_classes_resolver=cfg.existing_classes_resolver
                or (lambda: existing_classes or []),
                enabled=run_llm_proposal,
                context_text=context or cfg.context_text,
                max_retries=cfg.max_retries,
                force_cluster_level=cfg.prefer_cluster_level,
                proposal_mode=cfg.proposal_mode,
                proposal_schema_discovery=cfg.proposal_schema_discovery,
                proposal_schema_max_attributes=cfg.proposal_schema_max_attributes,
                proposal_hierarchical=cfg.proposal_hierarchical,
            ),
        ]

        return PipelineOrchestrator(stages=stages)

    @classmethod
    def from_pipeline_config(
        cls,
        config: Any,
        *,
        collect_sync: Callable[[list[str]], tuple[Any, dict]] | None = None,
        collect_async: Callable[[list[str]], Awaitable[tuple[Any, dict]]] | None = None,
        detector: Any = None,
        clusterer: Any = None,
        llm_proposer: Any = None,
        existing_classes_resolver: Callable[[], list[str]] | None = None,
    ) -> PipelineBuilder:
        """Factory to create PipelineBuilder from a PipelineConfig object."""
        return cls(
            PipelineStageConfig(
                collect_sync=collect_sync,
                collect_async=collect_async,
                match_enabled=config.match_enabled,
                detector=detector,
                clusterer=clusterer,
                llm_proposer=llm_proposer,
                use_novelty_detector=config.ood_enabled,
                clustering_enabled=config.clustering_enabled,
                clustering_backend=config.clustering_backend,
                similarity_threshold=config.similarity_threshold,
                min_cluster_size=config.min_cluster_size,
                clustering_metric=getattr(config, "clustering_metric", "cosine"),
                clustering_min_samples=getattr(config, "clustering_min_samples", None),
                clustering_cluster_selection_epsilon=getattr(
                    config, "clustering_cluster_selection_epsilon", 0.0
                ),
                evidence_enabled=config.evidence_enabled,
                evidence_method=getattr(config, "evidence_method", "tfidf"),
                max_keywords=config.max_keywords,
                max_examples=config.max_examples,
                token_budget=config.token_budget,
                use_tfidf=getattr(config, "use_tfidf", None),
                run_llm_proposal=config.proposal_enabled,
                existing_classes_resolver=existing_classes_resolver,
                context_text=None,
                max_retries=config.max_retries,
                prefer_cluster_level=config.prefer_cluster_level,
                ood_strategies=getattr(config, "ood_strategies", None),
                ood_calibration_mode=getattr(config, "ood_calibration_mode", "none"),
                ood_calibration_alpha=getattr(config, "ood_calibration_alpha", 0.1),
                ood_mahalanobis_mode=getattr(
                    config, "ood_mahalanobis_mode", "class_conditional"
                ),
                proposal_mode=getattr(config, "proposal_mode", "cluster"),
                proposal_schema_discovery=getattr(
                    config, "proposal_schema_discovery", False
                ),
                proposal_schema_max_attributes=getattr(
                    config, "proposal_schema_max_attributes", 10
                ),
                proposal_hierarchical=getattr(config, "proposal_hierarchical", True),
            )
        )
