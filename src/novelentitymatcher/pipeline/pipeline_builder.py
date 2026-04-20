"""Pipeline builder that consolidates 5-stage discovery pipeline construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

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

    collect_sync: Optional[Callable[[List[str]], tuple[Any, dict]]] = None
    collect_async: Optional[Callable[[List[str]], Awaitable[tuple[Any, dict]]]] = None
    detector: Any = None
    clusterer: Any = None
    llm_proposer: Any = None
    use_novelty_detector: bool = True
    clustering_enabled: bool = True
    clustering_backend: str = "auto"
    similarity_threshold: float = 0.75
    min_cluster_size: int = 5
    evidence_enabled: bool = True
    max_keywords: int = 8
    max_examples: int = 4
    token_budget: int = 256
    use_tfidf: bool = True
    run_llm_proposal: bool = True
    existing_classes_resolver: Optional[Callable[[], List[str]]] = None
    context_text: Optional[str] = None
    max_retries: int = 2
    prefer_cluster_level: bool = True


class PipelineBuilder:
    """Builds a 5-stage discovery pipeline orchestrator.

    Consolidates pipeline construction logic that was previously duplicated
    between DiscoveryPipeline and NovelEntityMatcher.
    """

    def __init__(self, config: Optional[PipelineStageConfig] = None, **kwargs: Any):
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
            evidence_enabled=kwargs.get("evidence_enabled", True),
            max_keywords=kwargs.get("max_keywords", 8),
            max_examples=kwargs.get("max_examples", 4),
            token_budget=kwargs.get("token_budget", 256),
            use_tfidf=kwargs.get("use_tfidf", True),
            run_llm_proposal=kwargs.get("run_llm_proposal", True),
            existing_classes_resolver=kwargs.get("existing_classes_resolver"),
            context_text=kwargs.get("context"),
            max_retries=kwargs.get("max_retries", 2),
            prefer_cluster_level=kwargs.get("prefer_cluster_level", True),
        )

    def build(
        self,
        *,
        existing_classes: Optional[List[str]] = None,
        context: Optional[str] = None,
        run_llm_proposal: Optional[bool] = None,
    ) -> PipelineOrchestrator:
        """Build the 5-stage pipeline orchestrator.

        Args:
            existing_classes: Optional list of known classes for proposal stage
            context: Optional context text for LLM proposal
            run_llm_proposal: Override whether to run proposal stage

        Returns:
            Configured PipelineOrchestrator with 5 stages
        """
        from ..novelty.clustering.scalable import ScalableClusterer

        cfg = self._cfg

        if run_llm_proposal is None:
            run_llm_proposal = cfg.run_llm_proposal

        # Ensure clusterer is instantiated if clustering is enabled
        clusterer = cfg.clusterer
        if clusterer is None and cfg.clustering_enabled:
            clusterer = ScalableClusterer(
                backend=cfg.clustering_backend,
                min_cluster_size=cfg.min_cluster_size,
            )

        stages = [
            MatcherMetadataStage(
                collect_sync=cfg.collect_sync,
                collect_async=cfg.collect_async,
            ),
            OODDetectionStage(
                detector=cfg.detector,
                enabled=cfg.use_novelty_detector,
            ),
            CommunityDetectionStage(
                clusterer=clusterer,
                enabled=cfg.clustering_enabled,
                similarity_threshold=cfg.similarity_threshold,
                min_cluster_size=max(2, cfg.min_cluster_size),
            ),
            ClusterEvidenceStage(
                enabled=cfg.evidence_enabled,
                max_keywords=cfg.max_keywords,
                max_examples=cfg.max_examples,
                token_budget=cfg.token_budget,
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
            ),
        ]

        return PipelineOrchestrator(stages=stages)

    @classmethod
    def from_pipeline_config(
        cls,
        config: Any,
        *,
        collect_sync: Optional[Callable[[List[str]], tuple[Any, dict]]] = None,
        collect_async: Optional[
            Callable[[List[str]], Awaitable[tuple[Any, dict]]]
        ] = None,
        detector: Any = None,
        clusterer: Any = None,
        llm_proposer: Any = None,
        existing_classes_resolver: Optional[Callable[[], List[str]]] = None,
    ) -> "PipelineBuilder":
        """Factory to create PipelineBuilder from a PipelineConfig object.

        Args:
            config: PipelineConfig instance with stage settings
            collect_sync: Sync collector for MatcherMetadataStage
            collect_async: Async collector for MatcherMetadataStage
            detector: NoveltyDetector instance
            clusterer: ScalableClusterer instance
            llm_proposer: LLMClassProposer instance
            existing_classes_resolver: Callable returning list of existing classes
        """
        return cls(
            PipelineStageConfig(
                collect_sync=collect_sync,
                collect_async=collect_async,
                detector=detector,
                clusterer=clusterer,
                llm_proposer=llm_proposer,
                use_novelty_detector=config.ood_enabled,
                clustering_enabled=config.clustering_enabled,
                clustering_backend=config.clustering_backend,
                similarity_threshold=config.similarity_threshold,
                min_cluster_size=config.min_cluster_size,
                evidence_enabled=config.evidence_enabled,
                max_keywords=config.max_keywords,
                max_examples=config.max_examples,
                token_budget=config.token_budget,
                use_tfidf=config.use_tfidf,
                run_llm_proposal=config.proposal_enabled,
                existing_classes_resolver=existing_classes_resolver,
                context_text=None,
                max_retries=config.max_retries,
                prefer_cluster_level=config.prefer_cluster_level,
            )
        )
