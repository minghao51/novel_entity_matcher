"""Shared match-pipeline logic used by both NovelEntityMatcher and DiscoveryPipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..pipeline.adapters import (
    ClusterEvidenceStage,
    CommunityDetectionStage,
    MatcherMetadataStage,
    OODDetectionStage,
    ProposalStage,
)
from ..pipeline.contracts import StageContext
from ..pipeline.match_result import MatchResultWithMetadata
from ..pipeline.orchestrator import PipelineOrchestrator

class MatchPipelineMixin:
    """Shared pipeline construction and result building for novelty-aware matching.

    Concrete classes must provide:
    - self.matcher (Matcher instance)
    - self.detector (NoveltyDetector instance)
    - self.clusterer (ScalableClusterer instance)
    - self.llm_proposer (LLMClassProposer instance)
    - self.use_novelty_detector (bool)
    - self.acceptance_threshold (float)
    - self.entities (list[dict])
    """

    matcher: Any
    detector: Any
    clusterer: Any
    llm_proposer: Any
    use_novelty_detector: bool
    acceptance_threshold: float
    entities: list[dict[str, Any]]

    def _derive_existing_classes(
        self, existing_classes: Optional[List[str]] = None
    ) -> List[str]:
        if existing_classes:
            return list(existing_classes)
        if self.entities:
            return [str(entity["id"]) for entity in self.entities if "id" in entity]
        reference = self.get_reference_corpus()
        return list(reference.get("labels", []))

    def get_reference_corpus(self) -> Dict[str, Any]:
        return self.matcher.get_reference_corpus()

    def _build_match_result(
        self,
        query: str,
        match_result: MatchResultWithMetadata,
        reference_corpus: Dict[str, Any],
        existing_classes: Optional[List[str]] = None,
        return_alternatives: bool = False,
    ) -> "NovelEntityMatchResult":
        from ..novelty.entity_matcher import NovelEntityMatchResult

        record = match_result.records[0]
        predicted_id = record.predicted_id if record.predicted_id else None
        score = float(record.confidence)
        alternatives = list(record.candidates)

        if self.use_novelty_detector:
            report = self.detector.detect_novel_samples(
                texts=[query],
                confidences=np.asarray(match_result.confidences, dtype=float),
                embeddings=np.asarray(match_result.embeddings),
                predicted_classes=list(match_result.predictions),
                candidate_results=match_result.candidate_results,
                reference_embeddings=reference_corpus["embeddings"],
                reference_labels=reference_corpus["labels"],
            )
            sample = report.novel_samples[0] if report.novel_samples else None
        else:
            report = None
            sample = None

        is_novel = False
        novel_score = max(0.0, 1.0 - score)
        signals: dict[str, bool] = {}
        if sample is not None:
            is_novel = True
            novel_score = float(sample.novelty_score or 0.0)
            signals = dict(sample.signals)

        accepted_known = (
            predicted_id not in (None, "unknown")
            and score >= self.acceptance_threshold
            and not is_novel
        )

        if accepted_known:
            match_method = "accepted_known"
        elif is_novel:
            match_method = "novelty_detector"
        elif predicted_id in (None, "unknown"):
            match_method = "no_match"
        else:
            match_method = "below_acceptance_threshold"

        return NovelEntityMatchResult(
            id=predicted_id if accepted_known else None,
            score=score,
            is_match=accepted_known,
            is_novel=is_novel,
            novel_score=novel_score,
            match_method=match_method,
            alternatives=alternatives if return_alternatives else [],
            signals=signals,
            predicted_id=predicted_id,
            metadata={
                "query": query,
                "acceptance_threshold": self.acceptance_threshold,
            },
        )


class DiscoveryPipelineMixin(MatchPipelineMixin):
    """Extended mixin with pipeline orchestrator construction.

    Adds methods that build the full 5-stage discovery pipeline.
    Concrete classes must additionally provide config access.
    """

    def _build_discovery_stages(
        self,
        *,
        existing_classes: Optional[List[str]] = None,
        context: Optional[str] = None,
        run_llm_proposal: bool = True,
        clustering_enabled: bool = True,
        min_cluster_size: int = 5,
        evidence_enabled: bool = True,
        max_keywords: int = 10,
        max_examples: int = 5,
        token_budget: int = 256,
    ) -> PipelineOrchestrator:
        return PipelineOrchestrator(
            stages=[
                MatcherMetadataStage(
                    collect_sync=self._collect_match_result_sync,
                    collect_async=self._collect_match_result_async,
                ),
                OODDetectionStage(
                    detector=self.detector,
                    enabled=self.use_novelty_detector,
                ),
                CommunityDetectionStage(
                    clusterer=self.clusterer,
                    enabled=clustering_enabled,
                    min_cluster_size=max(2, min_cluster_size),
                ),
                ClusterEvidenceStage(
                    enabled=evidence_enabled,
                    max_keywords=max_keywords,
                    max_examples=max_examples,
                    token_budget=token_budget,
                ),
                ProposalStage(
                    proposer=self.llm_proposer,
                    existing_classes_resolver=lambda: self._derive_existing_classes(
                        existing_classes
                    ),
                    enabled=run_llm_proposal,
                    context_text=context,
                ),
            ]
        )

    def _collect_match_result_sync(
        self, queries: List[str]
    ) -> tuple[MatchResultWithMetadata, Dict[str, Any]]:
        raise NotImplementedError

    async def _collect_match_result_async(
        self, queries: List[str]
    ) -> tuple[MatchResultWithMetadata, Dict[str, Any]]:
        raise NotImplementedError
