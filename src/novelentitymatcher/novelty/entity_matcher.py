"""
Primary orchestration API for classification plus novel-class detection.

This module promotes NovelEntityMatcher to the main public entry point for
novelty-aware matching. It wraps a fitted ``Matcher`` together with the
multi-signal ``NoveltyDetector`` and optional ``LLMClassProposer``.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .core.detector import NoveltyDetector
from .clustering.scalable import ScalableClusterer
from .config.base import DetectionConfig
from .config.strategies import ClusteringConfig, ConfidenceConfig, KNNConfig
from .proposal.llm import LLMClassProposer
from .schemas import NovelClassDiscoveryReport, NovelSampleMetadata, NovelSampleReport
from .storage.persistence import export_summary, save_proposals
from .storage.review import ProposalReviewManager
from ..core.matcher import Matcher
from ..pipeline.contracts import StageContext
from ..pipeline.discovery_support import (
    build_novel_match_result,
    build_stage_config,
    derive_existing_classes,
)
from ..pipeline.match_result import MatchResultWithMetadata
from ..pipeline.orchestrator import PipelineOrchestrator
from ..pipeline.pipeline_builder import PipelineBuilder
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class NovelEntityMatchResult:
    """Operational result for a single novelty-aware match decision."""

    id: str | None
    score: float
    is_match: bool
    is_novel: bool
    novel_score: float | None = None
    match_method: str = "accepted_known"
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, bool] = field(default_factory=dict)
    predicted_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


NoveltyMatchResult = NovelEntityMatchResult


class NovelEntityMatcher:
    """
    Primary public API for novelty-aware matching.

    The class orchestrates three stages:
    1. Retrieve rich matcher metadata with top-k candidates and embeddings.
    2. Score novelty using ANN-backed multi-signal detection.
    3. Optionally propose new class names for novel batches.
    """

    def __init__(
        self,
        entities: Optional[list[dict[str, Any]]] = None,
        *,
        matcher: Optional[Matcher] = None,
        model: str = "potion-32m",
        mode: str = "zero-shot",
        acceptance_threshold: Optional[float] = None,
        detection_config: Optional[Union[DetectionConfig, Dict[str, Any]]] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_keys: Optional[Dict[str, str]] = None,
        output_dir: str = "./proposals",
        auto_save: bool = True,
        match_threshold: Optional[float] = None,
        novelty_strategy: str = "confidence",
        confidence_threshold: float = 0.3,
        knn_k: int = 5,
        knn_distance_threshold: float = 0.6,
        min_cluster_size: int = 5,
        use_novelty_detector: bool = True,
        review_storage_path: str = "./proposals/review_records.json",
    ):
        if matcher is None:
            if entities is None:
                raise ValueError("entities is required when matcher is not provided")
            threshold = (
                acceptance_threshold
                if acceptance_threshold is not None
                else (match_threshold if match_threshold is not None else 0.5)
            )
            matcher = Matcher(
                entities=entities,
                model=model,
                mode=mode,
                threshold=threshold,
            )

        self.matcher = matcher
        self.entities = (
            entities if entities is not None else list(getattr(matcher, "entities", []))
        )
        self.acceptance_threshold = (
            acceptance_threshold
            if acceptance_threshold is not None
            else (
                match_threshold
                if match_threshold is not None
                else getattr(self.matcher, "threshold", 0.5)
            )
        )
        self.output_dir = output_dir
        self.auto_save = auto_save
        self.use_novelty_detector = use_novelty_detector

        self.detection_config = self._coerce_detection_config(
            detection_config=detection_config,
            novelty_strategy=novelty_strategy,
            confidence_threshold=confidence_threshold,
            knn_k=knn_k,
            knn_distance_threshold=knn_distance_threshold,
            min_cluster_size=min_cluster_size,
        )
        self.detector = NoveltyDetector(config=self.detection_config)
        clustering_config = self.detection_config.clustering or ClusteringConfig(
            min_cluster_size=min_cluster_size
        )
        self.clusterer = ScalableClusterer(
            min_cluster_size=clustering_config.min_cluster_size
        )
        self.llm_proposer = LLMClassProposer(
            primary_model=llm_model,
            provider=llm_provider,
            api_keys=llm_api_keys,
        )
        self.review_manager = ProposalReviewManager(review_storage_path)

    @staticmethod
    def _coerce_detection_config(
        detection_config: Optional[Union[DetectionConfig, Dict[str, Any]]],
        novelty_strategy: str,
        confidence_threshold: float,
        knn_k: int,
        knn_distance_threshold: float,
        min_cluster_size: int,
    ) -> DetectionConfig:
        if isinstance(detection_config, DetectionConfig):
            return detection_config
        if isinstance(detection_config, dict):
            return DetectionConfig(**detection_config)

        strategies: list[str]
        strategy = novelty_strategy.lower()
        if strategy == "confidence":
            strategies = ["confidence"]
        elif strategy in {"knn", "knn_distance", "distance"}:
            strategies = ["confidence", "knn_distance"]
        elif strategy in {"cluster", "clustering"}:
            strategies = ["confidence", "knn_distance", "clustering"]
        else:
            strategies = ["confidence", "knn_distance", "clustering"]

        return DetectionConfig(
            strategies=strategies,
            confidence=ConfidenceConfig(threshold=confidence_threshold),
            knn_distance=KNNConfig(
                k=knn_k,
                distance_threshold=knn_distance_threshold,
            ),
            clustering=ClusteringConfig(min_cluster_size=min_cluster_size),
        )

    def fit(
        self,
        training_data: Optional[List[dict]] = None,
        mode: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> "NovelEntityMatcher":
        self.matcher.fit(
            training_data=training_data,
            mode=mode,
            show_progress=show_progress,
            **kwargs,
        )
        return self

    async def fit_async(
        self,
        training_data: Optional[List[dict]] = None,
        mode: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> "NovelEntityMatcher":
        await self.matcher.fit_async(
            training_data=training_data,
            mode=mode,
            show_progress=show_progress,
            **kwargs,
        )
        return self

    def set_threshold(self, threshold: float) -> "NovelEntityMatcher":
        self.acceptance_threshold = threshold
        self.matcher.set_threshold(threshold)
        return self

    def adjust_threshold(self, new_threshold: float) -> None:
        self.set_threshold(new_threshold)

    def get_reference_corpus(self) -> Dict[str, Any]:
        return self.matcher.get_reference_corpus()

    def set_novelty_detector(self, detector: NoveltyDetector | None) -> None:
        if detector is None:
            self.use_novelty_detector = False
            self.detector = NoveltyDetector(config=self.detection_config)
            return
        self.use_novelty_detector = True
        self.detector = detector

    def get_stats(self) -> dict[str, Any]:
        return {
            "num_entities": len(self.entities),
            "model": getattr(self.matcher, "model_name", None),
            "mode": getattr(self.matcher, "_training_mode", None),
            "acceptance_threshold": self.acceptance_threshold,
            "use_novelty_detector": self.use_novelty_detector,
            "detection_config": self.detection_config.model_dump(),
        }

    def _derive_existing_classes(
        self, existing_classes: Optional[List[str]] = None
    ) -> List[str]:
        return derive_existing_classes(
            entities=self.entities,
            get_reference_corpus=self.get_reference_corpus,
            existing_classes=existing_classes,
        )

    async def _collect_match_result_async(
        self, queries: List[str]
    ) -> tuple[MatchResultWithMetadata, Dict[str, Any]]:
        match_async = getattr(self.matcher, "match_async", None)
        if callable(match_async):
            result = await match_async(
                queries,
                return_metadata=True,
                top_k=self.detection_config.candidate_top_k,
            )
        else:
            result = await asyncio.to_thread(
                self.matcher.match,
                queries,
                return_metadata=True,
                top_k=self.detection_config.candidate_top_k,
            )

        return result, self.get_reference_corpus()

    def _collect_match_result_sync(
        self, queries: List[str]
    ) -> tuple[MatchResultWithMetadata, Dict[str, Any]]:
        result = self.matcher.match(
            queries,
            return_metadata=True,
            top_k=self.detection_config.candidate_top_k,
        )
        return result, self.get_reference_corpus()

    def _build_orchestrator(
        self,
        *,
        existing_classes: Optional[List[str]] = None,
        context: Optional[str] = None,
        run_llm_proposal: bool = True,
    ) -> PipelineOrchestrator:
        clustering_cfg = self.detection_config.clustering
        stage_config = build_stage_config(
            collect_sync=self._collect_match_result_sync,
            collect_async=self._collect_match_result_async,
            detector=self.detector,
            clusterer=self.clusterer,
            llm_proposer=self.llm_proposer,
            use_novelty_detector=self.use_novelty_detector,
            clustering_enabled=True,
            clustering_backend=getattr(self.clusterer, "backend", "auto"),
            similarity_threshold=0.75,
            min_cluster_size=(
                clustering_cfg.min_cluster_size if clustering_cfg else 5
            ),
            evidence_enabled=True,
            use_tfidf=True,
            run_llm_proposal=run_llm_proposal,
            existing_classes_resolver=lambda: self._derive_existing_classes(
                existing_classes
            ),
            context_text=context,
            max_retries=2,
            prefer_cluster_level=True,
        )
        return PipelineBuilder(stage_config).build(
            existing_classes=existing_classes,
            context=context,
            run_llm_proposal=run_llm_proposal,
        )

    def _coerce_novel_sample_report(self, report: Any) -> NovelSampleReport:
        if isinstance(report, NovelSampleReport):
            return report

        samples = [
            sample
            if isinstance(sample, NovelSampleMetadata)
            else NovelSampleMetadata(
                text=str(getattr(sample, "text", "")),
                index=int(getattr(sample, "index", 0)),
                confidence=float(getattr(sample, "confidence", 0.0)),
                predicted_class=str(getattr(sample, "predicted_class", "unknown")),
                novelty_score=getattr(sample, "novelty_score", None),
                cluster_id=getattr(sample, "cluster_id", None),
                signals=dict(getattr(sample, "signals", {})),
            )
            for sample in getattr(report, "novel_samples", [])
        ]
        return NovelSampleReport(
            novel_samples=samples,
            detection_strategies=list(getattr(report, "detection_strategies", [])),
            config=dict(getattr(report, "config", {})),
            signal_counts=dict(getattr(report, "signal_counts", {})),
        )

    def match(
        self,
        text: str,
        return_alternatives: bool = False,
        existing_classes: Optional[List[str]] = None,
    ) -> NovelEntityMatchResult:
        match_result, reference_corpus = self._collect_match_result_sync([text])
        return build_novel_match_result(
            query=text,
            match_result=match_result,
            reference_corpus=reference_corpus,
            detector=self.detector,
            use_novelty_detector=self.use_novelty_detector,
            acceptance_threshold=self.acceptance_threshold,
            return_alternatives=return_alternatives,
        )

    async def match_async(
        self,
        text: str,
        return_alternatives: bool = False,
        existing_classes: Optional[List[str]] = None,
    ) -> NovelEntityMatchResult:
        match_result, reference_corpus = await self._collect_match_result_async([text])
        return build_novel_match_result(
            query=text,
            match_result=match_result,
            reference_corpus=reference_corpus,
            detector=self.detector,
            use_novelty_detector=self.use_novelty_detector,
            acceptance_threshold=self.acceptance_threshold,
            return_alternatives=return_alternatives,
        )

    def match_batch(
        self,
        texts: list[str],
        return_alternatives: bool = False,
        existing_classes: Optional[List[str]] = None,
    ) -> list[NovelEntityMatchResult]:
        match_result, reference_corpus = self._collect_match_result_sync(texts)
        return [
            build_novel_match_result(
                query=text,
                match_result=MatchResultWithMetadata(
                    predictions=[match_result.predictions[idx]],
                    confidences=np.asarray(
                        [match_result.confidences[idx]], dtype=float
                    ),
                    embeddings=np.asarray([match_result.embeddings[idx]]),
                    metadata={
                        "texts": [text],
                        "top_k": (match_result.metadata or {}).get("top_k"),
                    },
                    candidate_results=[match_result.candidate_results[idx]],
                    records=[match_result.records[idx]],
                ),
                reference_corpus=reference_corpus,
                detector=self.detector,
                use_novelty_detector=self.use_novelty_detector,
                acceptance_threshold=self.acceptance_threshold,
                return_alternatives=return_alternatives,
            )
            for idx, text in enumerate(texts)
        ]

    async def discover_novel_classes(
        self,
        queries: List[str],
        existing_classes: Optional[List[str]] = None,
        context: Optional[str] = None,
        return_metadata: bool = True,
        run_llm_proposal: bool = True,
    ) -> NovelClassDiscoveryReport:
        discovery_id = str(uuid.uuid4())[:8]
        logger.info(
            "[%s] Starting novel class discovery for %s queries",
            discovery_id,
            len(queries),
        )
        pipeline = self._build_orchestrator(
            existing_classes=existing_classes,
            context=context,
            run_llm_proposal=run_llm_proposal,
        )
        context_obj = StageContext(inputs=list(queries))

        if return_metadata:
            pipeline_result = await pipeline.run_async(context_obj)
        else:
            pipeline_result = pipeline.run(context_obj)

        known_classes = self._derive_existing_classes(existing_classes)
        novel_sample_report = self._coerce_novel_sample_report(
            pipeline_result.context.artifacts["novel_sample_report"]
        )
        discovery_clusters = pipeline_result.context.artifacts.get(
            "discovery_clusters", []
        )
        class_proposals = pipeline_result.context.artifacts.get("class_proposals")

        report = NovelClassDiscoveryReport(
            discovery_id=discovery_id,
            timestamp=datetime.now(),
            matcher_config=self._get_matcher_config(),
            detection_config=self.detection_config.model_dump(),
            novel_sample_report=novel_sample_report,
            discovery_clusters=discovery_clusters,
            class_proposals=class_proposals,
            diagnostics={
                "stage_metadata": pipeline_result.context.metadata,
            },
            metadata={
                "num_queries": len(queries),
                "num_existing_classes": len(known_classes),
                "num_novel_samples": len(novel_sample_report.novel_samples),
                "num_discovery_clusters": len(discovery_clusters),
                "context": context,
                "pipeline_stage_metadata": pipeline_result.context.metadata,
            },
        )

        report.review_records = self.review_manager.create_records(report)

        if self.auto_save:
            output_file = save_proposals(report, output_dir=self.output_dir)
            report.output_file = output_file
            summary_path = output_file.replace(
                f".{output_file.split('.')[-1]}",
                "_summary.md",
            )
            export_summary(report, summary_path)
            report.metadata["summary_file"] = summary_path

        return report

    def batch_discover(
        self,
        queries_batch: List[List[str]],
        existing_classes: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> List[NovelClassDiscoveryReport]:
        async def run_all():
            tasks = [
                self.discover_novel_classes(
                    queries=queries,
                    existing_classes=existing_classes,
                    context=context,
                )
                for queries in queries_batch
            ]
            return await asyncio.gather(*tasks)

        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_all())
                return future.result()
        except RuntimeError:
            return asyncio.run(run_all())

    def _get_matcher_config(self) -> Dict[str, Any]:
        config = {
            "matcher_type": self.matcher.__class__.__name__,
        }
        if hasattr(self.matcher, "model_name"):
            config["model"] = str(self.matcher.model_name)
        if hasattr(self.matcher, "threshold"):
            config["threshold"] = str(self.matcher.threshold)
        if hasattr(self.matcher, "_training_mode"):
            config["mode"] = getattr(self.matcher, "_training_mode")
        return config


def create_novel_entity_matcher(
    entities: list[dict[str, Any]],
    model: str = "potion-32m",
    mode: str = "zero-shot",
    threshold: float = 0.5,
    enable_novelty_detection: bool = True,
    **kwargs,
) -> NovelEntityMatcher:
    return NovelEntityMatcher(
        entities=entities,
        model=model,
        mode=mode,
        acceptance_threshold=threshold,
        use_novelty_detector=enable_novelty_detection,
        **kwargs,
    )
