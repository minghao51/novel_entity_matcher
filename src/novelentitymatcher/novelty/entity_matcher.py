"""Primary orchestration API for classification plus novel-class detection.

This module exposes NovelEntityMatcher as the single public entry point for
novelty-aware matching and discovery. It wraps a fitted ``Matcher`` together
with the multi-signal ``NoveltyDetector`` and optional ``LLMClassProposer``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.match_result import MatchResultWithMetadata
from ..core.matcher import Matcher
from ..pipeline.config import PipelineConfig
from ..pipeline.contracts import StageContext
from ..pipeline.discovery_support import (
    build_novel_match_result,
    collect_match_result_async,
    collect_match_result_sync,
)
from ..pipeline.orchestrator import PipelineOrchestrator
from ..pipeline.pipeline_builder import PipelineBuilder, PipelineStageConfig
from ..utils.logging_config import get_logger
from .clustering.scalable import ScalableClusterer
from .config.base import DetectionConfig
from .config.strategies import (
    ClusteringConfig,
    ConfidenceConfig,
    EnergyConfig,
    KNNConfig,
    MahalanobisConfig,
    MixtureGaussianConfig,
    ReActConfig,
)
from .core.detector import NoveltyDetector
from .discovery_base import DiscoveryBase
from .proposal.llm import LLMClassProposer
from .schemas import (
    NovelClassDiscoveryReport,
    PromotionResult,
    ProposalReviewRecord,
)
from .storage.review import ProposalReviewManager

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


class NovelEntityMatcher(DiscoveryBase):
    """Primary public API for novelty-aware matching and discovery.

    Orchestrates matching, novelty detection, clustering, and optional
    LLM-based class proposal through a PipelineOrchestrator.  Accepts
    either flat keyword arguments or a ``PipelineConfig`` for unified
    configuration.
    """

    def __init__(
        self,
        entities: list[dict[str, Any]] | None = None,
        *,
        matcher: Matcher | None = None,
        config: PipelineConfig | None = None,
        model: str = "potion-32m",
        mode: str = "zero-shot",
        acceptance_threshold: float | None = None,
        detection_config: DetectionConfig | dict[str, Any] | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_api_keys: dict[str, str] | None = None,
        output_dir: str = "./proposals",
        auto_save: bool = True,
        match_threshold: float | None = None,
        novelty_strategy: str = "confidence",
        confidence_threshold: float | None = None,
        knn_k: int = 5,
        knn_distance_threshold: float = 0.6,
        min_cluster_size: int = 5,
        use_novelty_detector: bool = True,
        review_storage_path: str = "./proposals/review_records.json",
        **kwargs: Any,
    ):
        self._pipeline_config = config

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

        if config is not None:
            self.detection_config = self._build_detection_config(
                kwargs, detection_config=detection_config
            )
            self.use_novelty_detector = config.ood_enabled
            clustering_cfg = self.detection_config.clustering or ClusteringConfig(
                min_cluster_size=config.min_cluster_size,
            )
            self.clusterer = ScalableClusterer(
                backend=config.clustering_backend,
                min_cluster_size=clustering_cfg.min_cluster_size,
                min_samples=(
                    config.clustering_min_samples or clustering_cfg.hdbscan_min_samples
                ),
                cluster_selection_epsilon=config.clustering_cluster_selection_epsilon,
                umap_metric=config.clustering_metric,
            )
        else:
            effective_confidence = (
                confidence_threshold if confidence_threshold is not None else 0.3
            )
            self.detection_config = self._coerce_detection_config(
                detection_config=detection_config,
                novelty_strategy=novelty_strategy,
                confidence_threshold=effective_confidence,
                knn_k=knn_k,
                knn_distance_threshold=knn_distance_threshold,
                min_cluster_size=min_cluster_size,
            )
            self.use_novelty_detector = use_novelty_detector
            clustering_cfg = self.detection_config.clustering or ClusteringConfig(
                min_cluster_size=min_cluster_size,
            )
            self.clusterer = ScalableClusterer(
                min_cluster_size=clustering_cfg.min_cluster_size,
            )

        self.detector = NoveltyDetector(config=self.detection_config)

        effective_llm_model = llm_model or (config.llm_model if config else None)
        effective_llm_provider = llm_provider or (
            config.llm_provider if config else None
        )
        self.llm_proposer = LLMClassProposer(
            primary_model=effective_llm_model,
            provider=effective_llm_provider,
            api_keys=llm_api_keys,
        )
        self.review_manager = ProposalReviewManager(review_storage_path)

        if config is not None:
            self._orchestrator = self._build_orchestrator()

    @property
    def config(self) -> PipelineConfig | None:
        return self._pipeline_config

    @property
    def novel_entity_matcher(self) -> NovelEntityMatcher:
        return self

    def _build_detection_config(
        self,
        kwargs: dict[str, Any],
        detection_config: DetectionConfig | dict[str, Any] | None = None,
    ) -> DetectionConfig:
        detection_config = detection_config or kwargs.get("detection_config")
        if isinstance(detection_config, DetectionConfig):
            return detection_config
        if isinstance(detection_config, dict):
            return DetectionConfig(**detection_config)

        assert self._pipeline_config is not None
        cfg = self._pipeline_config
        novelty_strategy = kwargs.get("novelty_strategy", "knn_distance")
        confidence_threshold = kwargs.get(
            "confidence_threshold", cfg.confidence_threshold
        )
        knn_k = kwargs.get("knn_k", 5)
        knn_distance_threshold = kwargs.get("knn_distance_threshold", 0.6)
        min_cluster_size = kwargs.get("min_cluster_size", cfg.min_cluster_size)

        strategies = list(cfg.ood_strategies)
        if not strategies:
            strategies = NovelEntityMatcher._resolve_strategy_list(novelty_strategy)

        return DetectionConfig(
            strategies=strategies,
            confidence=ConfidenceConfig(threshold=confidence_threshold),
            knn_distance=KNNConfig(
                k=knn_k,
                distance_threshold=knn_distance_threshold,
            ),
            clustering=ClusteringConfig(
                min_cluster_size=min_cluster_size,
                hdbscan_min_samples=(cfg.clustering_min_samples or min_cluster_size),
                cluster_selection_epsilon=cfg.clustering_cluster_selection_epsilon,
            ),
            mahalanobis=MahalanobisConfig(
                use_class_conditional=(cfg.ood_mahalanobis_mode == "class_conditional"),
                calibration_mode=cfg.ood_calibration_mode,
                calibration_alpha=cfg.ood_calibration_alpha,
                calibration_method=cfg.ood_calibration_method,
            ),
            energy_ood=(
                kwargs.get("energy_ood")
                if isinstance(kwargs.get("energy_ood"), EnergyConfig)
                else (
                    EnergyConfig(**kwargs["energy_ood"])
                    if isinstance(kwargs.get("energy_ood"), dict)
                    else None
                )
            ),
            mixture_gaussian=(
                kwargs.get("mixture_gaussian")
                if isinstance(kwargs.get("mixture_gaussian"), MixtureGaussianConfig)
                else (
                    MixtureGaussianConfig(**kwargs["mixture_gaussian"])
                    if isinstance(kwargs.get("mixture_gaussian"), dict)
                    else None
                )
            ),
            react_energy=(
                kwargs.get("react_energy")
                if isinstance(kwargs.get("react_energy"), ReActConfig)
                else (
                    ReActConfig(**kwargs["react_energy"])
                    if isinstance(kwargs.get("react_energy"), dict)
                    else None
                )
            ),
        )

    @staticmethod
    def _resolve_strategy_list(novelty_strategy: str) -> list[str]:
        strategy = novelty_strategy.lower()
        if strategy == "confidence":
            return ["confidence"]
        if strategy in {"knn", "knn_distance", "distance"}:
            return ["confidence", "knn_distance"]
        return ["confidence", "knn_distance", "clustering"]

    @staticmethod
    def _coerce_detection_config(
        detection_config: DetectionConfig | dict[str, Any] | None,
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

        strategies = NovelEntityMatcher._resolve_strategy_list(novelty_strategy)

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
        training_data: list[dict] | None = None,
        mode: str | None = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> NovelEntityMatcher:
        self.matcher.fit(
            training_data=training_data,
            mode=mode,
            show_progress=show_progress,
            **kwargs,
        )
        return self

    async def fit_async(
        self,
        training_data: list[dict] | None = None,
        mode: str | None = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> NovelEntityMatcher:
        await self.matcher.fit_async(
            training_data=training_data,
            mode=mode,
            show_progress=show_progress,
            **kwargs,
        )
        return self

    def set_threshold(self, threshold: float) -> NovelEntityMatcher:
        self.acceptance_threshold = threshold
        self.matcher.set_threshold(threshold)
        return self

    def adjust_threshold(self, new_threshold: float) -> None:
        self.set_threshold(new_threshold)

    def get_reference_corpus(self) -> dict[str, Any]:
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

    def add_entities(self, new_entities: list[dict[str, Any]]) -> None:
        self.matcher.add_entities(new_entities)
        self.entities = self.matcher.entities
        self.detector.reset()

    def _collect_top_k(self) -> int:
        if self._pipeline_config is not None:
            return self._pipeline_config.top_k
        return self.detection_config.candidate_top_k

    async def _collect_match_result_async(
        self, queries: list[str]
    ) -> tuple[MatchResultWithMetadata, dict[str, Any]]:
        return await collect_match_result_async(
            self.matcher,
            queries,
            top_k=self._collect_top_k(),
        )

    def _collect_match_result_sync(
        self, queries: list[str]
    ) -> tuple[MatchResultWithMetadata, dict[str, Any]]:
        return collect_match_result_sync(
            self.matcher,
            queries,
            top_k=self._collect_top_k(),
        )

    def _build_orchestrator(
        self,
        *,
        existing_classes: list[str] | None = None,
        context: str | None = None,
        run_llm_proposal: bool | None = None,
    ) -> PipelineOrchestrator:
        if self._pipeline_config is not None:
            builder = PipelineBuilder.from_pipeline_config(
                self._pipeline_config,
                collect_sync=self._collect_match_result_sync,
                collect_async=self._collect_match_result_async,
                detector=self.detector,
                clusterer=self.clusterer,
                llm_proposer=self.llm_proposer,
                existing_classes_resolver=lambda: self._derive_existing_classes(
                    existing_classes
                ),
            )
            return builder.build(
                existing_classes=existing_classes,
                context=context,
                run_llm_proposal=run_llm_proposal,
            )

        clustering_cfg = self.detection_config.clustering
        stage_config = PipelineStageConfig(
            collect_sync=self._collect_match_result_sync,
            collect_async=self._collect_match_result_async,
            detector=self.detector,
            clusterer=self.clusterer,
            llm_proposer=self.llm_proposer,
            use_novelty_detector=self.use_novelty_detector,
            clustering_enabled=True,
            clustering_backend=getattr(self.clusterer, "backend", "auto"),
            similarity_threshold=0.75,
            min_cluster_size=(clustering_cfg.min_cluster_size if clustering_cfg else 5),
            evidence_enabled=True,
            use_tfidf=True,
            run_llm_proposal=run_llm_proposal if run_llm_proposal is not None else True,
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

    def _build_single_match_response(
        self,
        text: str,
        match_result: MatchResultWithMetadata,
        reference_corpus: dict[str, Any],
        return_alternatives: bool = False,
        existing_classes: list[str] | None = None,
    ) -> NovelEntityMatchResult:
        return build_novel_match_result(
            query=text,
            match_result=match_result,
            reference_corpus=reference_corpus,
            detector=self.detector,
            use_novelty_detector=self.use_novelty_detector,
            acceptance_threshold=self.acceptance_threshold,
            return_alternatives=return_alternatives,
            existing_classes=existing_classes,
        )

    def match(
        self,
        text: str,
        return_alternatives: bool = False,
        existing_classes: list[str] | None = None,
    ) -> NovelEntityMatchResult:
        match_result, reference_corpus = self._collect_match_result_sync([text])
        return self._build_single_match_response(
            text, match_result, reference_corpus, return_alternatives, existing_classes
        )

    async def match_async(
        self,
        text: str,
        return_alternatives: bool = False,
        existing_classes: list[str] | None = None,
    ) -> NovelEntityMatchResult:
        match_result, reference_corpus = await self._collect_match_result_async([text])
        return self._build_single_match_response(
            text, match_result, reference_corpus, return_alternatives, existing_classes
        )

    def match_batch(
        self,
        texts: list[str],
        return_alternatives: bool = False,
        existing_classes: list[str] | None = None,
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
                existing_classes=existing_classes,
            )
            for idx, text in enumerate(texts)
        ]

    async def discover_novel_classes(
        self,
        queries: list[str],
        existing_classes: list[str] | None = None,
        context: str | None = None,
        return_metadata: bool = True,
        run_llm_proposal: bool | None = None,
    ) -> NovelClassDiscoveryReport:
        effective_run_llm = run_llm_proposal if run_llm_proposal is not None else True
        pipeline = self._build_orchestrator(
            existing_classes=existing_classes,
            context=context,
            run_llm_proposal=effective_run_llm,
        )
        ctx = StageContext(inputs=list(queries))

        if return_metadata:
            pipeline_result = await pipeline.run_async(ctx)
        else:
            pipeline_result = pipeline.run(ctx)

        report = self._build_discovery_report(
            pipeline_result=pipeline_result,
            detection_config_dump=self.detection_config.model_dump(),
            existing_classes=existing_classes,
            context=context,
        )

        if self._should_create_review_records():
            report.review_records = self.review_manager.create_records(report)

        return self._finalize_report(report)

    async def discover(
        self,
        queries: list[str],
        *,
        existing_classes: list[str] | None = None,
        context: str | None = None,
        return_metadata: bool = True,
        run_llm_proposal: bool | None = None,
    ) -> NovelClassDiscoveryReport:
        return await self.discover_novel_classes(
            queries=queries,
            existing_classes=existing_classes,
            context=context,
            return_metadata=return_metadata,
            run_llm_proposal=run_llm_proposal,
        )

    def _should_create_review_records(self) -> bool:
        if self._pipeline_config is not None:
            return self._pipeline_config.auto_create_review_records
        return True

    def batch_discover(
        self,
        queries_batch: list[list[str]],
        existing_classes: list[str] | None = None,
        context: str | None = None,
    ) -> list[NovelClassDiscoveryReport]:
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

    def approve_proposal(
        self, review_id: str, *, notes: str | None = None
    ) -> ProposalReviewRecord:
        return self.review_manager.update_state(review_id, "approved", notes=notes)

    def reject_proposal(
        self, review_id: str, *, notes: str | None = None
    ) -> ProposalReviewRecord:
        return self.review_manager.update_state(review_id, "rejected", notes=notes)

    def promote_proposal(
        self,
        review_id: str,
        *,
        promoter: Callable[[ProposalReviewRecord], Any] | None = None,
    ) -> PromotionResult:
        effective_promoter = promoter or self._default_promoter
        return self.review_manager.promote(review_id, promoter=effective_promoter)

    def _default_promoter(self, record: ProposalReviewRecord) -> None:
        proposal = getattr(record, "proposal", None)
        if proposal is None:
            return
        proposed_name = getattr(proposal, "name", None)
        if proposed_name is None:
            return
        new_entity = {"id": proposed_name, "name": proposed_name}
        if hasattr(self.matcher, "add_entity"):
            self.matcher.add_entity(new_entity)
        elif hasattr(self.matcher, "entities"):
            self.matcher.entities.append(new_entity)
        self.entities.append(new_entity)

    def list_review_records(
        self, discovery_id: str | None = None
    ) -> list[ProposalReviewRecord]:
        return self.review_manager.list_records(discovery_id)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        entities: list[dict[str, Any]] | None = None,
        matcher: Matcher | None = None,
        **overrides: Any,
    ) -> NovelEntityMatcher:
        from ..config import Config

        config = Config(config_path)
        matcher_kwargs: dict[str, Any] = {
            "model": config.get("embedding.model"),
            "acceptance_threshold": config.get("embedding.threshold"),
        }
        matcher_kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(entities=entities, matcher=matcher, **matcher_kwargs)


def create_novel_entity_matcher(
    entities: list[dict[str, Any]],
    model: str = "potion-32m",
    mode: str = "zero-shot",
    threshold: float = 0.5,
    enable_novelty_detection: bool = True,
    **kwargs: Any,
) -> NovelEntityMatcher:
    return NovelEntityMatcher(
        entities=entities,
        model=model,
        mode=mode,
        acceptance_threshold=threshold,
        use_novelty_detector=enable_novelty_detection,
        **kwargs,
    )
