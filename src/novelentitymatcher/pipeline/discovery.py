"""Public pipeline-first discovery API."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..core.matcher import Matcher
from ..novelty.clustering.scalable import ScalableClusterer
from ..novelty.config.base import DetectionConfig
from ..novelty.config.strategies import (
    ClusteringConfig,
    ConfidenceConfig,
    KNNConfig,
    MahalanobisConfig,
)
from ..novelty.core.detector import NoveltyDetector
from ..novelty.entity_matcher import NovelEntityMatchResult
from ..novelty.proposal.llm import LLMClassProposer
from ..novelty.schemas import (
    NovelClassDiscoveryReport,
    NovelSampleMetadata,
    NovelSampleReport,
    PromotionResult,
    ProposalReviewRecord,
)
from ..novelty.storage.persistence import export_summary, save_proposals
from ..novelty.storage.review import ProposalReviewManager
from ..utils.logging_config import get_logger
from .config import PipelineConfig
from .contracts import StageContext
from .discovery_support import (
    build_novel_match_result,
    collect_match_result_async,
    collect_match_result_sync,
    derive_existing_classes,
)
from .match_result import MatchResultWithMetadata
from .orchestrator import PipelineOrchestrator
from .pipeline_builder import PipelineBuilder

logger = get_logger(__name__)


class _NovelEntityMatcherCompat:
    """Thin compat shim so code that expects pipeline.novel_entity_matcher.detector still works."""

    def __init__(self, *, detector: NoveltyDetector, llm_proposer: LLMClassProposer):
        self.detector = detector
        self.llm_proposer = llm_proposer


class DiscoveryPipeline:
    """Pipeline-first public entry point for discovery and promotion workflows.

    Owns its own Matcher, NoveltyDetector, ScalableClusterer, and LLMClassProposer
    instances, and routes everything through a PipelineOrchestrator built from
    PipelineConfig.
    """

    def __init__(
        self,
        entities: list[dict[str, Any]] | None = None,
        *,
        matcher: Matcher | None = None,
        review_storage_path: str | Path = "./proposals/review_records.json",
        config: PipelineConfig | None = None,
        **kwargs: Any,
    ):
        self._config = config or PipelineConfig.from_dict(kwargs)
        review_storage_path = str(review_storage_path)

        # Build owned Matcher
        if matcher is not None:
            self.matcher = matcher
        else:
            if entities is None:
                raise ValueError("entities is required when matcher is not provided")
            threshold = kwargs.get(
                "acceptance_threshold",
                kwargs.get("match_threshold", 0.5),
            )
            self.matcher = Matcher(
                entities=entities,
                model=kwargs.get("model", "potion-32m"),
                mode=kwargs.get("mode", "zero-shot"),
                threshold=threshold,
            )

        self.entities = (
            entities
            if entities is not None
            else list(getattr(self.matcher, "entities", []))
        )
        self.acceptance_threshold = kwargs.get(
            "acceptance_threshold",
            kwargs.get("match_threshold", getattr(self.matcher, "threshold", 0.5)),
        )

        # Build owned NoveltyDetector
        detection_config = self._build_detection_config(kwargs)
        self.detector = NoveltyDetector(config=detection_config)
        self.use_novelty_detector = self._config.ood_enabled

        # Build owned ScalableClusterer
        clustering_cfg = detection_config.clustering or ClusteringConfig(
            min_cluster_size=self._config.min_cluster_size,
        )
        self.clusterer = ScalableClusterer(
            backend=self._config.clustering_backend,
            min_cluster_size=clustering_cfg.min_cluster_size,
            min_samples=(
                self._config.clustering_min_samples
                or clustering_cfg.hdbscan_min_samples
            ),
            cluster_selection_epsilon=(
                self._config.clustering_cluster_selection_epsilon
            ),
            umap_metric=self._config.clustering_metric,
        )

        # Build owned LLMClassProposer
        self.llm_proposer = LLMClassProposer(
            primary_model=self._config.llm_model,
            provider=self._config.llm_provider,
            api_keys=kwargs.get("llm_api_keys"),
        )

        # HITL
        self.review_manager = ProposalReviewManager(review_storage_path)
        self.output_dir = self._config.output_dir
        self.auto_save = self._config.auto_save

        # Build orchestrator
        self._orchestrator = self._build_orchestrator()

    # ------------------------------------------------------------------
    # Backward-compatibility: novel_entity_matcher alias
    # ------------------------------------------------------------------

    @property
    def novel_entity_matcher(self) -> Any:
        """Backward-compatible alias exposing detector/llm_proposer as if from NovelEntityMatcher."""
        return _NovelEntityMatcherCompat(
            detector=self.detector,
            llm_proposer=self.llm_proposer,
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def _build_detection_config(self, kwargs: dict[str, Any]) -> DetectionConfig:
        detection_config = kwargs.get("detection_config")
        if isinstance(detection_config, DetectionConfig):
            return detection_config
        if isinstance(detection_config, dict):
            return DetectionConfig(**detection_config)

        novelty_strategy = kwargs.get("novelty_strategy", "knn_distance")
        confidence_threshold = kwargs.get(
            "confidence_threshold", self._config.confidence_threshold
        )
        knn_k = kwargs.get("knn_k", 5)
        knn_distance_threshold = kwargs.get("knn_distance_threshold", 0.6)
        min_cluster_size = kwargs.get("min_cluster_size", self._config.min_cluster_size)

        strategies = list(self._config.ood_strategies)
        if not strategies:
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
            clustering=ClusteringConfig(
                min_cluster_size=min_cluster_size,
                hdbscan_min_samples=(
                    self._config.clustering_min_samples or min_cluster_size
                ),
                cluster_selection_epsilon=(
                    self._config.clustering_cluster_selection_epsilon
                ),
            ),
            mahalanobis=MahalanobisConfig(
                use_class_conditional=(
                    self._config.ood_mahalanobis_mode == "class_conditional"
                ),
                calibration_mode=self._config.ood_calibration_mode,
                calibration_alpha=self._config.ood_calibration_alpha,
                calibration_method=self._config.ood_calibration_method,
            ),
        )

    # ------------------------------------------------------------------
    # Orchestrator construction
    # ------------------------------------------------------------------

    def _build_orchestrator(
        self,
        *,
        existing_classes: list[str] | None = None,
        context: str | None = None,
        run_llm_proposal: bool | None = None,
    ) -> PipelineOrchestrator:
        builder = PipelineBuilder.from_pipeline_config(
            self._config,
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

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _derive_existing_classes(
        self, existing_classes: list[str] | None = None
    ) -> list[str]:
        return derive_existing_classes(
            entities=self.entities,
            get_reference_corpus=self.get_reference_corpus,
            existing_classes=existing_classes,
        )

    def get_reference_corpus(self) -> dict[str, Any]:
        return self.matcher.get_reference_corpus()

    async def _collect_match_result_async(
        self, queries: list[str]
    ) -> tuple[MatchResultWithMetadata, dict[str, Any]]:
        return await collect_match_result_async(
            self.matcher,
            queries,
            top_k=self._config.top_k,
        )

    def _collect_match_result_sync(
        self, queries: list[str]
    ) -> tuple[MatchResultWithMetadata, dict[str, Any]]:
        return collect_match_result_sync(
            self.matcher,
            queries,
            top_k=self._config.top_k,
        )

    # ------------------------------------------------------------------
    # Public API: fit
    # ------------------------------------------------------------------

    def fit(self, *args: Any, **kwargs: Any) -> DiscoveryPipeline:
        self.matcher.fit(*args, **kwargs)
        return self

    async def fit_async(self, *args: Any, **kwargs: Any) -> DiscoveryPipeline:
        await self.matcher.fit_async(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Public API: match
    # ------------------------------------------------------------------

    def match(
        self,
        text: str,
        return_alternatives: bool = False,
        existing_classes: list[str] | None = None,
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
            existing_classes=existing_classes,
        )

    async def match_async(
        self,
        text: str,
        return_alternatives: bool = False,
        existing_classes: list[str] | None = None,
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
            existing_classes=existing_classes,
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

    # ------------------------------------------------------------------
    # Public API: discover
    # ------------------------------------------------------------------

    async def discover(
        self,
        queries: list[str],
        *,
        existing_classes: list[str] | None = None,
        context: str | None = None,
        return_metadata: bool = True,
        run_llm_proposal: bool | None = None,
    ) -> NovelClassDiscoveryReport:
        """Run the full pipeline: match -> OOD -> cluster -> evidence -> propose."""
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
        ctx = StageContext(inputs=list(queries))

        if return_metadata:
            pipeline_result = await pipeline.run_async(ctx)
        else:
            pipeline_result = pipeline.run(ctx)

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
            detection_config=self.detector.config.model_dump(),
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

        if self._config.auto_create_review_records:
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

    def _get_matcher_config(self) -> dict[str, Any]:
        config = {
            "matcher_type": self.matcher.__class__.__name__,
        }
        if hasattr(self.matcher, "model_name"):
            config["model"] = str(self.matcher.model_name)
        if hasattr(self.matcher, "threshold"):
            config["threshold"] = str(self.matcher.threshold)
        if hasattr(self.matcher, "_training_mode"):
            config["mode"] = self.matcher._training_mode
        return config

    # ------------------------------------------------------------------
    # Public API: HITL
    # ------------------------------------------------------------------

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
        """Promote a review record, optionally invoking a promoter callback.

        If no promoter is provided, a default promoter updates the matcher's
        known entities from the proposal.
        """
        effective_promoter = promoter or self._default_promoter
        return self.review_manager.promote(review_id, promoter=effective_promoter)

    def _default_promoter(self, record: ProposalReviewRecord) -> None:
        """Default promoter: add the proposed class to the matcher's known entities."""
        proposal = getattr(record, "proposal", None)
        if proposal is None:
            return
        proposed_name = getattr(proposal, "name", None)
        if proposed_name is None:
            return
        if hasattr(self.matcher, "add_entity"):
            self.matcher.add_entity({"id": proposed_name, "name": proposed_name})
        elif hasattr(self.matcher, "entities"):
            self.matcher.entities.append({"id": proposed_name, "name": proposed_name})

    def list_review_records(
        self, discovery_id: str | None = None
    ) -> list[ProposalReviewRecord]:
        return self.review_manager.list_records(discovery_id)

    def export_metrics(
        self,
        format: str = "json",
        path: str | None = None,
    ) -> Path:
        """Export collected metrics to file.

        Args:
            format: Export format ('json' or 'csv')
            path: Output file path (default: './metrics_{timestamp}.{ext}')

        Returns:
            Path to exported metrics file

        Raises:
            ValueError: If format is not 'json' or 'csv'
        """
        from .discovery_support import export_pipeline_metrics

        metrics: dict[str, Any] = {
            "num_entities": len(self.entities),
            "acceptance_threshold": self.acceptance_threshold,
            "use_novelty_detector": bool(self._config.ood_enabled),
            "auto_save": bool(self._config.auto_save),
        }
        if hasattr(self.matcher, "model_name"):
            metrics["model_name"] = str(self.matcher.model_name)

        return export_pipeline_metrics(metrics=metrics, format=format, path=path)

    # ------------------------------------------------------------------
    # Backward-compatibility classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        entities: list[dict[str, Any]] | None = None,
        matcher: Matcher | None = None,
        **overrides: Any,
    ) -> DiscoveryPipeline:
        config = Config(config_path)
        matcher_kwargs: dict[str, Any] = {
            "model": config.get("embedding.model"),
            "acceptance_threshold": config.get("embedding.threshold"),
        }
        matcher_kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(entities=entities, matcher=matcher, **matcher_kwargs)
