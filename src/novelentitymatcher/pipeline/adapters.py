"""Adapters that route existing matcher and discovery capabilities through stages."""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import numpy as np

from ..novelty.extraction import ClusterEvidenceExtractor
from ..novelty.schemas import ClusterEvidence, DiscoveryCluster, NovelSampleReport
from .contracts import PipelineStage, StageContext, StageResult

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


class MatcherMetadataStage(PipelineStage):
    """Collect rich matcher metadata and reference corpus for downstream stages."""

    name = "match"

    def __init__(
        self,
        collect_sync: Callable[[list[str]], tuple[Any, dict[Any, Any]]] | None,
        collect_async: Callable[[list[str]], Awaitable[tuple[Any, dict[Any, Any]]]]
        | None,
    ):
        if collect_sync is None or collect_async is None:
            raise ValueError("collect_sync and collect_async must be provided")
        self._collect_sync = collect_sync
        self._collect_async = collect_async

    def run(self, context: StageContext) -> StageResult:
        match_result, reference = self._collect_sync(context.inputs)
        match_method = (match_result.metadata or {}).get("match_method")
        return StageResult(
            stage_name=self.name,
            artifacts={
                "match_result": match_result,
                "reference_corpus": reference,
            },
            metadata={
                "num_queries": len(context.inputs),
                "candidate_top_k": (match_result.metadata or {}).get("top_k"),
                "match_method": match_method,
            },
            stage_config_snapshot={"top_k": (match_result.metadata or {}).get("top_k")},
        )

    async def run_async(self, context: StageContext) -> StageResult:
        match_result, reference = await self._collect_async(context.inputs)
        match_method = (match_result.metadata or {}).get("match_method")
        return StageResult(
            stage_name=self.name,
            artifacts={
                "match_result": match_result,
                "reference_corpus": reference,
            },
            metadata={
                "num_queries": len(context.inputs),
                "candidate_top_k": (match_result.metadata or {}).get("top_k"),
                "match_method": match_method,
            },
            stage_config_snapshot={"top_k": (match_result.metadata or {}).get("top_k")},
        )


class OODDetectionStage(PipelineStage):
    """Run novelty detection against the stable matcher metadata contract."""

    name = "ood"

    def __init__(
        self,
        detector: Any,
        enabled: bool = True,
        ood_strategies: list[str] | None = None,
        ood_calibration_mode: str = "none",
        ood_calibration_alpha: float = 0.1,
        ood_mahalanobis_mode: str = "class_conditional",
    ):
        self.detector = detector
        self.enabled = enabled
        self.ood_strategies = ood_strategies
        self.ood_calibration_mode = ood_calibration_mode
        self.ood_calibration_alpha = ood_calibration_alpha
        self.ood_mahalanobis_mode = ood_mahalanobis_mode

    def run(self, context: StageContext) -> StageResult:
        if not self.enabled:
            report = NovelSampleReport()
        else:
            match_result = context.artifacts["match_result"]
            reference = context.artifacts["reference_corpus"]
            report = self.detector.detect_novel_samples(
                texts=context.inputs,
                confidences=match_result.confidences,
                embeddings=match_result.embeddings,
                predicted_classes=match_result.predictions,
                candidate_results=match_result.candidate_results,
                reference_embeddings=reference["embeddings"],
                reference_labels=reference["labels"],
            )

        return StageResult(
            stage_name=self.name,
            artifacts={"novel_sample_report": report},
            metadata={
                "num_novel_samples": len(report.novel_samples),
                "strategies": list(getattr(report, "detection_strategies", [])),
                "ood_calibration_mode": self.ood_calibration_mode,
                "ood_calibration_alpha": self.ood_calibration_alpha,
                "ood_mahalanobis_mode": self.ood_mahalanobis_mode,
            },
            stage_config_snapshot={
                "enabled": self.enabled,
                "ood_strategies": self.ood_strategies,
                "ood_calibration_mode": self.ood_calibration_mode,
                "ood_calibration_alpha": self.ood_calibration_alpha,
                "ood_mahalanobis_mode": self.ood_mahalanobis_mode,
            },
        )


class CommunityDetectionStage(PipelineStage):
    """Cluster likely novel samples into discovery communities."""

    name = "cluster"

    def __init__(
        self,
        clusterer: Any,
        *,
        enabled: bool = True,
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 2,
        clustering_metric: str = "cosine",
    ):
        self.clusterer = clusterer
        self.enabled = enabled
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.clustering_metric = clustering_metric

    def run(self, context: StageContext) -> StageResult:
        report = context.artifacts["novel_sample_report"]
        match_result = context.artifacts["match_result"]
        novel_samples = list(report.novel_samples)

        if not self.enabled or not novel_samples:
            return StageResult(
                stage_name=self.name,
                artifacts={"discovery_clusters": []},
                metadata={"num_clusters": 0, "backend": "disabled"},
            )

        sample_indices = [sample.index for sample in novel_samples]
        embeddings = np.asarray(
            [match_result.embeddings[idx] for idx in sample_indices]
        )

        cluster_labels, backend = self._cluster_embeddings(embeddings)
        index_to_cluster: dict[int, int] = {}
        groups: dict[int, list[Any]] = defaultdict(list)
        for sample, cluster_id in zip(novel_samples, cluster_labels, strict=False):
            sample.cluster_id = cluster_id if cluster_id >= 0 else None
            if cluster_id >= 0:
                groups[int(cluster_id)].append(sample)
                index_to_cluster[sample.index] = int(cluster_id)

        clusters = [
            self._build_cluster(cluster_id, samples)
            for cluster_id, samples in sorted(groups.items())
            if len(samples) >= self.min_cluster_size
        ]
        if not clusters and novel_samples:
            for sample in novel_samples:
                sample.cluster_id = 0
                index_to_cluster[sample.index] = 0
            clusters = [self._build_cluster(0, novel_samples)]

        return StageResult(
            stage_name=self.name,
            artifacts={
                "discovery_clusters": clusters,
                "cluster_assignments": index_to_cluster,
            },
            metadata={
                "num_clusters": len(clusters),
                "backend": backend,
                "cluster_sizes": [cluster.sample_count for cluster in clusters],
            },
        )

    def _cluster_embeddings(self, embeddings: np.ndarray) -> tuple[list[int], str]:
        if len(embeddings) == 1:
            return [0], "singleton"

        try:
            from ..novelty.clustering.scalable import ScalableClusterer

            if self.clusterer is None:
                raise ValueError(
                    "CommunityDetectionStage requires a clusterer instance. "
                    "Inject ScalableClusterer through PipelineBuilder."
                )

            if not isinstance(self.clusterer, ScalableClusterer):
                logger.warning(
                    "Expected ScalableClusterer but got %s",
                    type(self.clusterer).__name__,
                )

            labels, _, info = self.clusterer.fit_predict(
                embeddings,
                metric=self.clustering_metric,
            )
            backend_name = str(
                info.get("backend", getattr(self.clusterer, "backend", "unknown"))
            )
            return [int(label) for label in labels], backend_name
        except (ImportError, ValueError, TypeError, RuntimeError):
            logger.warning(
                "Clusterer execution failed, falling back to connected components"
            )

        labels = self._fallback_connected_components(embeddings)
        return labels, "fallback_connected_components"

    def _fallback_connected_components(self, embeddings: np.ndarray) -> list[int]:
        normalized = embeddings / np.clip(
            np.linalg.norm(embeddings, axis=1, keepdims=True),
            a_min=1e-12,
            a_max=None,
        )
        similarity = normalized @ normalized.T
        labels = [-1] * len(embeddings)
        cluster_id = 0

        for start in range(len(embeddings)):
            if labels[start] != -1:
                continue
            stack = [start]
            component: list[int] = []
            labels[start] = cluster_id
            while stack:
                current = stack.pop()
                component.append(current)
                neighbors = np.where(similarity[current] >= self.similarity_threshold)[
                    0
                ]
                for neighbor in neighbors:
                    if labels[int(neighbor)] == -1:
                        labels[int(neighbor)] = cluster_id
                        stack.append(int(neighbor))

            if len(component) < self.min_cluster_size:
                for member in component:
                    labels[member] = -1
            else:
                cluster_id += 1

        return labels

    def _build_cluster(
        self, cluster_id: int, samples: Iterable[Any]
    ) -> DiscoveryCluster:
        sample_list = list(samples)
        novelty_scores = [
            float(sample.novelty_score or 0.0)
            for sample in sample_list
            if sample.novelty_score is not None
        ]
        confidences = [float(sample.confidence) for sample in sample_list]
        return DiscoveryCluster(
            cluster_id=cluster_id,
            sample_indices=[sample.index for sample in sample_list],
            sample_count=len(sample_list),
            example_texts=[sample.text for sample in sample_list[:5]],
            mean_novelty_score=sum(novelty_scores) / len(novelty_scores)
            if novelty_scores
            else None,
            mean_confidence=sum(confidences) / len(confidences)
            if confidences
            else None,
            metadata={
                "predicted_classes": [sample.predicted_class for sample in sample_list],
            },
        )


class ClusterEvidenceStage(PipelineStage):
    """Extract compact evidence from clusters before proposal generation."""

    name = "evidence"

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_keywords: int = 8,
        max_examples: int = 4,
        token_budget: int = 256,
        use_rake: bool = True,
        evidence_method: str = "tfidf",
        use_tfidf: bool | None = None,
    ):
        self.enabled = enabled
        self.max_keywords = max_keywords
        self.max_examples = max_examples
        self.token_budget = token_budget
        self.use_rake = use_rake
        if use_tfidf is not None:
            evidence_method = "tfidf" if use_tfidf else "centroid"
        self.evidence_method = evidence_method
        self.use_tfidf = self.evidence_method == "tfidf"

    def run(self, context: StageContext) -> StageResult:
        clusters: list[DiscoveryCluster] = list(
            context.artifacts.get("discovery_clusters", [])
        )
        report = context.artifacts["novel_sample_report"]
        samples_by_index = {sample.index: sample for sample in report.novel_samples}

        if not self.enabled or not clusters:
            return StageResult(
                stage_name=self.name,
                artifacts={"discovery_clusters": clusters},
                metadata={"num_clusters_with_evidence": 0},
            )

        enriched: list[DiscoveryCluster] = []
        for cluster in clusters:
            samples = [
                samples_by_index[index]
                for index in cluster.sample_indices
                if index in samples_by_index
            ]
            evidence = self._build_evidence(samples, context)
            cluster.keywords = list(evidence.keywords)
            cluster.evidence = evidence
            enriched.append(cluster)

        return StageResult(
            stage_name=self.name,
            artifacts={"discovery_clusters": enriched},
            metadata={
                "num_clusters_with_evidence": len(enriched),
                "token_budget": self.token_budget,
                "evidence_method": self.evidence_method,
            },
            stage_config_snapshot={
                "evidence_method": self.evidence_method,
                "use_tfidf": self.use_tfidf,
                "max_keywords": self.max_keywords,
                "max_examples": self.max_examples,
                "token_budget": self.token_budget,
            },
        )

    def _build_evidence(
        self, samples: list[Any], context: StageContext | None = None
    ) -> ClusterEvidence:
        texts = [sample.text for sample in samples]
        predicted_classes = [str(sample.predicted_class) for sample in samples]
        confidences = [float(sample.confidence) for sample in samples]
        novelty_scores = [
            float(sample.novelty_score or 0.0)
            for sample in samples
            if sample.novelty_score is not None
        ]

        sample_indices = [sample.index for sample in samples]
        cluster_embeddings = None
        if context is not None:
            match_result = context.artifacts.get("match_result")
            if match_result is not None and hasattr(match_result, "embeddings"):
                try:
                    cluster_embeddings = np.asarray(
                        [
                            match_result.embeddings[idx]
                            for idx in sample_indices
                            if idx < len(match_result.embeddings)
                        ]
                    )
                except (IndexError, TypeError):
                    cluster_embeddings = None

        extractor = ClusterEvidenceExtractor(
            method=self.evidence_method,
            max_keywords=self.max_keywords,
            max_examples=self.max_examples,
            token_budget=self.token_budget,
        )
        base_evidence = extractor.extract(
            cluster_texts=texts,
            cluster_embeddings=cluster_embeddings,
        )
        keywords = list(base_evidence.keywords)
        rake_keywords: list[str] = []
        if self.use_rake:
            rake_keywords = self._extract_rake_keywords(texts)

        return ClusterEvidence(
            keywords=keywords,
            rake_keywords=rake_keywords,
            representative_examples=list(base_evidence.representative_examples),
            sample_indices=sample_indices,
            predicted_classes=sorted(set(predicted_classes)),
            confidence_summary={
                "mean_confidence": sum(confidences) / len(confidences)
                if confidences
                else 0.0,
                "mean_novelty_score": (
                    sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
                ),
            },
            token_budget=self.token_budget,
            metadata={
                "sample_count": len(samples),
                "evidence_method": self.evidence_method,
            },
        )

    def _extract_rake_keywords(self, texts: list[str]) -> list[str]:
        joined_text = " ".join(texts)
        sentences = re.split(r"[.!?;]+", joined_text)

        word_freq: Counter[str] = Counter()
        word_degree: dict[str, float] = defaultdict(float)

        for sentence in sentences:
            words = self._tokenize(sentence)
            if not words:
                continue
            for word in words:
                word_freq[word] += 1
            degree = len(words) - 1
            for word in words:
                word_degree[word] += degree

        candidates: list[str] = []
        for sentence in sentences:
            words = self._tokenize(sentence)
            if len(words) < 2:
                continue
            for size in range(2, 4):
                for i in range(len(words) - size + 1):
                    phrase = " ".join(words[i : i + size])
                    candidates.append(phrase)

        phrase_scores: dict[str, float] = {}
        for phrase in candidates:
            words = phrase.split()
            score = sum(word_degree.get(w, 0) for w in words) / max(
                sum(word_freq.get(w, 1) for w in words), 1
            )
            phrase_scores[phrase] = phrase_scores.get(phrase, 0) + score

        ranked = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        seen: set[str] = set()
        results: list[str] = []
        for phrase, _ in ranked:
            normalized = phrase.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                results.append(phrase)
                if len(results) >= self.max_keywords:
                    break
        return results

    def _tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(token) > 2 and token not in _STOPWORDS
        ]


class ProposalStage(PipelineStage):
    """Optionally generate class proposals from a novelty report."""

    name = "proposal"

    def __init__(
        self,
        proposer: Any,
        existing_classes_resolver: Callable[[], list[str]],
        enabled: bool = True,
        context_text: str | None = None,
        max_retries: int = 2,
        force_cluster_level: bool = True,
        proposal_mode: str = "cluster",
        proposal_schema_discovery: bool = False,
        proposal_schema_max_attributes: int = 10,
        proposal_hierarchical: bool = True,
    ):
        self.proposer = proposer
        self._existing_classes_resolver = existing_classes_resolver
        self.enabled = enabled
        self.context_text = context_text
        self.max_retries = max_retries
        self.force_cluster_level = force_cluster_level
        self.proposal_mode = proposal_mode
        self.proposal_schema_discovery = proposal_schema_discovery
        self.proposal_schema_max_attributes = proposal_schema_max_attributes
        self.proposal_hierarchical = proposal_hierarchical

    def run(self, context: StageContext) -> StageResult:
        report = context.artifacts["novel_sample_report"]
        discovery_clusters = list(context.artifacts.get("discovery_clusters", []))
        class_proposals = None
        existing_classes = self._existing_classes_resolver()
        error = None

        if self.enabled and report.novel_samples:
            try:
                class_proposals = self._generate_proposals(
                    report=report,
                    discovery_clusters=discovery_clusters,
                    existing_classes=existing_classes,
                )
            except (
                ValueError,
                TypeError,
                RuntimeError,
                ConnectionError,
            ) as exc:  # pragma: no cover - defensive wrapper
                error = str(exc)

        return StageResult(
            stage_name=self.name,
            artifacts={"class_proposals": class_proposals},
            metadata={
                "enabled": self.enabled,
                "num_existing_classes": len(existing_classes),
                "generated": class_proposals is not None,
                "num_clusters": len(discovery_clusters),
                "force_cluster_level": self.force_cluster_level,
                "proposal_mode": self.proposal_mode,
                "proposal_schema_discovery": self.proposal_schema_discovery,
                "proposal_schema_max_attributes": self.proposal_schema_max_attributes,
                "error": error,
            },
            stage_config_snapshot={
                "proposal_mode": self.proposal_mode,
                "proposal_schema_discovery": self.proposal_schema_discovery,
                "proposal_schema_max_attributes": self.proposal_schema_max_attributes,
                "max_retries": self.max_retries,
                "proposal_hierarchical": self.proposal_hierarchical,
            },
        )

    def _generate_proposals(
        self,
        *,
        report: Any,
        discovery_clusters: list[Any],
        existing_classes: list[str],
    ) -> Any:
        use_cluster_mode = self.proposal_mode in {"cluster", "rag_cluster"} and (
            discovery_clusters and self.force_cluster_level
        )
        use_cluster_fallback = (
            self.proposal_mode in {"cluster", "rag_cluster"}
            and discovery_clusters
            and hasattr(self.proposer, "propose_from_clusters")
        )

        if use_cluster_mode:
            return self._propose_from_clusters(discovery_clusters, existing_classes)

        if self.proposal_mode == "sample":
            return self.proposer.propose_classes(
                novel_samples=report.novel_samples,
                existing_classes=existing_classes,
                context=self.context_text,
            )

        if use_cluster_fallback:
            return self._propose_from_clusters(discovery_clusters, existing_classes)

        if discovery_clusters and self.proposal_mode == "rag_cluster":
            cluster_samples = self._flatten_clusters(discovery_clusters)
            return self.proposer.propose_classes(
                novel_samples=cluster_samples,
                existing_classes=existing_classes,
                context=self.context_text,
            )

        return self.proposer.propose_classes(
            novel_samples=report.novel_samples,
            existing_classes=existing_classes,
            context=self.context_text,
        )

    def _propose_from_clusters(
        self,
        discovery_clusters: list[Any],
        existing_classes: list[str],
    ) -> Any:
        if hasattr(self.proposer, "propose_from_clusters"):
            propose_kwargs: dict[str, Any] = {
                "max_retries": self.max_retries,
                "hierarchical": self.proposal_hierarchical,
            }
            if self.proposal_schema_discovery and hasattr(
                self.proposer, "propose_from_clusters_with_schema"
            ):
                propose_kwargs["max_attributes"] = self.proposal_schema_max_attributes
                return self.proposer.propose_from_clusters_with_schema(
                    discovery_clusters=discovery_clusters,
                    existing_classes=existing_classes,
                    context=self.context_text,
                    **propose_kwargs,
                )

            return self.proposer.propose_from_clusters(
                discovery_clusters=discovery_clusters,
                existing_classes=existing_classes,
                context=self.context_text,
                **propose_kwargs,
            )

        cluster_samples = self._flatten_clusters(discovery_clusters)
        return self.proposer.propose_classes(
            novel_samples=cluster_samples,
            existing_classes=existing_classes,
            context=self.context_text,
        )

    def _flatten_clusters(self, discovery_clusters: list[Any]) -> list[Any]:
        """Convert DiscoveryCluster list into NovelSampleMetadata list."""
        from ..novelty.schemas import NovelSampleMetadata

        flat: list[NovelSampleMetadata] = []
        for cluster in discovery_clusters:
            for i, text in enumerate(cluster.example_texts or []):
                flat.append(
                    NovelSampleMetadata(
                        text=text,
                        index=cluster.sample_indices[i]
                        if i < len(cluster.sample_indices)
                        else len(flat),
                        confidence=cluster.mean_confidence or 0.5,
                        predicted_class=cluster.metadata.get(
                            "predicted_classes", ["unknown"]
                        )[0]
                        if cluster.metadata.get("predicted_classes")
                        else "unknown",
                        cluster_id=cluster.cluster_id,
                    )
                )
            if not cluster.example_texts and cluster.sample_indices:
                for idx in cluster.sample_indices:
                    flat.append(
                        NovelSampleMetadata(
                            text=f"cluster_{cluster.cluster_id}_sample_{idx}",
                            index=idx,
                            confidence=cluster.mean_confidence or 0.5,
                            predicted_class="unknown",
                            cluster_id=cluster.cluster_id,
                        )
                    )
        return flat
