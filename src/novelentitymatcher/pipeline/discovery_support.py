"""Shared helpers for novelty-aware match and discovery orchestration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .match_result import MatchResultWithMetadata
from .pipeline_builder import PipelineStageConfig


def derive_existing_classes(
    *,
    entities: list[dict[str, Any]],
    get_reference_corpus: Callable[[], Dict[str, Any]],
    existing_classes: Optional[List[str]] = None,
) -> List[str]:
    if existing_classes:
        return list(existing_classes)
    if entities:
        return [str(entity["id"]) for entity in entities if "id" in entity]
    reference = get_reference_corpus()
    return list(reference.get("labels", []))


def build_novel_match_result(
    *,
    query: str,
    match_result: MatchResultWithMetadata,
    reference_corpus: Dict[str, Any],
    detector: Any,
    use_novelty_detector: bool,
    acceptance_threshold: float,
    return_alternatives: bool = False,
):
    from ..novelty.entity_matcher import NovelEntityMatchResult

    record = match_result.records[0]
    predicted_id = record.predicted_id if record.predicted_id else None
    score = float(record.confidence)
    alternatives = list(record.candidates)

    if use_novelty_detector:
        report = detector.detect_novel_samples(
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
        and score >= acceptance_threshold
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
            "acceptance_threshold": acceptance_threshold,
        },
    )


def build_stage_config(
    *,
    collect_sync: Callable[[List[str]], tuple[Any, dict]],
    collect_async: Callable[[List[str]], tuple[Any, dict]],
    detector: Any,
    clusterer: Any,
    llm_proposer: Any,
    use_novelty_detector: bool,
    clustering_enabled: bool = True,
    clustering_backend: str = "auto",
    similarity_threshold: float = 0.75,
    min_cluster_size: int = 5,
    evidence_enabled: bool = True,
    max_keywords: int = 8,
    max_examples: int = 4,
    token_budget: int = 256,
    use_tfidf: bool = True,
    run_llm_proposal: bool = True,
    existing_classes_resolver: Optional[Callable[[], List[str]]] = None,
    context_text: Optional[str] = None,
    max_retries: int = 2,
    prefer_cluster_level: bool = True,
) -> PipelineStageConfig:
    return PipelineStageConfig(
        collect_sync=collect_sync,
        collect_async=collect_async,
        detector=detector,
        clusterer=clusterer,
        llm_proposer=llm_proposer,
        use_novelty_detector=use_novelty_detector,
        clustering_enabled=clustering_enabled,
        clustering_backend=clustering_backend,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
        evidence_enabled=evidence_enabled,
        max_keywords=max_keywords,
        max_examples=max_examples,
        token_budget=token_budget,
        use_tfidf=use_tfidf,
        run_llm_proposal=run_llm_proposal,
        existing_classes_resolver=existing_classes_resolver,
        context_text=context_text,
        max_retries=max_retries,
        prefer_cluster_level=prefer_cluster_level,
    )
