"""Shared helpers for novelty-aware match and discovery orchestration."""

from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

from .match_result import MatchResultWithMetadata


async def collect_match_result_async(
    matcher: Any,
    queries: List[str],
    top_k: int = 5,
) -> tuple[MatchResultWithMetadata, Dict[str, Any]]:
    """Async helper to collect match result and reference corpus.

    Consolidates duplicated logic from NovelEntityMatcher and DiscoveryPipeline.

    Args:
        matcher: Matcher instance
        queries: List of query texts
        top_k: Number of top candidates to retrieve

    Returns:
        Tuple of (match_result, reference_corpus)
    """
    match_async = getattr(matcher, "match_async", None)
    if callable(match_async):
        result = await match_async(
            queries,
            return_metadata=True,
            top_k=top_k,
        )
    else:
        result = await asyncio.to_thread(
            matcher.match,
            queries,
            return_metadata=True,
            top_k=top_k,
        )

    return result, matcher.get_reference_corpus()


def collect_match_result_sync(
    matcher: Any,
    queries: List[str],
    top_k: int = 5,
) -> tuple[MatchResultWithMetadata, Dict[str, Any]]:
    """Sync helper to collect match result and reference corpus.

    Consolidates duplicated logic from NovelEntityMatcher and DiscoveryPipeline.

    Args:
        matcher: Matcher instance
        queries: List of query texts
        top_k: Number of top candidates to retrieve

    Returns:
        Tuple of (match_result, reference_corpus)
    """
    result = matcher.match(
        queries,
        return_metadata=True,
        top_k=top_k,
    )
    return result, matcher.get_reference_corpus()


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
    existing_classes: Optional[List[str]] = None,
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

    outside_existing = (
        existing_classes is not None
        and predicted_id not in (None, "unknown")
        and predicted_id not in existing_classes
    )

    if outside_existing:
        accepted_known = False
        is_novel = True
        match_method = "outside_existing_classes"
    elif accepted_known:
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


def export_pipeline_metrics(
    *,
    metrics: Dict[str, Any],
    format: str = "json",
    path: Optional[str] = None,
) -> Path:
    """Export pipeline metrics to file.

    Args:
        metrics: Key-value pairs of metric data.
        format: Export format ('json' or 'csv').
        path: Output file path (default: './metrics_{timestamp}.{ext}').

    Returns:
        Path to exported metrics file.

    Raises:
        ValueError: If format is not 'json' or 'csv'.
    """
    if format not in ("json", "csv"):
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"metrics_{timestamp}.{format}"

    output_path = Path(path)

    if format == "json":
        data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in metrics.items():
                writer.writerow([key, value])

    return output_path
