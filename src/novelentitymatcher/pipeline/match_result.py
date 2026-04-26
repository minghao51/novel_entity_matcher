"""
Stable matcher metadata contracts used by novelty and pipeline internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class MatchRecord:
    """Normalized per-query match metadata for downstream discovery stages."""

    text: str
    predicted_id: str
    confidence: float
    embedding: np.ndarray
    candidates: List[Any] = field(default_factory=list)
    raw_result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    match_method: Optional[str] = None
    reference_embedding: Optional[np.ndarray] = None
    distance: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)
        if self.reference_embedding is not None and not isinstance(
            self.reference_embedding, np.ndarray
        ):
            self.reference_embedding = np.array(self.reference_embedding)


@dataclass
class MatchResultWithMetadata:
    """
    Enhanced match result with stable downstream metadata.

    The legacy attributes (`predictions`, `confidences`, `embeddings`, `metadata`)
    remain available, while `candidate_results` and `records` provide a consistent
    contract for novelty and pipeline stages.
    """

    predictions: List[str]
    confidences: np.ndarray
    embeddings: np.ndarray
    scores: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    candidate_results: List[List[Any]] = field(default_factory=list)
    records: List[MatchRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.confidences, np.ndarray):
            self.confidences = np.array(self.confidences)
        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = np.array(self.embeddings)
        if self.scores is not None and not isinstance(self.scores, np.ndarray):
            self.scores = np.array(self.scores)

        if self.metadata is None:
            self.metadata = {}

        if not self.candidate_results and self.metadata is not None:
            raw_match_results = self.metadata.get("raw_match_results")
            if raw_match_results is not None:
                self.candidate_results = normalize_candidate_results(
                    raw_match_results,
                    len(self.predictions),
                )

        if not self.records:
            texts = list(self.metadata.get("texts", [])) if self.metadata else []
            self.records = build_match_records(
                texts=texts,
                predictions=self.predictions,
                confidences=self.confidences,
                embeddings=self.embeddings,
                candidate_results=self.candidate_results,
            )

    @property
    def num_samples(self) -> int:
        return len(self.predictions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictions": self.predictions,
            "confidences": self.confidences.tolist(),
            "embeddings": self.embeddings.tolist(),
            "scores": self.scores.tolist() if self.scores is not None else None,
            "metadata": self.metadata,
            "candidate_results": self.candidate_results,
            "records": [
                {
                    "text": record.text,
                    "predicted_id": record.predicted_id,
                    "confidence": record.confidence,
                    "embedding": record.embedding.tolist(),
                    "candidates": record.candidates,
                    "raw_result": record.raw_result,
                    "metadata": record.metadata,
                    "match_method": record.match_method,
                    "reference_embedding": (
                        record.reference_embedding.tolist()
                        if record.reference_embedding is not None
                        else None
                    ),
                    "distance": record.distance,
                }
                for record in self.records
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchResultWithMetadata":
        return cls(
            predictions=data["predictions"],
            confidences=np.array(data["confidences"]),
            embeddings=np.array(data["embeddings"]),
            scores=np.array(data["scores"]) if data.get("scores") is not None else None,
            metadata=data.get("metadata"),
            candidate_results=data.get("candidate_results", []),
            records=[
                MatchRecord(
                    text=record["text"],
                    predicted_id=record["predicted_id"],
                    confidence=record["confidence"],
                    embedding=np.array(record["embedding"]),
                    candidates=record.get("candidates", []),
                    raw_result=record.get("raw_result"),
                    metadata=record.get("metadata", {}),
                    match_method=record.get("match_method"),
                    reference_embedding=(
                        np.array(record["reference_embedding"])
                        if record.get("reference_embedding") is not None
                        else None
                    ),
                    distance=record.get("distance"),
                )
                for record in data.get("records", [])
            ],
        )


def normalize_candidate_results(
    raw_match_results: Any, num_queries: int
) -> List[List[Any]]:
    """Normalize raw matcher outputs into a stable list-of-lists shape."""
    if raw_match_results is None:
        return [[] for _ in range(num_queries)]

    if num_queries == 1:
        if isinstance(raw_match_results, list):
            if raw_match_results and all(
                isinstance(item, dict) for item in raw_match_results
            ):
                return [raw_match_results]
            if len(raw_match_results) == 1 and isinstance(raw_match_results[0], list):
                return [list(raw_match_results[0])]
            if len(raw_match_results) == 1:
                first = raw_match_results[0]
                return [[first] if first is not None else []]
        return [[raw_match_results] if raw_match_results is not None else []]

    if isinstance(raw_match_results, list):
        normalized: List[List[Any]] = []
        for result in raw_match_results:
            if result is None:
                normalized.append([])
            elif isinstance(result, list):
                normalized.append(list(result))
            else:
                normalized.append([result])
        return normalized

    return [[raw_match_results] for _ in range(num_queries)]


def build_match_records(
    texts: Sequence[str],
    predictions: Sequence[str],
    confidences: np.ndarray,
    embeddings: np.ndarray,
    candidate_results: Sequence[Sequence[Any]],
    match_method: Optional[str] = None,
    reference_embeddings: Optional[np.ndarray] = None,
) -> List[MatchRecord]:
    """Build normalized per-query records for downstream pipeline stages."""
    records: List[MatchRecord] = []
    for idx, prediction in enumerate(predictions):
        text = texts[idx] if idx < len(texts) else ""
        candidates = (
            list(candidate_results[idx]) if idx < len(candidate_results) else []
        )
        raw_result = (
            candidates
            if len(candidates) > 1
            else (candidates[0] if candidates else None)
        )
        distance: Optional[float] = None
        ref_emb: Optional[np.ndarray] = None
        if reference_embeddings is not None and idx < len(reference_embeddings):
            ref_emb = reference_embeddings[idx]
            if ref_emb is not None and idx < len(embeddings):
                norm_prod = np.linalg.norm(embeddings[idx]) * np.linalg.norm(ref_emb)
                if norm_prod > 1e-12:
                    distance = float(1.0 - np.dot(embeddings[idx], ref_emb) / norm_prod)
        records.append(
            MatchRecord(
                text=text,
                predicted_id=str(prediction),
                confidence=float(confidences[idx]) if idx < len(confidences) else 0.0,
                embedding=embeddings[idx],
                candidates=candidates,
                raw_result=raw_result,
                metadata={"index": idx},
                match_method=match_method,
                reference_embedding=ref_emb,
                distance=distance,
            )
        )
    return records


def build_match_result_with_metadata(
    texts: Sequence[str],
    predictions: Sequence[str],
    confidences: np.ndarray,
    embeddings: np.ndarray,
    raw_match_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
    scores: Optional[np.ndarray] = None,
    match_method: Optional[str] = None,
) -> MatchResultWithMetadata:
    """Create a stable metadata result from matcher outputs."""
    candidate_results = normalize_candidate_results(raw_match_results, len(predictions))
    combined_metadata = dict(metadata or {})
    combined_metadata.setdefault("texts", list(texts))
    combined_metadata.setdefault("raw_match_results", raw_match_results)
    combined_metadata.setdefault("candidate_results", candidate_results)
    if match_method is not None:
        combined_metadata.setdefault("match_method", match_method)

    records = build_match_records(
        texts=texts,
        predictions=predictions,
        confidences=confidences,
        embeddings=embeddings,
        candidate_results=candidate_results,
        match_method=match_method,
    )

    return MatchResultWithMetadata(
        predictions=list(predictions),
        confidences=confidences,
        embeddings=embeddings,
        scores=scores,
        metadata=combined_metadata,
        candidate_results=candidate_results,
        records=records,
    )


def convert_match_result_to_metadata(
    match_result: Any,
    embeddings: np.ndarray,
    confidences: Optional[np.ndarray] = None,
) -> MatchResultWithMetadata:
    """
    Convert standard match result to metadata-enhanced result.
    """
    if isinstance(match_result, dict):
        predictions = [match_result.get("id", "unknown")]
        scores = np.array([match_result.get("score", 0.0)])
        raw_match_results = [match_result]
    elif isinstance(match_result, list):
        if all(isinstance(r, dict) for r in match_result):
            predictions = [r.get("id", "unknown") for r in match_result]
            scores = np.array([r.get("score", 0.0) for r in match_result])
            raw_match_results = match_result
        else:
            predictions = [str(item) for item in match_result]
            scores = None
            raw_match_results = match_result
    else:
        predictions = [str(match_result)]
        scores = None
        raw_match_results = [match_result]

    if confidences is None:
        confidences = np.ones(len(predictions))

    return build_match_result_with_metadata(
        texts=[""] * len(predictions),
        predictions=predictions,
        confidences=confidences,
        embeddings=embeddings,
        raw_match_results=raw_match_results,
        scores=scores,
    )
