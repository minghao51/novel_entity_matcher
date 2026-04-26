from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np

from .normalizer import TextNormalizer

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from ..backends.static_embedding import StaticEmbeddingBackend


EmbeddingModel = Union["SentenceTransformer", "StaticEmbeddingBackend"]
TextInput = Union[str, list[str]]


def coerce_texts(texts: TextInput) -> tuple[list[str], bool]:
    if isinstance(texts, str):
        return [texts], True
    return texts, False


def extract_top_prediction_metadata(
    match_results: Any, single_input: bool
) -> tuple[list[str], np.ndarray]:
    """
    Normalize matcher output into top-1 predictions and confidences.

    Novel class detection only needs the best prediction per input. This keeps a
    stable shape even when the underlying matcher returns dicts, lists, strings,
    or ``None`` values.
    """

    def _from_result(result: Any) -> tuple[str, float]:
        if result is None:
            return "unknown", 0.0
        if isinstance(result, dict):
            return result.get("id", "unknown"), float(result.get("score", 0.0))
        if isinstance(result, list):
            if not result:
                return "unknown", 0.0
            first = result[0]
            if isinstance(first, dict):
                return first.get("id", "unknown"), float(first.get("score", 0.0))
            if first is None:
                return "unknown", 0.0
            return str(first), 1.0
        return str(result), 1.0

    if single_input:
        prediction, confidence = _from_result(match_results)
        return [prediction], np.array([confidence], dtype=float)

    predictions: list[str] = []
    confidences: list[float] = []
    for result in match_results:
        prediction, confidence = _from_result(result)
        predictions.append(prediction)
        confidences.append(confidence)

    return predictions, np.array(confidences, dtype=float)


def unwrap_single(results: list[Any], single_input: bool) -> Any:
    if single_input:
        return results[0]
    return results


def normalize_texts(
    texts: list[str],
    normalizer: TextNormalizer | None,
    normalize: bool,
) -> list[str]:
    if not (normalizer and normalize):
        return texts
    return [normalizer.normalize(text) for text in texts]


def normalize_training_data(
    training_data: list[dict],
    normalizer: TextNormalizer | None,
    normalize: bool,
) -> list[dict]:
    if not (normalizer and normalize):
        return training_data
    return [
        {"text": normalizer.normalize(item["text"]), "label": item["label"]}
        for item in training_data
    ]


def flatten_entity_texts(
    entities: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    entity_texts: list[str] = []
    entity_ids: list[str] = []
    for entity in entities:
        entity_texts.append(entity["name"])
        entity_ids.append(entity["id"])
        for alias in entity.get("aliases", []):
            entity_texts.append(alias)
            entity_ids.append(entity["id"])
    return entity_texts, entity_ids


def resolve_threshold(threshold_override: float | None, default: float) -> float:
    from ..utils.validation import validate_threshold

    if threshold_override is None:
        return default
    return validate_threshold(threshold_override)
