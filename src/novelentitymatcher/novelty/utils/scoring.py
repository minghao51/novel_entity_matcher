"""
Shared scoring utilities for novelty detection.

This module contains common scoring functions used across
multiple strategies and components.
"""

import numpy as np


def normalize_score(score: float | np.ndarray) -> float | np.ndarray:
    """
    Normalize a score to the range [0, 1].

    Args:
        score: Score or array of scores to normalize

    Returns:
        Normalized score(s) in range [0, 1]
    """
    if isinstance(score, np.ndarray):
        if score.max() == score.min():
            return np.ones_like(score) * 0.5
        return (score - score.min()) / (score.max() - score.min())
    else:
        # For single value, we need bounds
        # Default to assuming input is in range [-1, 1] or [0, inf]
        if score < 0:
            return np.clip((score + 1) / 2, 0.0, 1.0)
        else:
            return np.clip(score, 0.0, 1.0)


def compute_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
    metric: str = "cosine",
) -> float:
    """
    Compute similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector
        metric: Similarity metric ('cosine', 'euclidean', 'dot')

    Returns:
        Similarity score
    """
    if metric == "cosine":
        # Cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    elif metric == "euclidean":
        # Convert euclidean distance to similarity
        dist = np.linalg.norm(vec1 - vec2)
        return float(1.0 / (1.0 + dist))

    elif metric == "dot":
        # Dot product similarity
        return float(np.dot(vec1, vec2))

    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def compute_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    metric: str = "cosine",
) -> float:
    """
    Compute distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector
        metric: Distance metric ('cosine', 'euclidean')

    Returns:
        Distance score
    """
    if metric == "cosine":
        # Cosine distance = 1 - cosine similarity
        sim = compute_similarity(vec1, vec2, "cosine")
        return 1.0 - sim

    elif metric == "euclidean":
        return float(np.linalg.norm(vec1 - vec2))

    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def compute_uncertainty(
    confidences: np.ndarray,
    method: str = "entropy",
) -> np.ndarray:
    """
    Compute uncertainty scores from confidence values.

    Args:
        confidences: Array of confidence scores
        method: Uncertainty computation method ('entropy', 'margin', 'least_confident')

    Returns:
        Uncertainty scores (higher = more uncertain)
    """
    if method == "least_confident":
        # 1 - max confidence
        return 1.0 - confidences

    elif method == "margin":
        # For multi-class, this would be difference between top 2
        # For binary/single confidence, just use 1 - confidence
        return 1.0 - confidences

    elif method == "entropy":
        # Simple entropy approximation
        # For proper entropy, we'd need full probability distributions
        eps = 1e-10
        p = np.clip(confidences, eps, 1.0 - eps)
        return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)

    else:
        raise ValueError(f"Unknown uncertainty method: {method}")
