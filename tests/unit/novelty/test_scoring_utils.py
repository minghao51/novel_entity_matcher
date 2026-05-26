"""Unit tests for novelty scoring utilities."""

import numpy as np
import pytest

from novelentitymatcher.novelty.utils import compute_similarity, normalize_score
from novelentitymatcher.novelty.utils.scoring import (
    compute_distance,
    compute_uncertainty,
)


def test_normalize_score_array_constant_returns_half() -> None:
    values = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    normalized = normalize_score(values)
    assert isinstance(normalized, np.ndarray)
    assert np.allclose(normalized, np.array([0.5, 0.5, 0.5], dtype=np.float32))


def test_normalize_score_array_scales_min_max() -> None:
    values = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    normalized = normalize_score(values)
    assert isinstance(normalized, np.ndarray)
    assert np.allclose(normalized, np.array([0.0, 0.5, 1.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (-2.0, 0.0),
        (-1.0, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (2.0, 1.0),
    ],
)
def test_normalize_score_scalar_paths(score: float, expected: float) -> None:
    normalized = normalize_score(score)
    assert isinstance(normalized, float)
    assert normalized == expected


def test_compute_similarity_cosine_and_zero_norm() -> None:
    vec = np.array([1.0, 0.0], dtype=np.float32)
    same = compute_similarity(vec, vec, metric="cosine")
    zero_norm = compute_similarity(np.zeros(2, dtype=np.float32), vec, metric="cosine")
    assert same == pytest.approx(1.0)
    assert zero_norm == 0.0


def test_compute_similarity_euclidean_and_dot() -> None:
    vec1 = np.array([1.0, 2.0], dtype=np.float32)
    vec2 = np.array([1.0, 5.0], dtype=np.float32)
    euclidean = compute_similarity(vec1, vec2, metric="euclidean")
    dot = compute_similarity(vec1, vec2, metric="dot")
    assert euclidean == pytest.approx(1.0 / (1.0 + 3.0))
    assert dot == pytest.approx(11.0)


def test_compute_similarity_invalid_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unknown similarity metric"):
        compute_similarity(np.array([1.0]), np.array([1.0]), metric="invalid")


def test_compute_distance_metrics_and_invalid() -> None:
    vec1 = np.array([1.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0], dtype=np.float32)
    cosine_distance = compute_distance(vec1, vec2, metric="cosine")
    euclidean_distance = compute_distance(vec1, vec2, metric="euclidean")
    assert cosine_distance == pytest.approx(1.0)
    assert euclidean_distance == pytest.approx(np.sqrt(2.0))
    with pytest.raises(ValueError, match="Unknown distance metric"):
        compute_distance(vec1, vec2, metric="invalid")


@pytest.mark.parametrize("method", ["least_confident", "margin"])
def test_compute_uncertainty_linear_methods(method: str) -> None:
    confidences = np.array([0.2, 0.8], dtype=np.float32)
    uncertainty = compute_uncertainty(confidences, method=method)
    assert np.allclose(uncertainty, np.array([0.8, 0.2], dtype=np.float32))


def test_compute_uncertainty_entropy_and_invalid() -> None:
    confidences = np.array([0.5, 0.9], dtype=np.float32)
    uncertainty = compute_uncertainty(confidences, method="entropy")
    assert uncertainty[0] > uncertainty[1]
    with pytest.raises(ValueError, match="Unknown uncertainty method"):
        compute_uncertainty(confidences, method="invalid")
