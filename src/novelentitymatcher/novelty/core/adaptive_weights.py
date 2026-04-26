"""
Adaptive strategy weight computation.

Computes dataset-aware strategy weights based on characteristics like
class separability, sample density, and embedding dimensionality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config.weights import WeightConfig

logger = logging.getLogger(__name__)


@dataclass
class DatasetCharacteristics:
    """Summary statistics used for adaptive weight computation."""

    n_samples: int
    n_classes: int
    samples_per_class: dict[str, int]
    class_separability: float
    mean_intra_class_distance: float
    mean_inter_class_distance: float
    effective_dimensionality: float
    class_balance_entropy: float
    mean_knn_distance: float


def compute_characteristics(
    embeddings: np.ndarray,
    labels: list[str],
    k: int = 5,
) -> DatasetCharacteristics:
    """
    Compute dataset characteristics for adaptive weight tuning.

    Args:
        embeddings: (n_samples, dim) embedding matrix
        labels: class labels for each sample
        k: number of neighbors for density estimation

    Returns:
        DatasetCharacteristics with computed metrics
    """
    unique_labels = list(set(labels))
    n_samples = len(labels)
    n_classes = len(unique_labels)

    samples_per_class = {}
    centroids = {}
    for label in unique_labels:
        mask = np.array(labels) == label
        class_embeddings = embeddings[mask]
        samples_per_class[label] = int(np.sum(mask))
        centroids[label] = np.mean(class_embeddings, axis=0)

    intra_distances = []
    for label in unique_labels:
        mask = np.array(labels) == label
        class_embeddings = embeddings[mask]
        if len(class_embeddings) > 1:
            centroid = centroids[label]
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-12:
                emb_norms = np.linalg.norm(class_embeddings, axis=1, keepdims=True)
                emb_norms = np.clip(emb_norms, 1e-12, None)
                sims = (class_embeddings @ centroid) / (
                    emb_norms.squeeze() * centroid_norm
                )
                intra_distances.extend((1.0 - sims).tolist())

    inter_distances = []
    for i, l1 in enumerate(unique_labels):
        for j, l2 in enumerate(unique_labels):
            if j > i:
                c1 = centroids[l1]
                c2 = centroids[l2]
                n1 = np.linalg.norm(c1)
                n2 = np.linalg.norm(c2)
                if n1 > 1e-12 and n2 > 1e-12:
                    sim = np.dot(c1, c2) / (n1 * n2)
                    inter_distances.append(1.0 - sim)

    mean_intra = float(np.mean(intra_distances)) if intra_distances else 0.0
    mean_inter = float(np.mean(inter_distances)) if inter_distances else 0.0
    separability = mean_inter / mean_intra if mean_intra > 1e-12 else 0.0

    effective_dim = _effective_dimensionality(embeddings)

    class_counts = np.array(
        [samples_per_class[cls] for cls in unique_labels], dtype=float
    )
    probs = class_counts / class_counts.sum()
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = np.log(n_classes) if n_classes > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    mean_knn_dist = _mean_knn_distance(embeddings, k=min(k, n_samples - 1))

    return DatasetCharacteristics(
        n_samples=n_samples,
        n_classes=n_classes,
        samples_per_class=samples_per_class,
        class_separability=separability,
        mean_intra_class_distance=mean_intra,
        mean_inter_class_distance=mean_inter,
        effective_dimensionality=effective_dim,
        class_balance_entropy=normalized_entropy,
        mean_knn_distance=mean_knn_dist,
    )


def adaptive_weights(
    characteristics: DatasetCharacteristics,
    base_weights: WeightConfig | None = None,
) -> WeightConfig:
    """
    Compute adaptive strategy weights from dataset characteristics.

    Rules:
    - High separability → increase centroid distance weight
    - High density (low mean_knn) → increase kNN weight
    - Low samples per class → decrease LOF/Mahalanobis weight
    - High dimensionality → increase uncertainty weight
    - Imbalanced classes → increase confidence weight

    Args:
        characteristics: dataset metrics from compute_characteristics
        base_weights: starting weights (uses defaults if None)

    Returns:
        Adjusted WeightConfig
    """
    base = base_weights or WeightConfig()

    sep = characteristics.class_separability
    density = 1.0 / (1.0 + characteristics.mean_knn_distance)
    min_samples = (
        min(characteristics.samples_per_class.values())
        if characteristics.samples_per_class
        else 1
    )
    eff_dim = characteristics.effective_dimensionality
    imbalance = characteristics.class_balance_entropy

    knn_mult = 1.0 + 0.5 * density
    centroid_mult = 1.0 + 0.5 * min(sep / 3.0, 1.0)
    uncertainty_mult = 1.0 + 0.3 * eff_dim
    confidence_mult = 1.0 + 0.3 * (1.0 - imbalance)

    oneclass_mult = max(0.2, 1.0 - 0.5 * max(0, (5 - min_samples) / 5.0))

    return WeightConfig(
        confidence=base.confidence * confidence_mult,
        uncertainty=base.uncertainty * uncertainty_mult,
        knn=base.knn * knn_mult,
        cluster=base.cluster,
        self_knowledge=base.self_knowledge,
        pattern=base.pattern,
        oneclass=base.oneclass * oneclass_mult,
        prototypical=base.prototypical,
        setfit=base.setfit,
        setfit_centroid=base.setfit_centroid * centroid_mult,
    )


def _effective_dimensionality(embeddings: np.ndarray) -> float:
    """
    Estimate effective dimensionality via PCA explained variance.

    Returns a value in [0, 1] representing how many dimensions are
    actually informative (1.0 = all dimensions useful).
    """
    if embeddings.shape[0] < 2:
        return 0.0

    centered = embeddings - embeddings.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.5

    eigenvalues = s**2
    total = eigenvalues.sum()
    if total < 1e-12:
        return 0.0

    cumulative = np.cumsum(eigenvalues) / total
    n_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    return n_95 / len(eigenvalues)


def _mean_knn_distance(embeddings: np.ndarray, k: int = 5) -> float:
    """Compute mean kNN cosine distance across the dataset."""
    if embeddings.shape[0] <= k:
        return 0.5

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = embeddings / norms
    similarity = normalized @ normalized.T

    np.fill_diagonal(similarity, -np.inf)
    knn_similarities = np.partition(similarity, -k, axis=1)[:, -k:]
    knn_distances = 1.0 - knn_similarities
    return float(np.mean(knn_distances))
