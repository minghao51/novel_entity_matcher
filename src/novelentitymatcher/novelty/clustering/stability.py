"""Bootstrap-based cluster stability analysis.

Assesses cluster robustness by resampling the data, re-clustering, and
measuring Jaccard similarity of cluster membership across runs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...utils.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["ClusterStabilityScorer"]


class ClusterStabilityScorer:
    """Assess cluster stability via bootstrap resampling.

    Score > 0.7 = stable, < 0.3 = unstable.  Uses Jaccard index on
    per-cluster membership sets across ``n_bootstrap`` subsampled runs.
    """

    def __init__(
        self,
        n_bootstrap: int = 20,
        sample_fraction: float = 0.8,
        min_jaccard: float = 0.5,
        seed: int = 42,
    ):
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.min_jaccard = min_jaccard
        self.seed = seed

    def score(
        self,
        embeddings: np.ndarray,
        base_labels: np.ndarray,
        clusterer_factory: Callable[[], Any] | None = None,
    ) -> dict[int, float]:
        return self.score_simple(embeddings, base_labels, clusterer_factory)

    def score_simple(
        self,
        embeddings: np.ndarray,
        base_labels: np.ndarray,
        clusterer_factory: Callable[[], Any] | None = None,
    ) -> dict[int, float]:
        """Simplified stability scoring using pairwise Jaccard across bootstrap runs.

        More efficient than ``score()`` because it re-uses the RNG state
        and collects all bootstrap label arrays first, then computes
        pairwise Jaccard between base and each bootstrap.
        """
        from .scalable import ScalableClusterer

        X = np.asarray(embeddings, dtype=np.float32)
        n = X.shape[0]
        unique_clusters = sorted({int(lb) for lb in base_labels if int(lb) >= 0})
        if not unique_clusters:
            return {}

        rng = np.random.RandomState(self.seed)
        sample_size = max(int(n * self.sample_fraction), len(unique_clusters) + 1)

        bootstrap_labels: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(self.n_bootstrap):
            indices = rng.choice(n, size=sample_size, replace=True)
            X_sub = X[indices]

            if clusterer_factory is not None:
                clusterer = clusterer_factory()
            else:
                clusterer = ScalableClusterer(backend="auto", min_cluster_size=3)

            lbl, _, _ = clusterer.fit_predict(X_sub)
            bootstrap_labels.append((indices, lbl))

        stability: dict[int, float] = {}
        for cid in unique_clusters:
            orig_members = set(np.where(base_labels == cid)[0].tolist())
            jaccard_scores: list[float] = []

            for indices, lbl in bootstrap_labels:
                orig_in_sample = orig_members & set(indices.tolist())
                if not orig_in_sample:
                    continue

                best_j = 0.0
                for sub_cid in sorted({int(lb) for lb in lbl if int(lb) >= 0}):
                    sub_idx = set(np.where(lbl == sub_cid)[0].tolist())
                    sub_members = {int(indices[i]) for i in sub_idx}
                    intersection = len(orig_in_sample & sub_members)
                    union = len(orig_in_sample | sub_members)
                    if union > 0:
                        best_j = max(best_j, intersection / union)
                jaccard_scores.append(best_j)

            stability[cid] = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0

        return stability
