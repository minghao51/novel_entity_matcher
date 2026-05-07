"""Stability filter pipeline stage.

Inserts between CommunityDetectionStage and ClusterEvidenceStage.
Filters clusters with bootstrap stability below a configurable threshold.
"""

from __future__ import annotations

from typing import Any

from ...novelty.clustering.stability import ClusterStabilityScorer
from ..contracts import PipelineStage, StageContext, StageResult

__all__ = ["StabilityFilterStage"]


class StabilityFilterStage(PipelineStage):
    """Filter unstable clusters before evidence extraction.

    Uses ``ClusterStabilityScorer`` to compute per-cluster Jaccard
    stability scores and removes clusters below ``stability_threshold``.
    """

    name = "stability_filter"

    def __init__(
        self,
        *,
        enabled: bool = True,
        stability_threshold: float = 0.5,
        n_bootstrap: int = 10,
        sample_fraction: float = 0.8,
        seed: int = 42,
    ):
        self.enabled = enabled
        self.stability_threshold = stability_threshold
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.seed = seed

    def run(self, context: StageContext) -> StageResult:
        clusters: list[Any] = list(context.artifacts.get("discovery_clusters", []))
        report = context.artifacts.get("novel_sample_report")
        match_result = context.artifacts.get("match_result")

        if not self.enabled or not clusters:
            return StageResult(
                stage_name=self.name,
                artifacts={
                    "discovery_clusters": clusters,
                    "stability_scores": {},
                },
                metadata={"enabled": False, "num_clusters": len(clusters)},
            )

        embeddings = None
        if match_result is not None and hasattr(match_result, "embeddings"):
            embeddings = match_result.embeddings

        if embeddings is None or report is None:
            return StageResult(
                stage_name=self.name,
                artifacts={
                    "discovery_clusters": clusters,
                    "stability_scores": {},
                },
                metadata={
                    "enabled": True,
                    "num_clusters": len(clusters),
                    "skipped": True,
                    "skip_reason": "missing embeddings or report",
                },
            )

        sample_indices_by_cluster: dict[int, list[int]] = {}
        for cluster in clusters:
            cid = cluster.cluster_id
            sample_indices_by_cluster[cid] = cluster.sample_indices

        cluster_embedding_blocks: dict[int, list[Any]] = {}
        for cid, indices in sample_indices_by_cluster.items():
            block = []
            for idx in indices:
                if idx < len(embeddings):
                    block.append(embeddings[idx])
            if block:
                cluster_embedding_blocks[cid] = block

        if len(cluster_embedding_blocks) <= 1:
            return StageResult(
                stage_name=self.name,
                artifacts={
                    "discovery_clusters": clusters,
                    "stability_scores": {},
                },
                metadata={
                    "enabled": True,
                    "num_clusters": len(clusters),
                    "skipped": True,
                    "skip_reason": "too few clusters for stability analysis",
                },
            )

        all_embs = []
        all_labels = []
        for cid in sorted(cluster_embedding_blocks.keys()):
            block = cluster_embedding_blocks[cid]
            n = len(block)
            all_embs.extend(block)
            all_labels.extend([cid] * n)

        if len(set(all_labels)) < 2:
            return StageResult(
                stage_name=self.name,
                artifacts={
                    "discovery_clusters": clusters,
                    "stability_scores": {},
                },
                metadata={
                    "enabled": True,
                    "num_clusters": len(clusters),
                    "skipped": True,
                    "skip_reason": "only one cluster after dedup",
                },
            )

        import numpy as np

        emb_array = np.array(all_embs, dtype=np.float32)
        label_array = np.array(all_labels, dtype=int)

        scorer = ClusterStabilityScorer(
            n_bootstrap=self.n_bootstrap,
            sample_fraction=self.sample_fraction,
            seed=self.seed,
        )
        stability_scores = scorer.score_simple(emb_array, label_array)

        stable_clusters = [
            c
            for c in clusters
            if stability_scores.get(c.cluster_id, 1.0) >= self.stability_threshold
        ]

        for c in stable_clusters:
            score = stability_scores.get(c.cluster_id)
            if score is not None:
                if c.metadata is None:
                    c.metadata = {}
                c.metadata["stability_score"] = float(score)

        return StageResult(
            stage_name=self.name,
            artifacts={
                "discovery_clusters": stable_clusters,
                "stability_scores": stability_scores,
            },
            metadata={
                "enabled": True,
                "num_clusters_before": len(clusters),
                "num_clusters_after": len(stable_clusters),
                "stability_threshold": self.stability_threshold,
                "filtered_cluster_ids": [
                    c.cluster_id
                    for c in clusters
                    if stability_scores.get(c.cluster_id, 1.0)
                    < self.stability_threshold
                ],
            },
        )
