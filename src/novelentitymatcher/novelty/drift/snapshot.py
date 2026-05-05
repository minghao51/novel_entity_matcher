"""Reference distribution snapshot for drift detection.

Captures statistical summaries of reference embedding distributions so that
drift can be detected later by comparing new embeddings against the snapshot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DistributionSnapshot:
    """Statistical summary of the reference embedding distribution."""

    timestamp: datetime
    n_points: int
    mean: np.ndarray
    covariance: np.ndarray
    per_class_stats: dict[str, dict[str, Any]]
    embedding_hash: str

    @classmethod
    def from_embeddings(
        cls,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> DistributionSnapshot:
        """Build a snapshot from reference embeddings and labels."""
        mean = embeddings.mean(axis=0)
        cov = np.cov(embeddings, rowvar=False)
        if cov.ndim < 2:
            cov = np.array([[cov]])

        per_class = {}
        for label in set(labels):
            mask = np.array(labels) == label
            class_embs = embeddings[mask]
            class_cov = np.cov(class_embs, rowvar=False)
            if class_cov.ndim < 2:
                class_cov = np.array([[class_cov]])
            per_class[label] = {
                "mean": class_embs.mean(axis=0),
                "cov": class_cov,
                "count": int(mask.sum()),
            }

        hasher = hashlib.sha256()
        hasher.update(embeddings.tobytes())

        return cls(
            timestamp=datetime.now(tz=timezone.utc),
            n_points=len(embeddings),
            mean=mean,
            covariance=cov,
            per_class_stats=per_class,
            embedding_hash=hasher.hexdigest(),
        )

    def save(self, path: str | Path) -> None:
        """Persist snapshot to disk as compressed NPZ + JSON metadata."""
        path = Path(path)
        npz_path = path.with_suffix(".npz")

        np.savez_compressed(
            npz_path,
            mean=self.mean,
            covariance=self.covariance,
        )

        metadata = {
            "timestamp": self.timestamp.isoformat(),
            "n_points": self.n_points,
            "embedding_hash": self.embedding_hash,
            "per_class_stats": {
                k: {
                    "mean": v["mean"].tolist(),
                    "count": v["count"],
                }
                for k, v in self.per_class_stats.items()
            },
        }

        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> DistributionSnapshot:
        """Load a snapshot from disk."""
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        meta_path = path.with_suffix(".json")

        npz = np.load(npz_path)
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        per_class = {}
        for label, stats in metadata["per_class_stats"].items():
            mean = np.array(stats["mean"])
            # Covariance is not stored per-class in NPZ; recompute on load
            # if needed, or store separately. For now we keep mean/count only
            # and reconstruct empty cov to avoid bloating files.
            per_class[label] = {
                "mean": mean,
                "cov": np.eye(len(mean)) * 1e-6,
                "count": stats["count"],
            }

        return cls(
            timestamp=datetime.fromisoformat(metadata["timestamp"]),
            n_points=metadata["n_points"],
            mean=np.array(npz["mean"]),
            covariance=np.array(npz["covariance"]),
            per_class_stats=per_class,
            embedding_hash=metadata["embedding_hash"],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DistributionSnapshot):
            return NotImplemented
        return (
            self.n_points == other.n_points
            and self.embedding_hash == other.embedding_hash
            and np.allclose(self.mean, other.mean)
        )
