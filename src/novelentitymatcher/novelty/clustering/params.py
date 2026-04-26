"""Pydantic parameter models for clustering backends.

Provides clean, validated configuration objects for each clustering
backend, supporting benchmark sweeps over parameter combinations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["HDBSCANParams", "SOPTICSParams", "UMAPHDBSCANParams"]


class HDBSCANParams(BaseModel):
    """Parameters for the HDBSCAN clustering backend."""

    min_cluster_size: int = Field(default=5, ge=2)
    min_samples: int = Field(default=5, ge=1)
    cluster_selection_epsilon: float = Field(default=0.0, ge=0.0)
    metric: str = Field(default="cosine")


class SOPTICSParams(BaseModel):
    """Parameters for the sOPTICS clustering backend."""

    min_cluster_size: int = Field(default=5, ge=2)
    min_samples: int = Field(default=5, ge=1)
    metric: str = Field(default="cosine")


class UMAPHDBSCANParams(BaseModel):
    """Parameters for the UMAP+HDBSCAN clustering backend."""

    min_cluster_size: int = Field(default=5, ge=2)
    min_samples: int = Field(default=5, ge=1)
    cluster_selection_epsilon: float = Field(default=0.0, ge=0.0)
    n_neighbors: int = Field(default=15, ge=2)
    umap_dim: int = Field(default=10, ge=2)
    umap_metric: str = Field(default="cosine")
