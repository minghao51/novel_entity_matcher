from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ..novelty.storage.index import ANNBackend, ANNIndex


class VectorStore(Protocol):
    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None: ...

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    def delete(self, ids: list[str]) -> None: ...

    def count(self) -> int: ...


class InMemoryVectorStore:
    def __init__(
        self,
        dim: int,
        backend: str = ANNBackend.HNSWLIB,
        **kwargs: Any,
    ):
        self._index = ANNIndex(dim=dim, backend=backend, **kwargs)
        self._metadata: dict[str, dict[str, Any]] = {}
        self._deleted_ids: set[str] = set()

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self._index.add_vectors(vectors, labels=ids)
        if metadata:
            for id_, meta in zip(ids, metadata, strict=False):
                self._metadata[id_] = meta

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        # Over-fetch to account for deleted entries.
        fetch_k = top_k + len(self._deleted_ids)
        similarities, indices = self._index.knn_query(vector, k=max(fetch_k, top_k))
        results: list[dict[str, Any]] = []
        for sim, idx in zip(similarities[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._index.labels):
                continue
            label = self._index.labels[idx]
            if label in self._deleted_ids:
                continue
            if filter is not None:
                meta = self._metadata.get(label, {})
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue
            result: dict[str, Any] = {
                "id": label,
                "score": float(sim),
            }
            if label in self._metadata:
                result["metadata"] = self._metadata[label]
            results.append(result)
            if len(results) >= top_k:
                break
        return results

    def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            self._metadata.pop(id_, None)
            self._deleted_ids.add(id_)

    def count(self) -> int:
        return self._index.n_elements - len(self._deleted_ids)

    @property
    def dim(self) -> int:
        return self._index.dim
