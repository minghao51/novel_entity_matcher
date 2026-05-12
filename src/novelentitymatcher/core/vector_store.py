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
        self._latest_positions: dict[str, int] = {}

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self._index.add_vectors(vectors, labels=ids)
        for id_ in ids:
            self._deleted_ids.discard(id_)
        start_idx = len(self._index.labels) - len(ids)
        for offset, id_ in enumerate(ids):
            self._latest_positions[id_] = start_idx + offset
        if metadata:
            for id_, meta in zip(ids, metadata, strict=False):
                self._metadata[id_] = meta

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        # Over-fetch to account for deleted and stale superseded entries.
        stale_entries = max(0, len(self._index.labels) - len(self._latest_positions))
        fetch_k = top_k + len(self._deleted_ids) + stale_entries
        similarities, indices = self._index.knn_query(vector, k=max(fetch_k, top_k))
        results: list[dict[str, Any]] = []
        for sim, idx in zip(similarities[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._index.labels):
                continue
            label = self._index.labels[idx]
            latest_idx = self._latest_positions.get(label)
            if latest_idx is not None and idx != latest_idx:
                continue
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
        return sum(1 for id_ in self._latest_positions if id_ not in self._deleted_ids)

    @property
    def dim(self) -> int:
        return self._index.dim


class ChromaVectorStore:
    """Vector store backed by ChromaDB (optional dependency).

    Requires ``chromadb``. Import errors are raised lazily on first use.
    """

    def __init__(
        self,
        collection_name: str = "novel_entity_matcher",
        persist_directory: str | None = None,
        **kwargs: Any,
    ):
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._client: Any = None
        self._collection: Any = None
        self._dim: int | None = None
        self._kwargs = kwargs

    def _ensure_initialized(self) -> None:
        if self._collection is not None:
            return
        try:
            import chromadb  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            ) from exc

        self._client = (
            chromadb.PersistentClient(
                path=self._persist_directory,
            )
            if self._persist_directory
            else chromadb.EphemeralClient()
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ensure_initialized()
        metadatas: list[dict[str, Any] | None] = metadata or [None] * len(ids)  # type: ignore[list-item,assignment]
        self._collection.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
        )
        if self._dim is None and vectors.shape[1] > 0:
            self._dim = vectors.shape[1]

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_initialized()
        results = self._collection.query(
            query_embeddings=vector.reshape(1, -1).tolist(),
            n_results=top_k,
            where=filter,
        )
        out: list[dict[str, Any]] = []
        if not results["ids"]:
            return out
        for _, (id_, dist, meta) in enumerate(
            zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0],
                strict=False,
            )
        ):
            if id_ is None:
                continue
            entry: dict[str, Any] = {"id": id_, "score": float(1.0 - dist)}
            if meta is not None:
                entry["metadata"] = meta
            out.append(entry)
        return out

    def delete(self, ids: list[str]) -> None:
        self._ensure_initialized()
        self._collection.delete(ids=ids)

    def count(self) -> int:
        self._ensure_initialized()
        return self._collection.count()

    @property
    def dim(self) -> int | None:
        return self._dim
