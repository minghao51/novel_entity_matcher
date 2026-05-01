"""
Approximate Nearest Neighbor (ANN) index wrapper for efficient similarity search.

Supports HNSWlib and FAISS backends for O(log n) similarity search.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class ANNBackend:
    """Supported ANN backends."""

    HNSWLIB = "hnswlib"
    FAISS = "faiss"
    EXACT = "exact"


class ANNIndex:
    """
    Wrapper for Approximate Nearest Neighbor indexing.

    Provides efficient O(log n) similarity search using HNSWlib or FAISS.
    """

    def __init__(
        self,
        dim: int,
        backend: str = ANNBackend.HNSWLIB,
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
    ):
        """
        Initialize ANN index.

        Args:
            dim: Dimensionality of embeddings
            backend: ANN backend to use ('hnswlib' or 'faiss')
            max_elements: Maximum number of elements to index
            ef_construction: HNSW ef_construction parameter (higher = better quality)
            M: HNSW M parameter (higher = better quality, more memory)
        """
        self.dim = dim
        self.backend = backend
        self.max_elements = max_elements
        self._index: Any = None
        self._labels: list[str] = []
        self._vector_buffer: list[np.ndarray] = []
        self._vectors: np.ndarray | None = None
        self._hnsw_params: dict = {}

        if backend == ANNBackend.HNSWLIB:
            self._init_hnswlib(ef_construction, M)
        elif backend == ANNBackend.FAISS:
            self._init_faiss()
        elif backend == ANNBackend.EXACT:
            logger.info("Initialized exact ANN fallback with dim=%s", self.dim)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _init_hnswlib(self, ef_construction: int, M: int):
        """Initialize HNSWlib index."""
        try:
            import hnswlib

            self._hnsw_params = {"ef_construction": ef_construction, "M": M}
            self._index = hnswlib.Index(space="cosine", dim=self.dim)
            self._index.init_index(
                max_elements=self.max_elements,
                ef_construction=ef_construction,
                M=M,
            )
            self._index.set_ef(ef_construction)
            logger.info(f"Initialized HNSWlib index with dim={self.dim}")
        except ImportError:
            logger.warning(
                "hnswlib is unavailable; falling back to exact similarity search"
            )
            self.backend = ANNBackend.EXACT

    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss

            # Use IndexFlatIP for inner product (similar to cosine for normalized vectors)
            self._index = faiss.IndexFlatIP(self.dim)
            logger.info(f"Initialized FAISS index with dim={self.dim}")
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISS backend. "
                "Install with: pip install faiss-cpu"
            ) from None

    def add_vectors(self, vectors: np.ndarray, labels: list[str] | None = None) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: Array of shape (n_vectors, dim)
            labels: Optional labels for the vectors
        """
        if len(vectors) == 0:
            return

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dim}, got {vectors.shape[1]}"
            )

        # Normalize vectors for cosine similarity
        vectors = self._normalize(vectors).astype(np.float32, copy=False)

        if self.backend == ANNBackend.HNSWLIB:
            current_count = self._index.get_current_count()
            if current_count + len(vectors) > self.max_elements:
                self._resize_hnsw_index(current_count + len(vectors))
            self._index.add_items(vectors)
        elif self.backend == ANNBackend.FAISS:
            self._index.add(vectors)

        self._vector_buffer.append(vectors)
        self._vectors = None

        if labels:
            self._labels.extend(labels)
        else:
            start = len(self._labels)
            self._labels.extend([str(i) for i in range(start, start + len(vectors))])

    def _resize_hnsw_index(self, needed: int) -> None:
        new_max = max(needed, int(self.max_elements * 2))
        logger.info(
            "Resizing HNSW index from %d to %d elements",
            self.max_elements,
            new_max,
        )
        self._index.resize_index(new_max)
        self.max_elements = new_max

    def _ensure_vectors(self) -> np.ndarray:
        if self._vectors is not None:
            return self._vectors
        if not self._vector_buffer:
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
        elif len(self._vector_buffer) == 1:
            self._vectors = self._vector_buffer[0]
        else:
            self._vectors = np.vstack(self._vector_buffer)
            self._vector_buffer = [self._vectors]
        return self._vectors

    def knn_query(self, query: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Find k-nearest neighbors for query vector(s).

        Args:
            query: Query vector or vectors of shape (n_queries, dim)
            k: Number of neighbors to return

        Returns:
            Tuple of (distances, indices)
            - distances: Array of shape (n_queries, k) with similarity scores
            - indices: Array of shape (n_queries, k) with neighbor indices
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize query vectors
        query = self._normalize(query)

        if self.backend == ANNBackend.HNSWLIB:
            labels, distances = self._index.knn_query(query, k=k)
            # HNSWlib returns distances (lower is better), convert to similarities
            similarities = 1 - distances
            return similarities, labels
        if self.backend == ANNBackend.FAISS:
            distances, indices = self._index.search(query, k)
            # FAISS IndexFlatIP returns similarities directly
            return distances, indices

        if self._ensure_vectors().size == 0:
            empty = np.empty((len(query), 0), dtype=np.float32)
            return empty, empty.astype(int)

        vectors = self._ensure_vectors()
        k = min(k, len(vectors))
        similarities = np.dot(query.astype(np.float32, copy=False), vectors.T)
        top_indices = np.argsort(-similarities, axis=1)[:, :k]
        top_similarities = np.take_along_axis(similarities, top_indices, axis=1)
        return top_similarities, top_indices

    def get_distance_matrix(
        self, queries: np.ndarray, targets: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Get distance matrix between queries and all indexed vectors.

        Args:
            queries: Query vectors of shape (n_queries, dim)
            targets: Optional target vectors (if None, use all indexed vectors)

        Returns:
            Distance matrix of shape (n_queries, n_targets)
        """
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        # Normalize queries
        queries = self._normalize(queries).astype(np.float32, copy=False)

        if targets is None:
            vectors = self._ensure_vectors()
            if vectors.size == 0:
                return np.zeros((len(queries), 0), dtype=np.float32)
            return np.dot(queries, vectors.T)
        else:
            # Compute direct similarity
            targets = self._normalize(targets).astype(np.float32, copy=False)
            return np.dot(queries, targets.T)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms

    def save(self, path: str | Path) -> None:
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        labels_path = path.with_suffix(".labels.json")
        vectors_path = path.with_suffix(".vectors.npy")

        if self.backend == ANNBackend.HNSWLIB:
            self._index.save_index(str(path.with_suffix(".bin")))
            logger.info(f"Saved HNSWlib index to {path}")
        elif self.backend == ANNBackend.FAISS:
            import faiss

            faiss.write_index(self._index, str(path.with_suffix(".index")))
            logger.info(f"Saved FAISS index to {path}")
        else:
            logger.info(f"Saved exact ANN fallback index to {path}")

        labels_path.write_text(
            json.dumps(self._labels, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(vectors_path, self._ensure_vectors())

    def load(self, path: str | Path) -> None:
        """Load index from disk."""
        path = Path(path)
        labels_path = path.with_suffix(".labels.json")
        vectors_path = path.with_suffix(".vectors.npy")

        if self.backend == ANNBackend.HNSWLIB:
            bin_path = path.with_suffix(".bin")
            if not bin_path.exists():
                raise FileNotFoundError(f"Index file not found: {bin_path}")
            self._index.load_index(str(bin_path))
            logger.info(f"Loaded HNSWlib index from {path}")
        elif self.backend == ANNBackend.FAISS:
            import faiss

            index_path = path.with_suffix(".index")
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            self._index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index from {path}")
        else:
            logger.info(f"Loaded exact ANN fallback index from {path}")

        if labels_path.exists():
            loaded_labels = json.loads(labels_path.read_text(encoding="utf-8"))
            self._labels = [str(label) for label in loaded_labels]
        else:
            # Backward-compatible fallback for older saved indexes.
            self._labels = [str(i) for i in range(self.n_elements)]

        if vectors_path.exists():
            self._vectors = np.load(vectors_path).astype(np.float32, copy=False)
            self._vector_buffer = [self._vectors]
        else:
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
            self._vector_buffer = []

    @property
    def n_elements(self) -> int:
        """Get number of elements in the index."""
        if self.backend == ANNBackend.HNSWLIB:
            return self._index.get_current_count()
        if self.backend == ANNBackend.FAISS:
            return self._index.ntotal
        return len(self._ensure_vectors())

    def clear(self) -> None:
        """Clear all elements from the index."""
        if self.backend == ANNBackend.HNSWLIB:
            # HNSWlib doesn't support clear, need to reinitialize
            raise NotImplementedError(
                "HNSWlib doesn't support clearing. Create a new index instead."
            )
        elif self.backend == ANNBackend.FAISS:
            import faiss

            self._index = faiss.IndexFlatIP(self.dim)
            self._labels = []
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
            self._vector_buffer = []
            logger.info("Cleared FAISS index")
        else:
            self._labels = []
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
            self._vector_buffer = []
            logger.info("Cleared exact ANN fallback index")

    @property
    def labels(self) -> list[str]:
        """Return the labels stored alongside indexed vectors."""
        return list(self._labels)
