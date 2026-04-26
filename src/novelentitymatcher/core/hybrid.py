"""Hybrid matching pipeline with blocking, retrieval, and reranking."""

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .blocking import BlockingStrategy, NoOpBlocking
from .matcher import EmbeddingMatcher
from .reranker import CrossEncoderReranker


class HybridMatcher:
    """
    Three-stage waterfall pipeline for semantic entity matching.

    Combines fast blocking, semantic retrieval, and precise reranking
    for accurate and efficient matching.

    Pipeline Stages:
        1. Blocking (BM25/TF-IDF/Fuzzy) - Fast lexical filtering
        2. Bi-Encoder Retrieval - Semantic similarity search
        3. Cross-Encoder Reranking - Precise cross-attention scoring

    Example:
        >>> from novelentitymatcher import HybridMatcher
        >>> from novelentitymatcher.core.blocking import BM25Blocking
        >>>
        >>> matcher = HybridMatcher(
        ...     entities=products,
        ...     blocking_strategy=BM25Blocking(),
        ...     retriever_model="bge-base",
        ...     reranker_model="bge-m3"
        ... )
        >>>
        >>> results = matcher.match(
        ...     "iPhone 15 case",
        ...     blocking_top_k=1000,
        ...     retrieval_top_k=50,
        ...     final_top_k=5
        ... )
    """

    def __init__(
        self,
        entities: list[dict[str, Any]],
        blocking_strategy: BlockingStrategy | None = None,
        retriever_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        normalize: bool = True,
    ):
        """
        Initialize the hybrid matcher.

        Args:
            entities: List of entity dictionaries
            blocking_strategy: Blocking strategy (defaults to NoOpBlocking)
            retriever_model: Model name for bi-encoder retrieval
            reranker_model: Model name for cross-encoder reranking
            normalize: Whether to normalize text (lowercase, remove accents, etc.)
        """
        # Stage 1: Blocking
        self.blocker = blocking_strategy or NoOpBlocking()

        # Stage 2: Bi-Encoder Retrieval
        self.retriever = EmbeddingMatcher(
            entities=entities,
            model_name=retriever_model,
            normalize=normalize,
        )
        self.retriever.build_index()

        # Stage 3: Cross-Encoder Reranking
        self.reranker = CrossEncoderReranker(model=reranker_model)

    def match(
        self,
        query: str,
        blocking_top_k: int = 1000,
        retrieval_top_k: int = 50,
        final_top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Match query using three-stage waterfall pipeline.

        Args:
            query: Search query
            blocking_top_k: Number of candidates after blocking stage
            retrieval_top_k: Number of candidates after retrieval stage
            final_top_k: Number of final results after reranking

        Returns:
            List of matched entities with scores (bi-encoder and cross-encoder)
        """
        # Stage 1: Blocking - Fast lexical filtering
        candidates = self.blocker.block(
            query, self.retriever.entities, top_k=blocking_top_k
        )

        # Early exit if no candidates from blocking
        if not candidates:
            return []

        # Stage 2: Bi-Encoder Retrieval - Semantic similarity
        retrieved = self.retriever.match(
            query,
            candidates=candidates,
            top_k=retrieval_top_k,
        )

        # Ensure retrieved is a list (handle single result case)
        if retrieved is None:
            return []
        if not isinstance(retrieved, list):
            retrieved = [retrieved]

        # Filter out None results
        retrieved = [r for r in retrieved if r is not None]

        # Stage 3: Cross-Encoder Reranking - Precise scoring
        if not retrieved:
            return []

        final = self.reranker.rerank(query, retrieved, top_k=final_top_k)

        return final

    def match_bulk(
        self,
        queries: list[str],
        blocking_top_k: int = 1000,
        retrieval_top_k: int = 50,
        final_top_k: int = 5,
        n_jobs: int = -1,
        chunk_size: int | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Batch matching for multiple queries.

        Batches bi-encoder encoding across all queries (single model.encode call
        instead of one per query), then computes per-query similarity against
        blocked candidates.

        Args:
            queries: List of search queries
            blocking_top_k: Number of candidates after blocking stage
            retrieval_top_k: Number of candidates after retrieval stage
            final_top_k: Number of final results after reranking
            n_jobs: Ignored (kept for backwards compatibility).
            chunk_size: Ignored (kept for backwards compatibility).

        Returns:
            List of matched entity lists (one per query)
        """
        if not queries:
            return []

        # Stage 1: Blocking - per-query lexical filtering
        all_candidates: list[list[dict[str, Any]]] = []
        for query in queries:
            candidates = self.blocker.block(
                query, self.retriever.entities, top_k=blocking_top_k
            )
            all_candidates.append(candidates or [])

        # Stage 2: Bi-Encoder Retrieval - batched encoding
        query_embeddings = self.retriever.model.encode(queries)
        if isinstance(query_embeddings, list):
            query_embeddings = np.array(query_embeddings)

        entity_lookup = {e["id"]: e for e in self.retriever.entities}
        all_retrieved: list[list[dict[str, Any]]] = []

        for i in range(len(queries)):
            candidates = all_candidates[i]
            if not candidates:
                all_retrieved.append([])
                continue

            candidate_ids = {c["id"] for c in candidates}
            candidate_indices = [
                j
                for j, eid in enumerate(self.retriever.entity_ids)
                if eid in candidate_ids
            ]
            if not candidate_indices:
                all_retrieved.append([])
                continue

            candidate_embeddings = self.retriever.embeddings[candidate_indices]
            query_emb = query_embeddings[i : i + 1]
            similarities = cosine_similarity(query_emb, candidate_embeddings)[0]

            sorted_indices = np.argsort(similarities)[::-1]
            seen_ids: set[str] = set()
            retrieved: list[dict[str, Any]] = []
            for idx in sorted_indices:
                score = similarities[idx]
                if score < self.retriever.threshold:
                    continue
                entity_id = self.retriever.entity_ids[candidate_indices[idx]]
                if entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)
                entity = entity_lookup.get(entity_id, {})
                retrieved.append(
                    {
                        "id": entity_id,
                        "score": float(score),
                        "text": entity.get(
                            "name",
                            self.retriever.entity_texts[candidate_indices[idx]],
                        ),
                    }
                )
                if len(retrieved) >= retrieval_top_k:
                    break

            all_retrieved.append(retrieved)

        # Stage 3: Cross-Encoder Reranking - per-query
        results: list[list[dict[str, Any]]] = []
        for query, retrieved in zip(queries, all_retrieved, strict=True):
            if not retrieved:
                results.append([])
            else:
                results.append(
                    self.reranker.rerank(query, retrieved, top_k=final_top_k)
                )

        return results
