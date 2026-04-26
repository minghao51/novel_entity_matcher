"""Cross-encoder reranking for semantic entity matching."""

from typing import Any

from ..backends.reranker_st import STReranker
from ..config import resolve_reranker_alias


class CrossEncoderReranker:
    """
    User-facing API for cross-encoder reranking.

    Provides precise reranking of candidate entities using cross-encoder models.
    Typically used after initial retrieval with bi-encoder models.

    Example:
        >>> from novelentitymatcher import EmbeddingMatcher, CrossEncoderReranker
        >>>
        >>> # Initial retrieval
        >>> retriever = EmbeddingMatcher(entities, model_name="bge-base")
        >>> retriever.build_index()
        >>> candidates = retriever.match(query, top_k=50)
        >>>
        >>> # Rerank top candidates
        >>> reranker = CrossEncoderReranker(model="bge-m3")
        >>> final_results = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model: str = "bge-m3",
        backend=None,
        device: str | None = None,
        batch_size: int = 32,
    ):
        """
        Initialize the reranker.

        Args:
            model: Model alias or full model name
            backend: Custom backend implementation (defaults to STReranker)
            device: Device to run model on (None for auto-detection)
            batch_size: Batch size for inference
        """
        self.model_name = resolve_reranker_alias(model)

        if backend is None:
            backend = STReranker(
                model_name=self.model_name,
                device=device,
                batch_size=batch_size,
            )

        self.backend = backend
        self.device = device
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        text_field: str = "text",
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Query text
            candidates: List of candidate dictionaries
            top_k: Number of top results to return
            text_field: Field name containing text to score

        Returns:
            Reranked list of candidates (top_k only) with added 'cross_encoder_score' field
        """
        if not candidates:
            return []

        return self.backend.rerank(
            query, candidates, top_k=top_k, text_field=text_field
        )

    def rerank_batch(
        self,
        queries: list[str],
        candidates_list: list[list[dict[str, Any]]],
        top_k: int = 5,
        text_field: str = "text",
    ) -> list[list[dict[str, Any]]]:
        """
        Batch reranking for multiple queries.

        Args:
            queries: List of query texts
            candidates_list: List of candidate lists (one per query)
            top_k: Number of top results to return per query
            text_field: Field name containing text to score

        Returns:
            List of reranked candidate lists
        """
        if len(queries) != len(candidates_list):
            raise ValueError("queries and candidates_list must have the same length")
        return [
            self.backend.rerank(query, cands, top_k=top_k, text_field=text_field)
            for query, cands in zip(queries, candidates_list, strict=False)
        ]

    def score(self, query: str, docs: list[str]) -> list[float]:
        """
        Score query-document pairs.

        Args:
            query: Query text
            docs: List of document texts

        Returns:
            List of scores (one per document)
        """
        return self.backend.score(query, docs)
