from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .matcher_shared import (
    EmbeddingModel,
    TextInput,
    coerce_texts,
    flatten_entity_texts,
    normalize_texts,
    resolve_threshold,
    unwrap_single,
)
from .normalizer import TextNormalizer
from ..config import is_static_embedding_model, resolve_model_alias
from ..utils.embeddings import ModelCache, get_default_cache
from ..utils.validation import (
    validate_entities,
    validate_model_name,
    validate_threshold,
)


def _sentence_transformer_cls():
    try:
        from .matcher import SentenceTransformer
    except ImportError:
        from sentence_transformers import SentenceTransformer

    return SentenceTransformer


class EmbeddingMatcher:
    """Embedding-based similarity matching without training."""

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        threshold: float = 0.7,
        normalize: bool = True,
        embedding_dim: Optional[int] = None,
        cache: Optional[ModelCache] = None,
    ):
        validate_entities(entities)
        validate_model_name(model_name)

        self.entities = entities
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
        self.normalize = normalize
        self.embedding_dim = embedding_dim

        self.normalizer = TextNormalizer() if normalize else None
        self.cache = cache if cache is not None else get_default_cache()
        self.model: Optional[EmbeddingModel] = None
        self.entity_texts: List[str] = []
        self.entity_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._async_executor: Optional[Any] = None

    def _ensure_async_executor(self):
        if self._async_executor is None:
            from .async_utils import AsyncExecutor

            self._async_executor = AsyncExecutor()
        return self._async_executor

    def build_index(self, batch_size: Optional[int] = None):
        resolved_name = resolve_model_alias(self.model_name)

        if is_static_embedding_model(resolved_name):
            from ..backends.static_embedding import StaticEmbeddingBackend

            self.model = StaticEmbeddingBackend(
                resolved_name, embedding_dim=self.embedding_dim
            )
        else:
            sentence_transformer_cls = _sentence_transformer_cls()
            self.model = self.cache.get_or_load(
                resolved_name, lambda: sentence_transformer_cls(resolved_name)
            )

        if self.embedding_dim is not None:
            if isinstance(self.model, _sentence_transformer_cls()):
                actual_dim = self.model.get_sentence_embedding_dimension()
            elif hasattr(self.model, "embedding_dimension"):
                actual_dim = self.model.embedding_dimension
            else:
                actual_dim = None

            if actual_dim is not None and self.embedding_dim > actual_dim:
                raise ValueError(
                    f"embedding_dim ({self.embedding_dim}) cannot exceed "
                    f"model embedding dimension ({actual_dim})"
                )

            if self.embedding_dim <= 0:
                raise ValueError(
                    f"embedding_dim must be positive, got {self.embedding_dim}"
                )

        self.entity_texts, self.entity_ids = flatten_entity_texts(self.entities)
        self.entity_texts = normalize_texts(
            self.entity_texts, self.normalizer, self.normalize
        )

        if batch_size is not None:
            self.embeddings = self.model.encode(  # type: ignore[assignment]
                self.entity_texts, batch_size=batch_size
            )
        else:
            self.embeddings = self.model.encode(self.entity_texts)  # type: ignore[assignment]

        if isinstance(self.embeddings, list):
            self.embeddings = np.array(self.embeddings)

        if (
            self.embedding_dim is not None
            and self.embeddings.shape[1] > self.embedding_dim
        ):
            self.embeddings = self.embeddings[:, : self.embedding_dim]

    def match(
        self,
        texts: TextInput,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 1,
        batch_size: Optional[int] = None,
        threshold_override: Optional[float] = None,
    ) -> Any:
        if self.embeddings is None or self.model is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        texts, single_input = coerce_texts(texts)
        texts = normalize_texts(texts, self.normalizer, self.normalize)
        entity_lookup = {entity["id"]: entity for entity in self.entities}
        effective_threshold = resolve_threshold(threshold_override, self.threshold)

        if candidates is not None:
            candidate_ids = {candidate["id"] for candidate in candidates}
            candidate_indices = [
                i
                for i, entity_id in enumerate(self.entity_ids)
                if entity_id in candidate_ids
            ]
        else:
            candidate_indices = list(range(len(self.entity_ids)))

        if not candidate_indices:
            empty: Optional[List[Any]] = None if top_k == 1 else []
            return empty if single_input else [empty for _ in texts]

        candidate_embeddings = self.embeddings[candidate_indices]
        candidate_ids_list = [self.entity_ids[i] for i in candidate_indices]

        if batch_size is not None:
            query_embeddings = self.model.encode(texts, batch_size=batch_size)
        else:
            query_embeddings = self.model.encode(texts)

        if isinstance(query_embeddings, list):
            query_embeddings = np.array(query_embeddings)

        effective_dim = (
            self.embedding_dim
            if self.embedding_dim is not None
            else query_embeddings.shape[1]
        )
        if query_embeddings.shape[1] > effective_dim:
            query_embeddings = query_embeddings[:, :effective_dim]
        if candidate_embeddings.shape[1] > effective_dim:
            candidate_embeddings = candidate_embeddings[:, :effective_dim]

        similarities = cosine_similarity(query_embeddings, candidate_embeddings)

        results: List[Any] = []
        for sim_row in similarities:
            sorted_indices = np.argsort(sim_row)[::-1]
            matches: List[Dict[str, Any]] = []
            seen_ids = set()
            for idx in sorted_indices:
                score = sim_row[idx]
                if score < effective_threshold:
                    continue
                entity_id = candidate_ids_list[idx]
                if entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)
                entity = entity_lookup.get(entity_id, {})
                matches.append(
                    {
                        "id": entity_id,
                        "score": float(score),
                        "text": entity.get(
                            "text", self.entity_texts[candidate_indices[idx]]
                        ),
                    }
                )
                if len(matches) >= top_k:
                    break

            if top_k == 1:
                results.append(matches[0] if matches else None)
            else:
                results.append(matches)

        return unwrap_single(results, single_input)

    async def build_index_async(self, batch_size: Optional[int] = None):
        await self._ensure_async_executor().run_in_thread(self.build_index, batch_size)

    async def match_async(
        self,
        texts: TextInput,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 1,
        batch_size: Optional[int] = None,
        threshold_override: Optional[float] = None,
    ) -> Any:
        if self.embeddings is None or self.model is None:
            raise RuntimeError(
                "Index not built. Call build_index() or build_index_async() first."
            )

        return await self._ensure_async_executor().run_in_thread(
            self.match,
            texts=texts,
            candidates=candidates,
            top_k=top_k,
            batch_size=batch_size,
            threshold_override=threshold_override,
        )
