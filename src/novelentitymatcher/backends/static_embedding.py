import os
import platform
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import EmbeddingBackend

# Keep this aligned with matcher.py so direct backend imports behave the same way.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from sentence_transformers.models import StaticEmbedding

from ..utils.embeddings import get_cached_sentence_transformer


class StaticEmbeddingBackend(EmbeddingBackend):
    """
    Backend for static embeddings.

    Supports two approaches:
    1. SentenceTransformer's StaticEmbedding module
    2. model2vec StaticModel (for minishlab potion models and custom distillations)

    Models:
    - RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en (StaticEmbedding)
    - minishlab/potion-base-8M (model2vec)
    - minishlab/potion-base-32M (model2vec)
    """

    def __init__(self, model_name: str, embedding_dim: int | None = None):
        """
        Initialize static embedding backend.

        Args:
            model_name: HuggingFace model name or local path
            embedding_dim: Optional dimension for MRL models (None = full dimension)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.backend_type: str | None = None
        self.model: Any = None

        # Try to load with model2vec first (for minishlab models)
        try:
            from model2vec import StaticModel

            self.model = StaticModel.from_pretrained(model_name)
            self.backend_type = "model2vec"
            return
        except (ImportError, ValueError, RuntimeError):
            pass

        # Fall back to SentenceTransformer (for RikkaBotan MRL and others)
        try:
            # Some models (like RikkaBotan) require trust_remote_code=True
            # to load custom modules like SSE
            self.model = get_cached_sentence_transformer(
                model_name, trust_remote_code=True
            )
            self.backend_type = "sentence_transformers"
            self._is_native_static = any(
                isinstance(m, StaticEmbedding) for m in self.model.modules()
            )
            return
        except (OSError, RuntimeError, ValueError) as e:
            raise ValueError(
                f"Failed to load static embedding model {model_name}. "
                f"Tried both model2vec and SentenceTransformer. Last error: {e}"
            ) from e

    def encode(self, texts, batch_size: int = 32):
        """Generate embeddings using static lookup."""
        if self.backend_type == "model2vec":
            embeddings = self.model.encode(texts)
        else:  # sentence_transformers
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
            )

        # Convert to list if needed
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        elif not isinstance(embeddings, list):
            embeddings = list(embeddings)

        # Apply dimension reduction if requested (for MRL models)
        if self.embedding_dim is not None and embeddings:
            embeddings = [emb[: self.embedding_dim] for emb in embeddings]

        return embeddings

    def similarity(self, query_embeddings, corpus_embeddings, top_k):
        """Compute similarity using cosine similarity."""
        query_embeddings = np.array(query_embeddings)
        corpus_embeddings = np.array(corpus_embeddings)

        similarities = cosine_similarity(query_embeddings, corpus_embeddings)

        results = []
        for _idx, query_sim in enumerate(similarities):
            top_indices = query_sim.argsort()[-top_k:][::-1]
            results.append(
                [
                    {"corpus_id": int(i), "score": float(query_sim[i])}
                    for i in top_indices
                ]
            )
        return results

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension of the model."""
        if self.backend_type == "model2vec":
            # model2vec models have .dim attribute
            return int(self.model.dim)
        else:
            # SentenceTransformer models
            dim = self.model.get_sentence_embedding_dimension()
            if dim is None:
                raise ValueError(
                    f"Could not determine embedding dimension for {self.model_name}"
                )
            return dim
