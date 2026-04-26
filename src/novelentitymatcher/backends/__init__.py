from .base import EmbeddingBackend, RerankerBackend
from .reranker_st import STReranker
from .sentencetransformer import HFEmbedding, HFReranker
from .static_embedding import StaticEmbeddingBackend

__all__ = [
    "EmbeddingBackend",
    "HFEmbedding",
    "HFReranker",
    "RerankerBackend",
    "STReranker",
    "StaticEmbeddingBackend",
    "get_embedding_backend",
    "get_reranker_backend",
]


def get_embedding_backend(provider: str, model: str, **kwargs) -> EmbeddingBackend:
    if provider == "huggingface":
        return HFEmbedding(model)
    if provider == "static":
        return StaticEmbeddingBackend(model, **kwargs)
    raise ValueError(f"Unknown embedding provider: {provider}")


def get_reranker_backend(provider: str, model: str, **kwargs) -> RerankerBackend:
    if provider == "huggingface":
        return HFReranker(model)
    raise ValueError(f"Unknown reranker provider: {provider}")
