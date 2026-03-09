from .base import EmbeddingBackend, RerankerBackend
from .sentencetransformer import HFEmbedding, HFReranker
from .static_embedding import StaticEmbeddingBackend
from .reranker_st import STReranker

__all__ = [
    "EmbeddingBackend",
    "RerankerBackend",
    "HFEmbedding",
    "HFReranker",
    "StaticEmbeddingBackend",
    "STReranker",
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
