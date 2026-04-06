# backends/litellm.py
import os
from typing import Optional

from .base import EmbeddingBackend, RerankerBackend

try:
    from litellm import embedding, rerank

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    embedding = rerank = None


class LiteLLMEmbedding(EmbeddingBackend):
    def __init__(self, model: str, api_key: Optional[str] = None):
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for LiteLLMEmbedding")
        self.model = model
        self._api_key = api_key

    def _get_api_key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        return os.environ.get("LITELLM_API_KEY")

    def encode(self, texts):
        kwargs = {"model": self.model, "input": texts}
        api_key = self._get_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        response = embedding(**kwargs)
        return [item["embedding"] for item in response["data"]]


class LiteLLMReranker(RerankerBackend):
    def __init__(self, model: str, api_key: Optional[str] = None):
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for LiteLLMReranker")
        self.model = model
        self._api_key = api_key

    def _get_api_key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        return os.environ.get("LITELLM_API_KEY")

    def score(self, query, docs):
        kwargs = {
            "model": self.model,
            "query": query,
            "documents": docs,
            "return_documents": False,
        }
        api_key = self._get_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        response = rerank(**kwargs)
        return [pair["relevance_score"] for pair in response["results"]]
