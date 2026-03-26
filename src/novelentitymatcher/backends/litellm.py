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
        if api_key:
            os.environ["LITELLM_API_KEY"] = api_key
        self.model = model

    def encode(self, texts):
        response = embedding(model=self.model, input=texts)
        return [item["embedding"] for item in response["data"]]


class LiteLLMReranker(RerankerBackend):
    def __init__(self, model: str, api_key: Optional[str] = None):
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for LiteLLMReranker")
        if api_key:
            os.environ["LITELLM_API_KEY"] = api_key
        self.model = model

    def score(self, query, docs):
        response = rerank(
            model=self.model, query=query, documents=docs, return_documents=False
        )
        return [pair["relevance_score"] for pair in response["results"]]
