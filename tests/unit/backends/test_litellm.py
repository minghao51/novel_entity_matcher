from novelentitymatcher.backends import litellm as litellm_backend


def test_litellm_embedding_prefers_explicit_api_key(monkeypatch):
    monkeypatch.setattr(litellm_backend, "LITELLM_AVAILABLE", True)
    captured = {}

    def fake_embedding(**kwargs):
        captured.update(kwargs)
        return {"data": [{"embedding": [0.1, 0.2]}]}

    monkeypatch.setattr(litellm_backend, "embedding", fake_embedding)
    monkeypatch.setenv("LITELLM_API_KEY", "env-key")

    backend = litellm_backend.LiteLLMEmbedding("test-model", api_key="explicit-key")
    backend.encode(["hello"])

    assert captured["api_key"] == "explicit-key"


def test_litellm_reranker_uses_env_api_key_when_explicit_key_absent(monkeypatch):
    monkeypatch.setattr(litellm_backend, "LITELLM_AVAILABLE", True)
    captured = {}

    def fake_rerank(**kwargs):
        captured.update(kwargs)
        return {"results": [{"relevance_score": 0.9}]}

    monkeypatch.setattr(litellm_backend, "rerank", fake_rerank)
    monkeypatch.setenv("LITELLM_API_KEY", "env-key")

    backend = litellm_backend.LiteLLMReranker("test-model")
    backend.score("query", ["doc"])

    assert captured["api_key"] == "env-key"
