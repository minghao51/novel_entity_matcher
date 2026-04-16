import novelentitymatcher.backends.reranker_st as reranker_st_module
import novelentitymatcher.backends.sentencetransformer as st_backend_module


def test_hf_reranker_score_returns_numeric_list(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs):
            return [0.9, 0.2, -0.1]

    monkeypatch.setattr(
        st_backend_module,
        "get_cached_cross_encoder",
        lambda m, d=None, c=None: FakeCrossEncoder(m),
    )

    backend = st_backend_module.HFReranker("fake-model")
    scores = backend.score("query", ["doc1", "doc2", "doc3"])

    assert scores == [0.9, 0.2, -0.1]
    assert all(isinstance(score, float) for score in scores)


def test_hf_reranker_rerank_uses_cross_encoder_scores(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs):
            # Matches order of input docs: A, B, C
            return [0.1, 0.8, 0.3]

    monkeypatch.setattr(
        st_backend_module,
        "get_cached_cross_encoder",
        lambda m, d=None, c=None: FakeCrossEncoder(m),
    )

    backend = st_backend_module.HFReranker("fake-model")
    candidates = [
        {"id": "a", "text": "A"},
        {"id": "b", "text": "B"},
        {"id": "c", "text": "C"},
    ]
    reranked = backend.rerank("query", candidates, top_k=2)

    assert [item["id"] for item in reranked] == ["b", "c"]
    assert reranked[0]["cross_encoder_score"] == 0.8


def test_st_reranker_forwards_device(monkeypatch):
    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name, device=None):
            captured["model_name"] = model_name
            captured["device"] = device

    def fake_get_cached_cross_encoder(model_name, device=None, cache=None):
        return FakeCrossEncoder(model_name, device)

    monkeypatch.setattr(
        reranker_st_module, "get_cached_cross_encoder", fake_get_cached_cross_encoder
    )

    reranker_st_module.STReranker("fake-reranker", device="cpu")

    assert captured == {"model_name": "fake-reranker", "device": "cpu"}
