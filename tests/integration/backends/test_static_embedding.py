import pytest

# Skip RikkaBotan tests if dependencies are not available
pytest.importorskip("model2vec", reason="model2vec not installed")

from novelentitymatcher.backends.static_embedding import StaticEmbeddingBackend


@pytest.mark.hf
@pytest.mark.skip(reason="RikkaBotan model requires additional SSE module dependencies")
def test_static_backend_mrl_model():
    """Test RikkaBotan MRL model via StaticEmbedding."""
    backend = StaticEmbeddingBackend(
        "RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en"
    )
    assert backend.model is not None
    assert backend.backend_type == "sentence_transformers"


@pytest.mark.hf
def test_static_backend_potion_model():
    """Test minishlab potion model via model2vec."""
    backend = StaticEmbeddingBackend("minishlab/potion-base-8M")
    assert backend.model is not None
    assert backend.backend_type == "model2vec"


@pytest.mark.hf
def test_static_backend_encode():
    """Test encoding with static backend."""
    backend = StaticEmbeddingBackend("minishlab/potion-base-8M")
    texts = ["Hello world", "Test sentence"]
    embeddings = backend.encode(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0


@pytest.mark.hf
def test_static_backend_dimension_reduction():
    """Test MRL dimension reduction capability."""
    # Use model2vec model for testing since RikkaBotan has dependency issues
    full_dim_backend = StaticEmbeddingBackend("minishlab/potion-base-32M")
    reduced_dim_backend = StaticEmbeddingBackend(
        "minishlab/potion-base-32M", embedding_dim=256
    )

    texts = ["test query"]
    full_emb = full_dim_backend.encode(texts)[0]
    reduced_emb = reduced_dim_backend.encode(texts)[0]

    assert len(reduced_emb) <= 256
    assert len(full_emb) >= len(reduced_emb)


@pytest.mark.hf
def test_static_vs_dynamic_comparison():
    """Compare static vs dynamic embedding quality."""
    from novelentitymatcher.backends.sentencetransformer import HFEmbedding

    static_backend = StaticEmbeddingBackend("minishlab/potion-base-8M")
    dynamic_backend = HFEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    texts = ["semantic search", "machine learning"]

    static_emb = static_backend.encode(texts)
    dynamic_emb = dynamic_backend.encode(texts)

    # Both should produce embeddings
    assert len(static_emb) == len(dynamic_emb) == 2


@pytest.mark.hf
def test_embedding_dimension_property():
    """Test embedding_dimension property for model2vec backend."""
    # Test model2vec backend
    potion_backend = StaticEmbeddingBackend("minishlab/potion-base-8M")
    assert potion_backend.embedding_dimension > 0


@pytest.mark.hf
def test_static_backend_similarity():
    """Test similarity computation with static backend."""
    backend = StaticEmbeddingBackend("minishlab/potion-base-8M")

    query_texts = ["machine learning"]
    corpus_texts = ["deep learning", "computer vision", "natural language processing"]

    query_embeddings = backend.encode(query_texts)
    corpus_embeddings = backend.encode(corpus_texts)

    results = backend.similarity(query_embeddings, corpus_embeddings, top_k=2)

    assert len(results) == 1
    assert len(results[0]) == 2
    assert "corpus_id" in results[0][0]
    assert "score" in results[0][0]
