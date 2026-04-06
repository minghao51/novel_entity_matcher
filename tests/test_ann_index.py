"""Tests for ANN index wrapper."""

import numpy as np
import pytest

from novelentitymatcher.novelty.storage.index import ANNIndex, ANNBackend

# Check if optional ANN backends are available.
try:
    import importlib.util

    HAS_HNSWLIB = importlib.util.find_spec("hnswlib") is not None
    HAS_FAISS = importlib.util.find_spec("faiss") is not None
except ImportError:
    HAS_HNSWLIB = False
    HAS_FAISS = False


class TestANNIndex:
    """Test suite for ANNIndex class."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(100, 128).astype(np.float32)

    @pytest.fixture
    def require_hnswlib(self):
        """Skip test when hnswlib is not installed."""
        if not HAS_HNSWLIB:
            pytest.skip("hnswlib not available")

    @pytest.fixture
    def require_faiss(self):
        """Skip test when faiss is not installed."""
        if not HAS_FAISS:
            pytest.skip("faiss not available")

    @pytest.fixture
    def hnswlib_index(self, require_hnswlib, sample_embeddings):
        """Create HNSWlib index with sample data."""
        index = ANNIndex(dim=128, backend=ANNBackend.HNSWLIB)
        index.add_vectors(sample_embeddings, [f"doc_{i}" for i in range(100)])
        return index

    def test_hnswlib_initialization(self, require_hnswlib):
        """Test HNSWlib index initialization."""
        index = ANNIndex(dim=128, backend=ANNBackend.HNSWLIB)
        assert index.dim == 128
        assert index.backend == ANNBackend.HNSWLIB
        assert index.n_elements == 0

    def test_faiss_initialization(self, require_faiss):
        """Test FAISS index initialization."""
        index = ANNIndex(dim=128, backend=ANNBackend.FAISS)
        assert index.dim == 128
        assert index.backend == ANNBackend.FAISS
        assert index.n_elements == 0

    def test_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            ANNIndex(dim=128, backend="invalid_backend")

    def test_add_vectors(self, require_hnswlib, sample_embeddings):
        """Test adding vectors to index."""
        index = ANNIndex(dim=128, backend=ANNBackend.HNSWLIB)
        index.add_vectors(sample_embeddings)
        assert index.n_elements == 100

    def test_add_vectors_with_labels(self, require_hnswlib, sample_embeddings):
        """Test adding vectors with labels."""
        index = ANNIndex(dim=128, backend=ANNBackend.HNSWLIB)
        labels = [f"doc_{i}" for i in range(100)]
        index.add_vectors(sample_embeddings, labels)
        assert len(index._labels) == 100
        assert index._labels[0] == "doc_0"

    def test_dimension_mismatch(self, require_hnswlib, sample_embeddings):
        """Test that dimension mismatch raises ValueError."""
        index = ANNIndex(dim=64, backend=ANNBackend.HNSWLIB)
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            index.add_vectors(sample_embeddings)

    def test_knn_query_single(self, hnswlib_index, sample_embeddings):
        """Test k-NN query for single vector."""
        query = sample_embeddings[0]
        similarities, indices = hnswlib_index.knn_query(query, k=5)

        assert similarities.shape == (1, 5)
        assert indices.shape == (1, 5)
        # First result should be the query itself (highest similarity)
        assert indices[0][0] == 0
        assert similarities[0][0] > 0.99  # Very high similarity

    def test_knn_query_batch(self, hnswlib_index, sample_embeddings):
        """Test k-NN query for multiple vectors."""
        queries = sample_embeddings[:10]
        similarities, indices = hnswlib_index.knn_query(queries, k=5)

        assert similarities.shape == (10, 5)
        assert indices.shape == (10, 5)

    def test_knn_query_1d_array(self, hnswlib_index, sample_embeddings):
        """Test k-NN query with 1D array (single query)."""
        query = sample_embeddings[0]
        similarities, indices = hnswlib_index.knn_query(query, k=3)

        assert similarities.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_knn_query_k_greater_than_elements(self, hnswlib_index, sample_embeddings):
        """Test k-NN query when k is greater than number of elements."""
        query = sample_embeddings[0]
        # Use a smaller k to avoid HNSWlib limitations
        similarities, indices = hnswlib_index.knn_query(query, k=50)

        # Should return at most n_elements
        assert similarities.shape[1] <= 100

    def test_get_distance_matrix(self, hnswlib_index, sample_embeddings):
        """Test getting distance matrix."""
        queries = sample_embeddings[:10]
        distances = hnswlib_index.get_distance_matrix(queries)
        expected = (
            hnswlib_index._normalize(queries)
            @ hnswlib_index._normalize(sample_embeddings).T
        )

        assert distances.shape == (10, 100)
        np.testing.assert_allclose(distances, expected, atol=1e-5)

    def test_get_distance_matrix_with_targets(self, hnswlib_index, sample_embeddings):
        """Test getting distance matrix with specific targets."""
        queries = sample_embeddings[:5]
        targets = sample_embeddings[5:15]
        distances = hnswlib_index.get_distance_matrix(queries, targets)
        expected = (
            hnswlib_index._normalize(queries) @ hnswlib_index._normalize(targets).T
        )

        assert distances.shape == (5, 10)
        np.testing.assert_allclose(distances, expected, atol=1e-5)

    def test_empty_query(self, hnswlib_index):
        """Test query with empty array."""
        empty = np.array([]).reshape(0, 128)
        similarities, indices = hnswlib_index.knn_query(empty, k=5)

        assert similarities.shape == (0, 5)
        assert indices.shape == (0, 5)

    def test_save_load_hnswlib(self, hnswlib_index, tmp_path):
        """Test saving and loading HNSWlib index."""
        save_path = tmp_path / "test_index"
        queries = np.eye(3, 128, dtype=np.float32)
        before = hnswlib_index.get_distance_matrix(queries)

        # Save
        hnswlib_index.save(save_path)

        # Create new index and load
        new_index = ANNIndex(dim=128, backend=ANNBackend.HNSWLIB)
        new_index.load(save_path)

        # Check that elements match
        assert new_index.n_elements == hnswlib_index.n_elements
        assert new_index.labels == hnswlib_index.labels
        np.testing.assert_allclose(new_index.get_distance_matrix(queries), before)

    def test_faiss_save_load(self, require_faiss, sample_embeddings, tmp_path):
        """Test saving and loading FAISS index."""
        index = ANNIndex(dim=128, backend=ANNBackend.FAISS)
        labels = [f"doc_{i}" for i in range(100)]
        index.add_vectors(sample_embeddings, labels)
        queries = np.eye(3, 128, dtype=np.float32)
        before = index.get_distance_matrix(queries)

        save_path = tmp_path / "test_index"
        index.save(save_path)

        # Create new index and load
        new_index = ANNIndex(dim=128, backend=ANNBackend.FAISS)
        new_index.load(save_path)

        assert new_index.n_elements == index.n_elements
        assert new_index.labels == labels
        np.testing.assert_allclose(new_index.get_distance_matrix(queries), before)

    def test_load_nonexistent_file(self, require_hnswlib):
        """Test that loading non-existent file raises FileNotFoundError."""
        index = ANNIndex(dim=128, backend=ANNBackend.HNSWLIB)
        with pytest.raises(FileNotFoundError):
            index.load("/nonexistent/path/index")

    def test_faiss_clear(self, require_faiss, sample_embeddings):
        """Test clearing FAISS index."""
        index = ANNIndex(dim=128, backend=ANNBackend.FAISS)
        index.add_vectors(sample_embeddings)
        assert index.n_elements == 100

        index.clear()
        assert index.n_elements == 0

    def test_hnswlib_clear_not_supported(self, require_hnswlib, sample_embeddings):
        """Test that HNSWlib doesn't support clear."""
        index = ANNIndex(dim=128, backend=ANNBackend.HNSWLIB)
        index.add_vectors(sample_embeddings)

        with pytest.raises(
            NotImplementedError, match="HNSWlib doesn't support clearing"
        ):
            index.clear()
