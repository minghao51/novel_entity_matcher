"""SentenceTransformer-based cross-encoder reranker backend."""

from ..utils.embeddings import get_cached_cross_encoder
from .base import RerankerBackend


class STReranker(RerankerBackend):
    """
    SentenceTransformer cross-encoder reranker.

    Uses CrossEncoder models for precise query-document scoring.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """
        Initialize the reranker.

        Args:
            model_name: Name or path of the CrossEncoder model
            device: Device to run model on (None for auto-detection)
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = get_cached_cross_encoder(model_name, device=device)

    def score(self, query: str, docs: list[str]) -> list[float]:
        """
        Score query-document pairs.

        Args:
            query: Query text
            docs: List of document texts

        Returns:
            List of scores (one per document)
        """
        # Prepare pairs
        pairs = [[query, doc] for doc in docs]

        # Score all pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=False,
            convert_to_tensor=True,
        )

        # Convert to list of floats
        if hasattr(scores, "cpu"):
            scores = scores.cpu().tolist()
        elif hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = list(scores)

        return scores
