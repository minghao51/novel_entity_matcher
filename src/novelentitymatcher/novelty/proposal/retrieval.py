"""
Retrieval-Augmented LLM Class Proposer.

Enhances LLM-based class proposal with retrieval of in-context examples
using dense embeddings (BGE-M3 style) for improved class naming.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...utils.embeddings import get_cached_sentence_transformer
from ...utils.logging_config import get_logger

if TYPE_CHECKING:
    from ...backends.embedding import EmbeddingBackend

logger = get_logger(__name__)


class RetrievalAugmentedProposer:
    """
    LLM class proposer enhanced with retrieval-based in-context examples.

    Retrieves most relevant examples from a corpus to include in the LLM
    prompt, improving class naming quality.
    """

    def __init__(
        self,
        retriever: EmbeddingBackend | None = None,
        llm_proposer: Any | None = None,
        k_examples: int = 5,
        k_novel_per_class: int = 3,
        retrieval_metric: str = "cosine",
        rerank: bool = False,
    ):
        """
        Initialize retrieval-augmented proposer.

        Args:
            retriever: Embedding backend for retrieval (e.g., BGE-M3)
            llm_proposer: Existing LLMClassProposer to enhance
            k_examples: Number of in-context examples to retrieve
            k_novel_per_class: Number of novel examples per proposed class
            retrieval_metric: Similarity metric for retrieval
            rerank: Whether to use reranking for better examples
        """
        self.retriever = retriever
        self.llm_proposer = llm_proposer
        self.k_examples = k_examples
        self.k_novel_per_class = k_novel_per_class
        self.retrieval_metric = retrieval_metric
        self.rerank = rerank

        self._example_corpus: list[str] = []
        self._example_embeddings: Any | None = None
        self._is_indexed: bool = False

    def index_examples(
        self,
        examples: list[str],
        embeddings: Any | None = None,
    ) -> None:
        """
        Index examples for retrieval.

        Args:
            examples: List of example texts to index
            embeddings: Pre-computed embeddings (if None, will compute)
        """
        self._example_corpus = examples

        if embeddings is not None:
            self._example_embeddings = embeddings
        elif self.retriever is not None:
            self._example_embeddings = self.retriever.encode(examples)

        self._is_indexed = True
        logger.info(f"Indexed {len(examples)} examples for retrieval")

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve k most relevant examples for a query.

        Args:
            query: Query text
            k: Number of examples to retrieve (default: k_examples)

        Returns:
            List of dicts with 'text', 'score', 'index'
        """
        if not self._is_indexed:
            raise RuntimeError("Must call index_examples() before retrieve()")

        k = k or self.k_examples

        if self.retriever is None:
            logger.warning("No retriever available, returning empty results")
            return []

        query_embedding = self.retriever.encode([query])

        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(
            query_embedding,
            self._example_embeddings,
        )[0]

        top_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )[:k]

        results = [
            {
                "text": self._example_corpus[idx],
                "score": float(similarities[idx]),
                "index": int(idx),
            }
            for idx in top_indices
        ]

        return results

    def retrieve_by_class(
        self,
        class_name: str,
        novel_samples: list[Any],
        existing_classes: list[str],
    ) -> dict[str, Any]:
        """
        Retrieve examples relevant to a proposed class.

        Args:
            class_name: Proposed class name
            novel_samples: Novel samples to find examples for
            existing_classes: List of existing class names

        Returns:
            Dict with retrieved examples and metadata
        """
        if not novel_samples:
            return {"examples": [], "class_name": class_name}

        texts = [s.text if hasattr(s, "text") else str(s) for s in novel_samples]
        query = f"{class_name}: {', '.join(texts[:3])}"

        retrieved = self.retrieve(query, k=self.k_novel_per_class)

        return {
            "class_name": class_name,
            "examples": retrieved,
            "query": query,
        }

    def build_prompt(
        self,
        novel_samples: list[Any],
        existing_classes: list[str],
        context: str | None = None,
        use_retrieval: bool = True,
    ) -> str:
        """
        Build prompt for LLM class proposal with retrieval.

        Args:
            novel_samples: Novel samples to propose classes for
            existing_classes: List of existing class names
            context: Optional domain context
            use_retrieval: Whether to include retrieved examples

        Returns:
            Formatted prompt string
        """
        sample_texts = [
            f"- {s.text if hasattr(s, 'text') else str(s)}" for s in novel_samples[:20]
        ]
        if len(novel_samples) > 20:
            sample_texts.append(f"... and {len(novel_samples) - 20} more samples")

        samples_section = "\n".join(sample_texts)

        existing_section = ", ".join(existing_classes) if existing_classes else "None"

        context_section = f"\n\nDomain Context: {context}" if context else ""

        retrieval_section = ""
        if use_retrieval and self._is_indexed and self.retriever:
            retrieved_examples = []
            for sample in novel_samples[:5]:
                text = sample.text if hasattr(sample, "text") else str(sample)
                results = self.retrieve(text, k=2)
                for r in results:
                    retrieved_examples.append(
                        f'- Example: "{r["text"]}" (relevance: {r["score"]:.2f})'
                    )

            if retrieved_examples:
                retrieval_section = "\n\nRetrieved relevant examples:\n" + "\n".join(
                    retrieved_examples[:10]
                )

        prompt = f"""You are analyzing text samples that don't fit well into existing categories.

Existing Classes: {existing_section}{context_section}{retrieval_section}

Novel Samples (detected as not fitting existing classes):
{samples_section}

Your task is to:
1. Analyze these samples to identify meaningful new categories
2. Propose concise, descriptive class names
3. Provide justifications for each proposal
4. Identify samples that should be rejected as noise

IMPORTANT RESPONSE FORMAT:
You must respond with a valid JSON object matching this schema:
{{
  "proposed_classes": [
    {{
      "name": "class name (2-4 words)",
      "description": "clear description of what this class represents",
      "confidence": 0.0-1.0,
      "sample_count": number of samples fitting this class,
      "example_samples": ["sample1", "sample2", "sample3"],
      "justification": "why this class makes sense",
      "suggested_parent": null or "parent class name if hierarchical"
    }}
  ],
  "rejected_as_noise": ["sample text to reject"],
  "analysis_summary": "brief summary of your analysis",
  "cluster_count": number of distinct clusters found
}}

Guidelines:
- Class names should be concise (2-4 words), descriptive
- Confidence should reflect how clearly the samples form a coherent category
- Only propose classes with at least 3 supporting samples
- Reject samples that appear to be noise, errors, or too diverse
- Return "proposed_classes": [] if no coherent new class should be created
- Consider hierarchical relationships if relevant to the domain

Provide your analysis as a JSON object:"""

        return prompt

    def propose_classes(
        self,
        novel_samples: list[Any],
        existing_classes: list[str],
        context: str | None = None,
    ) -> Any | None:
        """
        Propose new classes with retrieval-augmented prompting.

        Args:
            novel_samples: Novel samples to propose classes for
            existing_classes: List of existing class names
            context: Optional domain context

        Returns:
            NovelClassAnalysis from LLM or None if unavailable
        """
        if not self.llm_proposer:
            logger.warning("No LLM proposer configured")
            return None

        prompt = self.build_prompt(
            novel_samples=novel_samples,
            existing_classes=existing_classes,
            context=context,
            use_retrieval=True,
        )

        try:
            response, model_used = self._call_llm_with_fallback(prompt)
            analysis = self._parse_response(response, model_used)
            return analysis
        except (ValueError, TypeError, ConnectionError, RuntimeError) as e:
            logger.error(f"LLM proposal failed: {e}")
            return None

    def _call_llm_with_fallback(self, prompt: str) -> tuple[str, str]:
        """Call LLM with fallback."""
        if self.llm_proposer and hasattr(self.llm_proposer, "_call_llm_with_fallback"):
            return self.llm_proposer._call_llm_with_fallback(prompt)
        raise RuntimeError("No LLM proposer available")

    def _parse_response(self, response: str, model_used: str) -> Any:
        """Parse LLM response into NovelClassAnalysis."""
        if self.llm_proposer and hasattr(self.llm_proposer, "_parse_response"):
            return self.llm_proposer._parse_response(response, [], model_used)
        raise RuntimeError("No LLM proposer available")

    @property
    def is_ready(self) -> bool:
        """Check if proposer is ready for use."""
        return self._is_indexed or self.retriever is None


class BGERetriever:
    """
    BGE-M3 style dense retriever for examples.

    Simple wrapper that uses sentence-transformers for
    dense retrieval of in-context examples.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
    ):
        """
        Initialize BGE retriever.

        Args:
            model_name: Model name for sentence-transformers
            device: Device to use ("cuda", "cpu", or None for auto)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any | None = None
        self._is_initialized = False

    def _initialize(self) -> None:
        """Lazy initialization of the model."""
        if self._is_initialized:
            return

        try:
            self._model = get_cached_sentence_transformer(self.model_name)
            if self.device:
                self._model = self._model.to(self.device)
            self._is_initialized = True
            logger.info(f"Loaded BGE model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> Any:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Override batch size

        Returns:
            numpy array of embeddings (n, dim)
        """
        self._initialize()

        batch_size = batch_size or self.batch_size
        assert self._model is not None, "Model should be initialized"
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings

    def similarity(
        self,
        query_embeddings: Any,
        corpus_embeddings: Any,
    ) -> np.ndarray:
        """
        Compute similarity between query and corpus.

        Args:
            query_embeddings: Query embeddings (n, dim)
            corpus_embeddings: Corpus embeddings (m, dim)

        Returns:
            Similarity matrix (n, m)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(query_embeddings, corpus_embeddings)
