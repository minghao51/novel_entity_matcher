import os
import platform
from typing import Optional, Union, List, Dict, Any, Tuple, TYPE_CHECKING
from collections import defaultdict
import numpy as np

# Enable CPU fallback for unsupported MPS ops before torch/sentence-transformers import.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .classifier import SetFitClassifier
from .normalizer import TextNormalizer
from ..utils.validation import (
    validate_entities,
    validate_model_name,
    validate_threshold,
)
from ..utils.embeddings import ModelCache, get_default_cache
from ..config import (
    is_static_embedding_model,
    resolve_model_alias,
    resolve_training_model_alias,
    supports_training_model,
)
from ..exceptions import ModeError, TrainingError, ValidationError

if TYPE_CHECKING:
    from ..backends.static_embedding import StaticEmbeddingBackend

# Type alias for embedding backend models
EmbeddingModel = Union[SentenceTransformer, "StaticEmbeddingBackend"]

TextInput = Union[str, List[str]]


def _coerce_texts(texts: TextInput) -> Tuple[List[str], bool]:
    if isinstance(texts, str):
        return [texts], True
    return texts, False


def _unwrap_single(results: List[Any], single_input: bool) -> Any:
    if single_input:
        return results[0]
    return results


def _normalize_texts(
    texts: List[str],
    normalizer: Optional[TextNormalizer],
    normalize: bool,
) -> List[str]:
    if not (normalizer and normalize):
        return texts
    return [normalizer.normalize(text) for text in texts]


def _normalize_training_data(
    training_data: List[dict],
    normalizer: Optional[TextNormalizer],
    normalize: bool,
) -> List[dict]:
    if not (normalizer and normalize):
        return training_data
    return [
        {"text": normalizer.normalize(item["text"]), "label": item["label"]}
        for item in training_data
    ]


def _flatten_entity_texts(
    entities: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    entity_texts: List[str] = []
    entity_ids: List[str] = []
    for entity in entities:
        entity_texts.append(entity["name"])
        entity_ids.append(entity["id"])
        for alias in entity.get("aliases", []):
            entity_texts.append(alias)
            entity_ids.append(entity["id"])
    return entity_texts, entity_ids


class Matcher:
    """
    Unified entity matcher with smart auto-selection.

    Automatically chooses the best matching strategy:
    - No training data → zero-shot (embedding similarity)
    - < 3 examples/entity → head-only training (~30s)
    - ≥ 3 examples/entity → full training (~3min)

    Example:
        matcher = Matcher(entities=[
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
        ])

        # Zero-shot mode (no training)
        matcher.fit()
        result = matcher.match("America")  # {"id": "US", "score": 0.95}

        # With training data → auto-detects training type
        training_data = [
            {"text": "Germany", "label": "DE"},
            {"text": "USA", "label": "US"},
        ]
        matcher.fit(training_data)

        # Explicit mode override
        matcher.fit(training_data, mode="full")
    """

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        model: str = "default",
        threshold: float = 0.7,
        normalize: bool = True,
        mode: Optional[str] = None,
        blocking_strategy: Optional[Any] = None,
        reranker_model: str = "default",
        verbose: bool = False,
    ):
        """
        Initialize the unified Matcher.

        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys.
                Optional 'aliases' key for alternative names.
            model: Model name or alias (e.g., 'default', 'bge-base', 'mpnet').
                Use 'default' for the recommended model.
            threshold: Minimum confidence threshold (0-1) for matches.
            normalize: Whether to apply text normalization.
            mode: Explicit mode selection. One of:
                - None (default): Auto-detect based on training data
                - 'auto': Same as None, auto-detect
                - 'zero-shot': No training, embedding similarity only
                - 'head-only': Train classifier head only (fast, ~30s)
                - 'full': Full SetFit training (accurate, ~3min)
                - 'hybrid': Blocking + retrieval + reranking pipeline
            blocking_strategy: Optional blocking strategy used by hybrid mode.
            reranker_model: Reranker model alias or name used by hybrid mode.
            verbose: Whether to print detailed logging information.
        """
        validate_entities(entities)
        validate_threshold(threshold)

        self.entities = entities
        self.model_name = resolve_model_alias(model)
        self._requested_model = model
        self._training_model_name = resolve_training_model_alias(model)
        self.threshold = threshold
        self.normalize = normalize
        self.mode = mode
        self.blocking_strategy = blocking_strategy
        self.reranker_model = reranker_model
        self._verbose = verbose

        # Auto-detect mode if not explicitly set
        if mode is None or mode == "auto":
            self._training_mode = "auto"
        elif mode in ("zero-shot", "head-only", "full", "hybrid"):
            self._training_mode = mode
        else:
            raise ModeError(
                f"Invalid mode: {mode}",
                invalid_mode=mode,
            )

        # Lazy initialization of underlying matchers
        # Using Any type to avoid forward reference issues since Matcher is defined before EntityMatcher/EmbeddingMatcher
        self._embedding_matcher: Optional[Any] = None
        self._entity_matcher: Optional[Any] = None
        self._hybrid_matcher: Optional[Any] = None
        self._has_training_data = False
        self._active_matcher: Optional[Any] = None
        self._detected_mode: Optional[str] = None  # Store auto-detected mode

    @property
    def embedding_matcher(self) -> Any:
        """Lazy initialization of EmbeddingMatcher."""
        if self._embedding_matcher is None:
            # Reference classes defined later in this module
            self._embedding_matcher = EmbeddingMatcher(
                entities=self.entities,
                model_name=self.model_name,
                threshold=self.threshold,
                normalize=self.normalize,
            )
        return self._embedding_matcher

    @property
    def entity_matcher(self) -> Any:
        """Lazy initialization of EntityMatcher."""
        if self._entity_matcher is None:
            # Reference classes defined later in this module
            self._entity_matcher = EntityMatcher(
                entities=self.entities,
                model_name=self._training_model_name,
                threshold=self.threshold,
                normalize=self.normalize,
            )
        return self._entity_matcher

    @property
    def hybrid_matcher(self) -> Any:
        """Lazy initialization of HybridMatcher."""
        if self._hybrid_matcher is None:
            from .hybrid import HybridMatcher

            self._hybrid_matcher = HybridMatcher(
                entities=self.entities,
                blocking_strategy=self.blocking_strategy,
                retriever_model=self.model_name,
                reranker_model=self.reranker_model,
                normalize=self.normalize,
            )
        return self._hybrid_matcher

    def _log(self, message: str, level: str = "info") -> None:
        """Log message if verbose mode is enabled.

        Args:
            message: The message to log
            level: Log level ('info', 'warning', 'debug')
        """
        if self._verbose:
            prefix = {
                "info": "[INFO]",
                "warning": "[WARN]",
                "debug": "[DEBUG]",
            }.get(level, "[LOG]")
            print(f"{prefix} {message}")

    def _detect_training_mode(self, training_data: Optional[List[dict]]) -> str:
        """
        Auto-detect appropriate training mode based on data.

        Rules:
        - No training data → zero-shot
        - < 3 examples per entity → head-only (fast)
        - ≥ 3 examples per entity → full training

        Args:
            training_data: Training examples with 'text' and 'label' keys.

        Returns:
            Detected mode: 'zero-shot', 'head-only', or 'full'.
        """
        if training_data is None:
            detected = "zero-shot"
        else:
            # Count examples per entity
            entity_counts = defaultdict(int)
            for item in training_data:
                entity_counts[item["label"]] += 1

            examples_per_entity = list(entity_counts.values())

            # Check if we have enough examples for full training
            detected = "head-only" if max(examples_per_entity) < 3 else "full"

        # Store for transparency
        self._detected_mode = detected
        return detected

    def _select_matcher(self) -> Any:
        """Select the appropriate underlying matcher based on current mode."""
        mode = self._training_mode

        if mode == "zero-shot":
            return self.embedding_matcher
        elif mode in ("head-only", "full"):
            return self.entity_matcher
        elif mode == "hybrid":
            return self.hybrid_matcher
        elif mode == "auto":
            # In auto mode, default to zero-shot until fit() is called
            return self.embedding_matcher
        else:
            raise ModeError(f"Unknown mode: {mode}", invalid_mode=mode)

    def fit(
        self,
        training_data: Optional[List[dict]] = None,
        mode: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> "Matcher":
        """
        Train the matcher if needed. Auto-detects mode if not specified.

        Args:
            training_data: Optional training examples. Each dict must have:
                - 'text': The input text to match
                - 'label': The entity ID to match to
            mode: Override auto-detection. One of:
                - None: Use auto-detection based on training_data
                - 'zero-shot': Skip training, use embedding similarity
                - 'head-only': Train classifier head only (fast)
                - 'full': Full SetFit training (accurate)
            show_progress: Whether to show progress bar during training.
                Requires tqdm installed (pip install semantic-matcher[jupyter]).
            **kwargs: Additional arguments passed to training:
                - num_epochs: Number of training epochs (default: 4)
                - batch_size: Training batch size (default: 16)

        Returns:
            self, for method chaining.

        Example:
            # Zero-shot (no training)
            matcher.fit()

            # Auto-detect training mode
            matcher.fit(training_data)

            # Explicit full training
            matcher.fit(training_data, mode="full")
        """
        self._log(f"Starting fit with mode: {self._training_mode}")

        # Determine the mode to use
        if mode is not None:
            if mode not in ("zero-shot", "head-only", "full", "hybrid"):
                raise ModeError(
                    f"Invalid mode: {mode}",
                    invalid_mode=mode,
                )
            self._training_mode = mode
        elif training_data is not None and self._training_mode == "auto":
            # Auto-detect mode based on training data
            self._training_mode = self._detect_training_mode(training_data)
            self._log(f"Auto-detected mode: {self._training_mode}", "debug")
        elif training_data is None and self._training_mode == "auto":
            # No training data, use zero-shot
            self._training_mode = "zero-shot"

        # Try to import tqdm for progress messages
        if show_progress:
            try:
                from tqdm.auto import tqdm

                # Display mode if it was auto-detected
                if self._detected_mode and self._training_mode != "zero-shot":
                    tqdm.write(f"Auto-detected mode: {self._detected_mode}")
            except ImportError:
                # tqdm not available, silently disable
                show_progress = False

        if self._training_mode == "hybrid":
            if training_data is not None:
                self._log(
                    "Ignoring training_data in hybrid mode; hybrid matching is inference-only",
                    "warning",
                )
            self._log("Initializing hybrid pipeline")
            self._active_matcher = self.hybrid_matcher
            self._has_training_data = False
            return self

        # Build index for zero-shot mode
        if self._training_mode == "zero-shot":
            self._log("Building zero-shot index (no training required)")
            self.embedding_matcher.build_index()
            self._active_matcher = self.embedding_matcher
            return self

        # Train the entity matcher for head-only or full training
        if training_data is None:
            raise ValidationError(
                "training_data is required for modes 'head-only' and 'full'",
                suggestion="Provide training_data or use mode='zero-shot' for matching without training",
            )

        # For now, both head-only and full use the same training
        # SetFit handles head-only vs full via different internal settings
        # In the future, we can add explicit head_only parameter to EntityMatcher.train()
        if self._training_mode in ("head-only", "full"):
            self._log(f"Training in {self._training_mode} mode")
            if not supports_training_model(self._requested_model):
                self._log(
                    "Requested model is retrieval-only; falling back to "
                    f"{self._training_model_name} for training",
                    "warning",
                )
            self.entity_matcher.train(
                training_data, show_progress=show_progress, **kwargs
            )
            self._active_matcher = self.entity_matcher
            self._has_training_data = True
            self._log("Training complete")
            return self

        raise ModeError(
            f"Unknown mode: {self._training_mode}",
            invalid_mode=self._training_mode,
        )

    def match(
        self,
        texts: TextInput,
        top_k: int = 1,
        **kwargs,
    ) -> Any:
        """
        Match texts against entities using the active strategy.

        Args:
            texts: Query text(s) to match. Can be a string or list of strings.
            top_k: Number of top results to return.
            **kwargs: Additional arguments for specific matchers:
                - candidates: Optional list of candidate entities to restrict search
                - batch_size: Batch size for encoding queries

        Returns:
            Matched entity/ies with scores:
            - Single input, top_k=1: Dict or None
            - Single input, top_k>1: List of dicts or empty list
            - Multiple inputs: List of results (one per input)

        Example:
            matcher = Matcher(entities=entities)
            matcher.fit()

            # Single match
            result = matcher.match("America")  # {"id": "US", "score": 0.95}

            # Top-k matches
            results = matcher.match("America", top_k=3)  # [{"id": "US", ...}, ...]

            # Batch matches
            results = matcher.match(["America", "Germany"])  # [..., ...]
        """
        if self._active_matcher is None:
            # Auto-fit if not yet fitted
            self.fit()

        # Route to appropriate matcher based on mode
        if self._training_mode == "hybrid":
            return self._match_hybrid(texts, top_k=top_k, **kwargs)
        if self._training_mode == "zero-shot":
            return self.embedding_matcher.match(texts, top_k=top_k, **kwargs)
        else:
            # Trained mode currently supports candidate filtering, but not batch_size.
            return self.entity_matcher.match(
                texts,
                candidates=kwargs.get("candidates"),
                top_k=top_k,
            )

    def predict(
        self,
        texts: TextInput,
        **kwargs,
    ) -> Union[Optional[str], List[Optional[str]]]:
        """
        Predict entity IDs for input texts.

        Convenience method that returns only entity IDs (not full match dicts).

        Args:
            texts: Query text(s) to match.
            **kwargs: Additional arguments passed to match().

        Returns:
            Entity ID(s) or None if no match above threshold.
        """
        results = self.match(texts, top_k=1, **kwargs)

        # Extract entity IDs from results
        if isinstance(results, list):
            return [r["id"] if r else None for r in results]
        else:
            return results["id"] if results else None

    def set_threshold(self, threshold: float) -> "Matcher":
        """Set the matching threshold.

        Args:
            threshold: New threshold value (0-1).

        Returns:
            self, for method chaining.
        """
        self.threshold = validate_threshold(threshold)

        # Update threshold in underlying matchers if they exist
        if self._embedding_matcher:
            self._embedding_matcher.threshold = self.threshold
        if self._entity_matcher:
            self._entity_matcher.threshold = self.threshold

        return self

    def _match_hybrid(self, texts: TextInput, top_k: int = 1, **kwargs) -> Any:
        """Run hybrid matching while preserving unified Matcher return semantics."""
        blocking_top_k = kwargs.get("blocking_top_k", 1000)
        retrieval_top_k = kwargs.get("retrieval_top_k", max(50, top_k))
        final_top_k = kwargs.get("final_top_k", top_k)
        n_jobs = kwargs.get("n_jobs", -1)
        chunk_size = kwargs.get("chunk_size")

        texts, single_input = _coerce_texts(texts)

        if single_input:
            raw_results = self.hybrid_matcher.match(
                texts[0],
                blocking_top_k=blocking_top_k,
                retrieval_top_k=retrieval_top_k,
                final_top_k=final_top_k,
            )
            return self._format_hybrid_results(raw_results, top_k=top_k)

        raw_results = self.hybrid_matcher.match_bulk(
            texts,
            blocking_top_k=blocking_top_k,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )
        return [
            self._format_hybrid_results(results, top_k=top_k) for results in raw_results
        ]

    def _format_hybrid_results(
        self, results: Optional[List[Dict[str, Any]]], top_k: int
    ) -> Any:
        """Apply threshold filtering and shape normalization to hybrid results."""
        filtered = []
        for result in results or []:
            if result.get("score", 0.0) < self.threshold:
                continue
            filtered.append(result)

        if top_k == 1:
            return filtered[0] if filtered else None
        return filtered[:top_k]

    def get_training_info(self) -> Dict[str, Any]:
        """Get information about current training configuration.

        Returns:
            Dict with keys:
                - mode: Current training mode
                - detected_mode: Auto-detected mode (if applicable)
                - is_trained: Whether the matcher is trained
                - active_matcher: Name of the active matcher class
                - has_training_data: Whether training data was provided
                - threshold: Current threshold value
        """
        return {
            "mode": self._training_mode,
            "detected_mode": self._detected_mode,
            "is_trained": self._active_matcher is not None,
            "active_matcher": (
                type(self._active_matcher).__name__ if self._active_matcher else None
            ),
            "has_training_data": self._has_training_data,
            "threshold": self.threshold,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the matcher.

        Returns:
            Dict with matcher statistics including entity count,
            configuration, and state information.
        """
        stats = {
            "num_entities": len(self.entities),
            "model_name": self.model_name,
            "threshold": self.threshold,
            "normalize": self.normalize,
            "training_mode": self._training_mode,
            "is_trained": self._active_matcher is not None,
        }

        # Add entity-specific stats if available
        if hasattr(self, "_embedding_matcher") and self._embedding_matcher:
            stats["has_embeddings"] = self._embedding_matcher.embeddings is not None

        if hasattr(self, "_entity_matcher") and self._entity_matcher:
            classifier = getattr(self._entity_matcher, "classifier", None)
            stats["classifier_trained"] = (
                getattr(classifier, "is_trained", self._entity_matcher.is_trained)
                if classifier is not None
                else False
            )

        return stats

    def explain_match(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Explain matching results for debugging.

        Args:
            query: The input text to match
            top_k: Number of top candidates to show

        Returns:
            Dict with:
                - query: The original query
                - query_normalized: Normalized query if enabled
                - matched: Whether a match was found above threshold
                - best_match: Top result (entity_id, score, etc.)
                - top_k: List of top k candidates with scores
                - threshold: Current threshold
                - mode: Current training mode
        """
        if not self._active_matcher:
            raise TrainingError(
                "Matcher not ready. Call fit() first.",
                details={"mode": self._training_mode},
            )

        # Get all candidates with scores by temporarily lowering matcher threshold.
        original_threshold = self.threshold
        self.set_threshold(0.0)
        try:
            results = self.match(query, top_k=top_k)
        finally:
            self.set_threshold(original_threshold)

        if results is None:
            result_list = []
        elif isinstance(results, list):
            result_list = results
        else:
            result_list = [results]

        # Normalize query if applicable
        query_normalized = None
        if self.normalize:
            normalizer = TextNormalizer()
            query_normalized = normalizer.normalize(query)

        # Get best match and check if it passes threshold
        best = result_list[0] if result_list else None
        matched = best and best.get("score", 0) >= self.threshold

        return {
            "query": query,
            "query_normalized": query_normalized,
            "matched": matched,
            "best_match": best,
            "top_k": result_list,
            "threshold": self.threshold,
            "mode": self._training_mode,
        }

    def diagnose(self, query: str) -> Dict[str, Any]:
        """Run diagnostics on a query to help debug issues.

        Args:
            query: Input text to diagnose

        Returns:
            Dict with diagnostic information and suggestions.
        """
        diagnosis = {
            "query": query,
            "matcher_ready": self._active_matcher is not None,
            "active_matcher": (
                type(self._active_matcher).__name__ if self._active_matcher else None
            ),
        }

        if not self._active_matcher:
            diagnosis["issue"] = "Matcher not ready"
            diagnosis["suggestion"] = "Call matcher.fit() to initialize the matcher"
            return diagnosis

        # Try the match
        try:
            explanation = self.explain_match(query, top_k=3)
            diagnosis.update(explanation)

            # Add suggestions based on results
            if not explanation["matched"]:
                if explanation["best_match"]:
                    score = explanation["best_match"].get("score", 0)
                    threshold = explanation["threshold"]
                    diagnosis["issue"] = (
                        f"Score {score:.2f} below threshold {threshold}"
                    )
                    suggested_threshold = max(0.1, threshold - 0.1)
                    diagnosis["suggestion"] = (
                        f"Lower threshold with matcher.set_threshold({suggested_threshold:.1f}) "
                        f"or add more training examples"
                    )
                else:
                    diagnosis["issue"] = "No candidates found"
                    diagnosis["suggestion"] = (
                        "Check entity data and text normalization. "
                        "Ensure entities have relevant names/aliases."
                    )
        except Exception as e:
            diagnosis["error"] = str(e)
            diagnosis["suggestion"] = "Check input format and entity configuration"

        return diagnosis

    def __repr__(self) -> str:
        """Simple string representation."""
        status = "trained" if self._active_matcher else "untrained"
        return f"Matcher(mode={self._training_mode}, status={status})"


class EntityMatcher:
    """SetFit-based entity matching with optional text normalization."""

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        threshold: float = 0.7,
        normalize: bool = True,
    ):
        validate_entities(entities)
        validate_model_name(model_name)

        self.entities = entities
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
        self.normalize = normalize

        self.normalizer = TextNormalizer() if normalize else None
        self.classifier: Optional[SetFitClassifier] = None
        self.is_trained = False

    def _get_training_data(self, training_data: List[dict]) -> List[dict]:
        return _normalize_training_data(training_data, self.normalizer, self.normalize)

    def train(
        self,
        training_data: List[dict],
        num_epochs: int = 4,
        batch_size: int = 16,
        show_progress: bool = True,
    ):
        normalized_data = self._get_training_data(training_data)
        labels = list(dict.fromkeys(item["label"] for item in normalized_data))

        self.classifier = SetFitClassifier(
            labels=labels,
            model_name=self.model_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        self.classifier.train(
            normalized_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        self.is_trained = True

    def predict(self, texts: TextInput) -> Union[Optional[str], List[Optional[str]]]:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("Model not trained. Call train() first.")

        matches = self.match(texts, top_k=1)

        if isinstance(matches, list):
            return [match["id"] if match else None for match in matches]
        return matches["id"] if matches else None

    def match(
        self,
        texts: TextInput,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 1,
    ) -> Any:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("Model not trained. Call train() first.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        texts, single_input = _coerce_texts(texts)
        texts = _normalize_texts(texts, self.normalizer, self.normalize)
        entity_lookup = {entity["id"]: entity for entity in self.entities}
        candidate_ids = None
        if candidates is not None:
            candidate_ids = {candidate["id"] for candidate in candidates}

        results = []
        for text in texts:
            try:
                proba = self.classifier.predict_proba(text)
                ranked_matches = sorted(
                    zip(self.classifier.labels, proba),
                    key=lambda item: item[1],
                    reverse=True,
                )
                matches = []
                for label, score in ranked_matches:
                    score = float(score)
                    if score < self.threshold:
                        continue
                    if candidate_ids is not None and label not in candidate_ids:
                        continue
                    entity = entity_lookup.get(label, {})
                    matches.append(
                        {
                            "id": label,
                            "score": score,
                            "text": entity.get("name", ""),
                        }
                    )
                    if len(matches) >= top_k:
                        break
                if top_k == 1:
                    results.append(matches[0] if matches else None)
                else:
                    results.append(matches)
            except ValueError:
                results.append(None if top_k == 1 else [])

        return _unwrap_single(results, single_input)


class EmbeddingMatcher:
    """Embedding-based similarity matching without training."""

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        threshold: float = 0.7,
        normalize: bool = True,
        embedding_dim: Optional[int] = None,
        cache: Optional[ModelCache] = None,
    ):
        """
        Initialize the embedding matcher.

        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys
            model_name: Name of the sentence-transformer model
            threshold: Minimum similarity score threshold (0-1)
            normalize: Whether to normalize text
            embedding_dim: Optional dimension for Matryoshka embeddings
            cache: Optional ModelCache instance. If None, uses global default cache.
        """
        validate_entities(entities)
        validate_model_name(model_name)

        self.entities = entities
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
        self.normalize = normalize
        self.embedding_dim = embedding_dim  # Matryoshka support

        self.normalizer = TextNormalizer() if normalize else None
        self.cache = cache if cache is not None else get_default_cache()
        self.model: Optional[EmbeddingModel] = None
        self.entity_texts: List[str] = []
        self.entity_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, batch_size: Optional[int] = None):
        """
        Build the embedding index from entities.

        Args:
            batch_size: Batch size for encoding. None = use model's default.
        """
        resolved_name = resolve_model_alias(self.model_name)

        if is_static_embedding_model(resolved_name):
            # Use StaticEmbeddingBackend for static models
            from ..backends.static_embedding import StaticEmbeddingBackend

            self.model = StaticEmbeddingBackend(
                resolved_name, embedding_dim=self.embedding_dim
            )
        else:
            # Use cache to get or load the SentenceTransformer model
            self.model = self.cache.get_or_load(
                resolved_name, lambda: SentenceTransformer(resolved_name)
            )

        # Validate embedding_dim if provided
        if self.embedding_dim is not None:
            # Get actual model embedding dimension
            if isinstance(self.model, SentenceTransformer):
                actual_dim = self.model.get_sentence_embedding_dimension()
            elif hasattr(self.model, "embedding_dimension"):
                actual_dim = self.model.embedding_dimension
            else:
                actual_dim = None

            # Validate against model's actual dimension
            if actual_dim is not None and self.embedding_dim > actual_dim:
                raise ValueError(
                    f"embedding_dim ({self.embedding_dim}) cannot exceed "
                    f"model embedding dimension ({actual_dim})"
                )

            # Validate positive value
            if self.embedding_dim <= 0:
                raise ValueError(
                    f"embedding_dim must be positive, got {self.embedding_dim}"
                )

        self.entity_texts, self.entity_ids = _flatten_entity_texts(self.entities)
        self.entity_texts = _normalize_texts(
            self.entity_texts, self.normalizer, self.normalize
        )

        if batch_size is not None:
            self.embeddings = self.model.encode(
                self.entity_texts, batch_size=batch_size
            )
        else:
            self.embeddings = self.model.encode(self.entity_texts)

        # Convert to numpy array if the model returns a list
        if isinstance(self.embeddings, list):
            self.embeddings = np.array(self.embeddings)

        # Matryoshka embedding support: truncate to specified dimension
        if (
            self.embedding_dim is not None
            and self.embeddings.shape[1] > self.embedding_dim
        ):
            self.embeddings = self.embeddings[:, : self.embedding_dim]

    def match(
        self,
        texts: TextInput,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 1,
        batch_size: Optional[int] = None,
    ) -> Any:
        """
        Match texts against indexed entities.

        Args:
            texts: Query text(s) to match
            candidates: Optional list of candidate entities to restrict search
            top_k: Number of top results to return
            batch_size: Batch size for encoding queries. None = use model's default.

        Returns:
            Matched entity/ies with scores
        """
        if self.embeddings is None or self.model is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        texts, single_input = _coerce_texts(texts)
        texts = _normalize_texts(texts, self.normalizer, self.normalize)
        entity_lookup = {entity["id"]: entity for entity in self.entities}

        # Use provided candidates or all entities
        if candidates is not None:
            candidate_ids = {c["id"] for c in candidates}
            candidate_indices = [
                i for i, eid in enumerate(self.entity_ids) if eid in candidate_ids
            ]
        else:
            candidate_indices = list(range(len(self.entity_ids)))

        if not candidate_indices:
            empty = None if top_k == 1 else []
            return empty if single_input else [empty for _ in texts]

        candidate_embeddings = self.embeddings[candidate_indices]
        candidate_ids_list = [self.entity_ids[i] for i in candidate_indices]

        if batch_size is not None:
            query_embeddings = self.model.encode(texts, batch_size=batch_size)
        else:
            query_embeddings = self.model.encode(texts)

        # Convert to numpy array if the model returns a list
        if isinstance(query_embeddings, list):
            query_embeddings = np.array(query_embeddings)

        # Ensure both query and candidate embeddings use same dimension
        # Use the smaller of: model's output dim or configured embedding_dim
        effective_dim = (
            self.embedding_dim
            if self.embedding_dim is not None
            else query_embeddings.shape[1]
        )

        # Truncate query embeddings if needed
        if query_embeddings.shape[1] > effective_dim:
            query_embeddings = query_embeddings[:, :effective_dim]

        # Ensure candidate embeddings match (may have been truncated in build_index)
        if candidate_embeddings.shape[1] > effective_dim:
            candidate_embeddings = candidate_embeddings[:, :effective_dim]

        similarities = cosine_similarity(query_embeddings, candidate_embeddings)

        results = []
        for sim_row in similarities:
            sorted_indices = np.argsort(sim_row)[::-1]
            matches = []
            seen_ids = set()
            for idx in sorted_indices:
                score = sim_row[idx]
                if score < self.threshold:
                    continue
                entity_id = candidate_ids_list[idx]
                if entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)
                entity = entity_lookup.get(entity_id, {})
                matches.append(
                    {
                        "id": entity_id,
                        "score": float(score),
                        "text": entity.get(
                            "text", self.entity_texts[candidate_indices[idx]]
                        ),
                    }
                )
                if len(matches) >= top_k:
                    break

            if top_k == 1:
                results.append(matches[0] if matches else None)
            else:
                results.append(matches)

        return _unwrap_single(results, single_input)
