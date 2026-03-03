from typing import Optional, Union, List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
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
from ..config import resolve_model_alias

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
        """
        validate_entities(entities)
        validate_threshold(threshold)

        self.entities = entities
        self.model_name = resolve_model_alias(model)
        self.threshold = threshold
        self.normalize = normalize
        self.mode = mode

        # Auto-detect mode if not explicitly set
        if mode is None or mode == "auto":
            self._training_mode = "auto"
        elif mode in ("zero-shot", "head-only", "full", "hybrid"):
            self._training_mode = mode
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of: "
                "'auto', 'zero-shot', 'head-only', 'full', 'hybrid'"
            )

        # Lazy initialization of underlying matchers
        # Using Any type to avoid forward reference issues since Matcher is defined before EntityMatcher/EmbeddingMatcher
        self._embedding_matcher: Optional[Any] = None
        self._entity_matcher: Optional[Any] = None
        self._has_training_data = False
        self._active_matcher: Optional[Any] = None

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
                model_name=self.model_name,
                threshold=self.threshold,
                normalize=self.normalize,
            )
        return self._entity_matcher

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
            return "zero-shot"

        # Count examples per entity
        entity_counts = defaultdict(int)
        for item in training_data:
            entity_counts[item["label"]] += 1

        examples_per_entity = list(entity_counts.values())

        # Check if we have enough examples for full training
        if max(examples_per_entity) < 3:
            return "head-only"
        else:
            return "full"

    def _select_matcher(self) -> Any:
        """Select the appropriate underlying matcher based on current mode."""
        mode = self._training_mode

        if mode == "zero-shot":
            return self.embedding_matcher
        elif mode in ("head-only", "full"):
            return self.entity_matcher
        elif mode == "auto":
            # In auto mode, default to zero-shot until fit() is called
            return self.embedding_matcher
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def fit(
        self,
        training_data: Optional[List[dict]] = None,
        mode: Optional[str] = None,
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
        # Determine the mode to use
        if mode is not None:
            if mode not in ("zero-shot", "head-only", "full"):
                raise ValueError(
                    f"Invalid mode: {mode}. Must be one of: "
                    "'zero-shot', 'head-only', 'full'"
                )
            self._training_mode = mode
        elif training_data is not None and self._training_mode == "auto":
            # Auto-detect mode based on training data
            self._training_mode = self._detect_training_mode(training_data)
        elif training_data is None and self._training_mode == "auto":
            # No training data, use zero-shot
            self._training_mode = "zero-shot"

        # Build index for zero-shot mode
        if self._training_mode == "zero-shot":
            self.embedding_matcher.build_index()
            self._active_matcher = self.embedding_matcher
            return self

        # Train the entity matcher for head-only or full training
        if training_data is None:
            raise ValueError(
                "training_data is required for modes 'head-only' and 'full'"
            )

        # For now, both head-only and full use the same training
        # SetFit handles head-only vs full via different internal settings
        # In the future, we can add explicit head_only parameter to EntityMatcher.train()
        if self._training_mode in ("head-only", "full"):
            self.entity_matcher.train(training_data, **kwargs)
            self._active_matcher = self.entity_matcher
            self._has_training_data = True
            return self

        raise ValueError(f"Unknown mode: {self._training_mode}")

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
        if self._training_mode == "zero-shot":
            return self.embedding_matcher.match(texts, top_k=top_k, **kwargs)
        else:
            # EntityMatcher returns entity IDs, we need to convert to match format
            predictions = self.entity_matcher.predict(texts)
            texts_list, single_input = _coerce_texts(texts)

            # Build result dicts with scores
            results = []
            entity_lookup = {entity["id"]: entity for entity in self.entities}

            for i, pred_id in enumerate(predictions if isinstance(predictions, list) else [predictions]):
                if pred_id is None:
                    results.append(None if top_k == 1 else [])
                    continue

                # For EntityMatcher, we return the entity with a placeholder score
                # (EntityMatcher doesn't expose scores directly)
                entity = entity_lookup.get(pred_id, {})
                result = {
                    "id": pred_id,
                    "score": 1.0,  # EntityMatcher doesn't expose raw scores
                    "text": entity.get("name", ""),
                }
                results.append(result if top_k == 1 else [result])

            return _unwrap_single(results, single_input)

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
    ):
        normalized_data = self._get_training_data(training_data)
        labels = list(set(item["label"] for item in normalized_data))

        self.classifier = SetFitClassifier(
            labels=labels,
            model_name=self.model_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        self.classifier.train(
            normalized_data, num_epochs=num_epochs, batch_size=batch_size
        )
        self.is_trained = True

    def predict(self, texts: TextInput) -> Union[Optional[str], List[Optional[str]]]:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("Model not trained. Call train() first.")

        texts, single_input = _coerce_texts(texts)
        texts = _normalize_texts(texts, self.normalizer, self.normalize)

        predictions = []
        for text in texts:
            try:
                proba = self.classifier.predict_proba(text)
                if float(np.max(proba)) < self.threshold:
                    predictions.append(None)
                    continue
                pred = self.classifier.predict(text)
                predictions.append(pred)
            except ValueError:
                predictions.append(None)

        return _unwrap_single(predictions, single_input)


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
        self.model: Optional[SentenceTransformer] = None
        self.entity_texts: List[str] = []
        self.entity_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, batch_size: Optional[int] = None):
        """
        Build the embedding index from entities.

        Args:
            batch_size: Batch size for encoding. None = use model's default.
        """
        # Use cache to get or load the model
        self.model = self.cache.get_or_load(
            self.model_name, lambda: SentenceTransformer(self.model_name)
        )

        # Validate embedding_dim if provided
        if self.embedding_dim is not None:
            # Get actual model embedding dimension
            actual_dim = self.model.get_sentence_embedding_dimension()

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
