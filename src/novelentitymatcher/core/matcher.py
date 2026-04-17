import asyncio
import inspect
import os
import platform
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .matching_strategy import MatchingStrategy

import numpy as np
from sentence_transformers import SentenceTransformer

# Enable CPU fallback for unsupported MPS ops before torch/sentence-transformers import.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


from .embedding_matcher import EmbeddingMatcher
from .matcher_components import MatcherComponentFactory
from .matcher_runtime import MatcherRuntimeState
from .matcher_shared import (
    TextInput,
    coerce_texts,
    extract_top_prediction_metadata,
    normalize_texts,
    normalize_training_data,
    resolve_threshold,
    unwrap_single,
)
from .matching_strategy import MatcherFacade
from ..pipeline.match_result import build_match_result_with_metadata
from .normalizer import TextNormalizer
from ..config import (
    is_bert_model,
    supports_training_model,
)
from ..exceptions import ModeError, TrainingError, ValidationError
from ..utils.logging_config import configure_logging, get_logger
from ..utils.validation import (
    validate_entities,
    validate_model_name,
    validate_threshold,
)

if TYPE_CHECKING:
    from .async_utils import AsyncExecutor
    from .bert_classifier import BERTClassifier
    from .classifier import SetFitClassifier

EmbeddingModel = SentenceTransformer
__all__ = ["EmbeddingMatcher", "EmbeddingModel", "Matcher"]

# Backwards-compatible aliases for internal helpers that some callers may import.
_coerce_texts = coerce_texts
_extract_top_prediction_metadata = extract_top_prediction_metadata
_resolve_threshold = resolve_threshold


class _EntityMatcher:
    """SetFit-based or BERT-based entity matching with optional text normalization."""

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        threshold: float = 0.7,
        normalize: bool = True,
        classifier_type: str = "setfit",
    ):
        validate_entities(entities)
        validate_model_name(model_name)

        self.entities = entities
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
        self.normalize = normalize
        self.classifier_type = classifier_type

        self.normalizer = TextNormalizer() if normalize else None
        self.logger = get_logger(__name__)
        self.classifier: Optional[Union[SetFitClassifier, BERTClassifier]] = None
        self.is_trained = False
        self._async_executor: Optional[Any] = None
        self._reference_texts: List[str] = []
        self._reference_labels: List[str] = []
        self._reference_embeddings: Optional[np.ndarray] = None

    def _ensure_async_executor(self):
        if self._async_executor is None:
            from .async_utils import AsyncExecutor

            self._async_executor = AsyncExecutor()
        return self._async_executor

    def _get_training_data(self, training_data: List[dict]) -> List[dict]:
        return normalize_training_data(training_data, self.normalizer, self.normalize)

    def train(
        self,
        training_data: List[dict],
        num_epochs: int = 4,
        batch_size: int = 16,
        show_progress: bool = True,
        weight_decay: float = 0.01,
        head_c: float = 1.0,
        num_iterations: int = 5,
        pca_dims: Optional[int] = None,
        skip_body_training: bool = False,
    ):
        normalized_data = self._get_training_data(training_data)
        labels = list(dict.fromkeys(item["label"] for item in normalized_data))

        if self.classifier_type == "bert":
            from .bert_classifier import BERTClassifier

            self.classifier = BERTClassifier(
                labels=labels,
                model_name=self.model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
        else:
            from .classifier import SetFitClassifier

            self.classifier = SetFitClassifier(
                labels=labels,
                model_name=self.model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                weight_decay=weight_decay,
                head_c=head_c,
                num_iterations=num_iterations,
                pca_dims=pca_dims,
                skip_body_training=skip_body_training,
            )

        self.classifier.train(
            normalized_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        self.is_trained = True
        self._reference_texts = [item["text"] for item in normalized_data]
        self._reference_labels = [item["label"] for item in normalized_data]
        self._reference_embeddings = None

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
        threshold_override: Optional[float] = None,
    ) -> Any:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("Model not trained. Call train() first.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        texts, single_input = coerce_texts(texts)
        texts = normalize_texts(texts, self.normalizer, self.normalize)
        entity_lookup = {entity["id"]: entity for entity in self.entities}
        candidate_ids = None
        effective_threshold = resolve_threshold(threshold_override, self.threshold)
        if candidates is not None:
            candidate_ids = {candidate["id"] for candidate in candidates}

        results: List[Any] = []
        for text in texts:
            try:
                proba = self.classifier.predict_proba(text)
                ranked_matches = sorted(
                    zip(self.classifier.labels, proba),
                    key=lambda item: item[1],
                    reverse=True,
                )
                matches: List[Dict[str, Any]] = []
                for label, score in ranked_matches:
                    score = float(score)
                    if score < effective_threshold:
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
            except (ValueError, RuntimeError) as exc:
                self.logger.warning("Match prediction failed for text sample: %s", exc)
                results.append(None if top_k == 1 else [])

        return unwrap_single(results, single_input)

    async def train_async(
        self,
        training_data: List[dict],
        num_epochs: int = 4,
        batch_size: int = 16,
        show_progress: bool = True,
    ):
        await self._ensure_async_executor().run_in_thread(
            self.train,
            training_data,
            num_epochs,
            batch_size,
            show_progress,
        )

    async def match_async(
        self,
        texts: TextInput,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 1,
        threshold_override: Optional[float] = None,
    ) -> Any:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError(
                "Model not trained. Call train() or train_async() first."
            )

        return await self._ensure_async_executor().run_in_thread(
            self.match,
            texts=texts,
            candidates=candidates,
            top_k=top_k,
            threshold_override=threshold_override,
        )

    async def predict_async(
        self, texts: TextInput
    ) -> Union[Optional[str], List[Optional[str]]]:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError(
                "Model not trained. Call train() or train_async() first."
            )

        matches = await self.match_async(texts, top_k=1)
        if isinstance(matches, list):
            return [match["id"] if match else None for match in matches]
        return matches["id"] if matches else None

    def get_reference_corpus(self, encoder: Optional[Any] = None) -> Dict[str, Any]:
        if not self._reference_texts or not self._reference_labels:
            raise RuntimeError(
                "No reference corpus available. Train the matcher before novelty detection."
            )

        if self._reference_embeddings is None:
            encode_model = encoder
            if encode_model is None and self.classifier is not None:
                encode_model = getattr(self.classifier, "model", None)
            if encode_model is None or not hasattr(encode_model, "encode"):
                raise RuntimeError(
                    "Could not derive reference embeddings from the trained matcher. "
                    "Provide an embedding-capable encoder or use a matcher mode with "
                    "reference embeddings."
                )
            embeddings = encode_model.encode(self._reference_texts)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            self._reference_embeddings = embeddings

        return {
            "texts": list(self._reference_texts),
            "labels": list(self._reference_labels),
            "embeddings": self._reference_embeddings,
            "source": "training_examples",
        }


class Matcher:
    """
    Unified entity matcher with smart auto-selection.

    Automatically chooses the best matching strategy:
    - No training data -> zero-shot (embedding similarity)
    - < 3 examples/entity -> head-only training (~30s)
    - >= 3 examples/entity -> full training (~3min)
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
        metrics_callback: Optional[Callable] = None,
    ):
        validate_entities(entities)
        validate_threshold(threshold)

        env_verbose = (
            os.getenv("NOVEL_ENTITY_MATCHER_VERBOSE", "false").lower() == "true"
        )
        verbose = verbose or env_verbose

        configure_logging(verbose=verbose)
        self.logger = get_logger(__name__)

        self.entities = entities
        self._runtime_state = MatcherRuntimeState.create(
            model=model,
            threshold=threshold,
            mode=mode,
        )
        self.model_name = self._runtime_state.model_name
        self._requested_model = self._runtime_state.requested_model
        self._training_model_name = self._runtime_state.training_model_name
        self._bert_model_name = self._runtime_state.bert_model_name
        self.threshold = self._runtime_state.threshold
        self.normalize = normalize
        self.mode = mode
        self.blocking_strategy = blocking_strategy
        self.reranker_model = reranker_model
        self._verbose = verbose
        self._metrics_callback = metrics_callback

        self._async_executor: Optional["AsyncExecutor"] = None
        self._async_fit_lock = asyncio.Lock()

        self._training_mode = self._runtime_state.training_mode
        self._components = MatcherComponentFactory(self)
        self._has_training_data = self._runtime_state.has_training_data
        self._active_matcher: Optional[Any] = None
        self._detected_mode: Optional[str] = self._runtime_state.detected_mode

    def _emit_metric(
        self,
        name: str,
        value: float,
        unit: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Emit a metric event if callback is registered.

        Args:
            name: Metric name
            value: Numeric value
            unit: Unit of measurement
            labels: Optional labels dictionary
        """
        if self._metrics_callback is None:
            return

        from ..monitoring.metrics import create_metric

        event = create_metric(name, value, unit, labels)
        self._metrics_callback(event)

    def _ensure_async_executor(self):
        if self._async_executor is None:
            from .async_utils import AsyncExecutor

            self._async_executor = AsyncExecutor()
        return self._async_executor

    def _apply_threshold(self, threshold: float) -> None:
        self.threshold = self._runtime_state.apply_threshold(
            threshold,
            self._components.iter_threshold_targets(),
        )

    @staticmethod
    def _resolve_threshold(
        threshold_override: Optional[float], default: float
    ) -> float:
        return resolve_threshold(threshold_override, default)

    def _get_strategy(self) -> "MatchingStrategy":
        """Get the matching strategy for the current training mode."""
        from .matching_strategy import StrategyConfig

        config = StrategyConfig(
            threshold=self.threshold,
            model_name=self.model_name,
            training_mode=self._training_mode,
            normalize=self.normalize,
        )
        facade = MatcherFacade(
            embedding_matcher=self.embedding_matcher,
            entity_matcher=self.entity_matcher,
            bert_matcher=self.bert_matcher,
            hybrid_matcher=self.hybrid_matcher,
            config=config,
        )
        return facade.get_strategy()

    def _match_sync_impl(
        self,
        texts: TextInput,
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        if self._training_mode == "hybrid":
            return self._match_hybrid(
                texts,
                top_k=top_k,
                threshold_override=threshold_override,
                **kwargs,
            )
        return self._get_strategy().match(
            texts,
            top_k=top_k,
            threshold_override=threshold_override,
            **kwargs,
        )

    def _match_with_metadata(
        self,
        texts: TextInput,
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ):
        texts_list, single_input = coerce_texts(texts)

        model = self.model
        if model is None:
            raise RuntimeError(
                "return_metadata=True requires an embedding-capable fitted matcher"
            )
        embeddings = model.encode(texts_list)

        metadata_threshold = 0.0 if threshold_override is None else threshold_override
        match_results = self._match_sync_impl(
            texts_list,
            top_k=top_k,
            threshold_override=metadata_threshold,
            **kwargs,
        )
        predictions, confidences = extract_top_prediction_metadata(
            match_results, single_input
        )

        return build_match_result_with_metadata(
            texts=texts_list,
            predictions=predictions,
            confidences=confidences,
            embeddings=embeddings,
            raw_match_results=match_results,
            metadata={
                "top_k": top_k,
                "threshold_override": metadata_threshold,
                "evaluation_threshold": self.threshold
                if threshold_override is None
                else threshold_override,
                "model_name": str(self.model_name),
                "single_input": single_input,
            },
        )

    async def _match_with_metadata_async(
        self,
        texts: TextInput,
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ):
        executor = self._ensure_async_executor()
        texts_list, single_input = coerce_texts(texts)

        model = self.model
        if model is None:
            raise RuntimeError(
                "return_metadata=True requires an embedding-capable fitted matcher"
            )
        embeddings = await executor.run_in_thread(model.encode, texts_list)

        metadata_threshold = 0.0 if threshold_override is None else threshold_override
        match_results = await self._match_async_impl(
            texts_list,
            top_k=top_k,
            threshold_override=metadata_threshold,
            **kwargs,
        )
        predictions, confidences = extract_top_prediction_metadata(
            match_results, single_input
        )

        return build_match_result_with_metadata(
            texts=texts_list,
            predictions=predictions,
            confidences=confidences,
            embeddings=embeddings,
            raw_match_results=match_results,
            metadata={
                "top_k": top_k,
                "threshold_override": metadata_threshold,
                "evaluation_threshold": self.threshold
                if threshold_override is None
                else threshold_override,
                "model_name": str(self.model_name),
                "single_input": single_input,
            },
        )

    async def _match_async_impl(
        self,
        texts: TextInput,
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        if self._training_mode == "hybrid":
            return await self._match_hybrid_async(
                texts,
                top_k=top_k,
                threshold_override=threshold_override,
                **kwargs,
            )
        return await self._get_strategy().match_async(
            texts,
            top_k=top_k,
            threshold_override=threshold_override,
            **kwargs,
        )

    @property
    def embedding_matcher(self) -> Any:
        return self._components.get_embedding_matcher()

    @property
    def _embedding_matcher(self) -> Any:
        return self._components._embedding_matcher

    @_embedding_matcher.setter
    def _embedding_matcher(self, value: Any) -> None:
        self._components._embedding_matcher = value

    @property
    def entity_matcher(self) -> Any:
        return self._components.get_entity_matcher()

    @property
    def _entity_matcher(self) -> Any:
        return self._components._entity_matcher

    @_entity_matcher.setter
    def _entity_matcher(self, value: Any) -> None:
        self._components._entity_matcher = value

    @property
    def bert_matcher(self) -> Any:
        return self._components.get_bert_matcher()

    @property
    def _bert_matcher(self) -> Any:
        return self._components._bert_matcher

    @_bert_matcher.setter
    def _bert_matcher(self, value: Any) -> None:
        self._components._bert_matcher = value

    @property
    def hybrid_matcher(self) -> Any:
        return self._components.get_hybrid_matcher()

    @property
    def _hybrid_matcher(self) -> Any:
        return self._components._hybrid_matcher

    @_hybrid_matcher.setter
    def _hybrid_matcher(self, value: Any) -> None:
        self._components._hybrid_matcher = value

    def get_reference_corpus(self) -> Dict[str, Any]:
        if self._training_mode in ("zero-shot", "hybrid", "auto"):
            matcher = self.embedding_matcher
            if matcher.embeddings is None or matcher.model is None:
                matcher.build_index()
            return {
                "texts": list(matcher.entity_texts),
                "labels": list(matcher.entity_ids),
                "embeddings": matcher.embeddings,
                "source": "entity_embeddings",
            }

        if self._training_mode in ("head-only", "full"):
            return self.entity_matcher.get_reference_corpus()

        if self._training_mode == "bert":
            encoder_matcher = self.embedding_matcher
            if encoder_matcher.model is None:
                encoder_matcher.build_index()
            return self.bert_matcher.get_reference_corpus(encoder=encoder_matcher.model)

        raise RuntimeError(
            f"Reference corpus is not available for matcher mode '{self._training_mode}'"
        )

    def _detect_training_mode(self, training_data: Optional[List[dict]]) -> str:
        if training_data is None:
            detected = "zero-shot"
        else:
            entity_counts: Dict[str, int] = defaultdict(int)
            for item in training_data:
                entity_counts[item["label"]] += 1

            examples_per_entity = list(entity_counts.values())
            min_examples = min(examples_per_entity) if examples_per_entity else 0
            max_examples = max(examples_per_entity) if examples_per_entity else 0
            total_examples = len(training_data)

            if min_examples >= 8 and total_examples >= 100:
                detected = "bert"
            elif max_examples < 3:
                detected = "head-only"
            else:
                detected = "full"

        self._detected_mode = self._runtime_state.set_detected_mode(detected)
        return detected

    def _select_matcher(self) -> Any:
        mode = self._training_mode

        if mode == "zero-shot":
            return self.embedding_matcher
        if mode in ("head-only", "full"):
            return self.entity_matcher
        if mode == "bert":
            return self.bert_matcher
        if mode == "hybrid":
            return self.hybrid_matcher
        if mode == "auto":
            return self.embedding_matcher
        raise ModeError(f"Unknown mode: {mode}", invalid_mode=mode)

    def _resolve_classifier_matcher(self) -> Any:
        return (
            self.bert_matcher if self._training_mode == "bert" else self.entity_matcher
        )

    def fit(
        self,
        training_data: Optional[List[dict]] = None,
        mode: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> "Matcher":
        self.logger.info(f"Starting fit with mode: {self._training_mode}")

        if mode is not None:
            self._training_mode = self._runtime_state.update_training_mode(mode)
        elif training_data is not None and self._training_mode == "auto":
            self._training_mode = self._runtime_state.update_training_mode(
                self._detect_training_mode(training_data)
            )
            self.logger.debug(f"Auto-detected mode: {self._training_mode}")
        elif training_data is None and self._training_mode == "auto":
            self._training_mode = self._runtime_state.update_training_mode("zero-shot")

        if show_progress:
            try:
                from tqdm.auto import tqdm

                if self._detected_mode and self._training_mode != "zero-shot":
                    tqdm.write(f"Auto-detected mode: {self._detected_mode}")
            except ImportError:
                show_progress = False

        if self._training_mode == "hybrid":
            if training_data is not None:
                self.logger.warning(
                    "Ignoring training_data in hybrid mode; hybrid matching is inference-only"
                )
            self.logger.info("Initializing hybrid pipeline")
            self._active_matcher = self.hybrid_matcher
            self._runtime_state.has_training_data = False
            self._has_training_data = False
            return self

        if self._training_mode == "zero-shot":
            self.logger.info("Building zero-shot index (no training required)")
            self.embedding_matcher.build_index()
            self._active_matcher = self.embedding_matcher
            return self

        if training_data is None:
            raise ValidationError(
                "training_data is required for modes 'head-only', 'full', and 'bert'",
                suggestion="Provide training_data or use mode='zero-shot' for matching without training",
            )

        if self._training_mode in ("head-only", "full", "bert"):
            self.logger.info(f"Training in {self._training_mode} mode")

            if self._training_mode == "bert" and not is_bert_model(
                self._requested_model
            ):
                self.logger.warning(
                    f"Using non-BERT model '{self._requested_model}' with bert mode. "
                    "For optimal results, use a BERT-based model."
                )
            elif self._training_mode in (
                "head-only",
                "full",
            ) and not supports_training_model(self._requested_model):
                self.logger.warning(
                    "Requested model is retrieval-only; falling back to "
                    f"{self._training_model_name} for training"
                )

            matcher = self._resolve_classifier_matcher()
            matcher.train(training_data, show_progress=show_progress, **kwargs)
            self._active_matcher = matcher
            self._runtime_state.has_training_data = True
            self._has_training_data = True
            self.logger.info("Training complete")
            return self

        raise ModeError(
            f"Unknown mode: {self._training_mode}",
            invalid_mode=self._training_mode,
        )

    async def fit_async(
        self,
        training_data: Optional[List[dict]] = None,
        mode: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> "Matcher":
        async with self._async_fit_lock:
            if (
                self._active_matcher is not None
                and training_data is None
                and mode is None
            ):
                return self

            await self._ensure_async_executor().run_in_thread(
                self.fit, training_data, mode, show_progress, **kwargs
            )
        return self

    def match(
        self,
        texts: TextInput,
        top_k: int = 1,
        return_metadata: bool = False,
        **kwargs,
    ) -> Any:
        if self._active_matcher is None:
            self.fit()
        threshold_override = kwargs.pop("_threshold_override", None)

        if return_metadata:
            return self._match_with_metadata(
                texts,
                top_k=top_k,
                threshold_override=threshold_override,
                **kwargs,
            )
        return self._match_sync_impl(
            texts,
            top_k=top_k,
            threshold_override=threshold_override,
            **kwargs,
        )

    async def match_async(
        self,
        texts: TextInput,
        top_k: int = 1,
        return_metadata: bool = False,
        **kwargs,
    ) -> Any:
        if self._active_matcher is None:
            await self.fit_async()
        threshold_override = kwargs.pop("_threshold_override", None)

        if return_metadata:
            return await self._match_with_metadata_async(
                texts,
                top_k=top_k,
                threshold_override=threshold_override,
                **kwargs,
            )
        return await self._match_async_impl(
            texts,
            top_k=top_k,
            threshold_override=threshold_override,
            **kwargs,
        )

    def predict(
        self,
        texts: TextInput,
        **kwargs,
    ) -> Union[Optional[str], List[Optional[str]]]:
        results = self.match(texts, top_k=1, **kwargs)
        if isinstance(results, list):
            return [result["id"] if result else None for result in results]
        return results["id"] if results else None

    def set_threshold(self, threshold: float) -> "Matcher":
        self._apply_threshold(threshold)
        return self

    def _match_hybrid(
        self,
        texts: TextInput,
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        blocking_top_k = kwargs.get("blocking_top_k", 1000)
        retrieval_top_k = kwargs.get("retrieval_top_k", max(50, top_k))
        final_top_k = kwargs.get("final_top_k", top_k)
        n_jobs = kwargs.get("n_jobs", -1)
        chunk_size = kwargs.get("chunk_size")
        effective_threshold = self._resolve_threshold(
            threshold_override, self.threshold
        )

        texts, single_input = coerce_texts(texts)
        if single_input:
            raw_results = self.hybrid_matcher.match(
                texts[0],
                blocking_top_k=blocking_top_k,
                retrieval_top_k=retrieval_top_k,
                final_top_k=final_top_k,
            )
            return self._format_hybrid_results(
                raw_results,
                top_k=top_k,
                threshold=effective_threshold,
            )

        raw_results = self.hybrid_matcher.match_bulk(
            texts,
            blocking_top_k=blocking_top_k,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )
        return [
            self._format_hybrid_results(
                results,
                top_k=top_k,
                threshold=effective_threshold,
            )
            for results in raw_results
        ]

    async def _match_hybrid_async(
        self,
        texts: TextInput,
        top_k: int = 1,
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> Any:
        executor = self._ensure_async_executor()
        effective_threshold = self._resolve_threshold(
            threshold_override, self.threshold
        )

        texts, single_input = coerce_texts(texts)
        if single_input:
            raw_results = await executor.run_in_thread(
                self.hybrid_matcher.match,
                texts[0],
                kwargs.get("blocking_top_k", 1000),
                kwargs.get("retrieval_top_k", max(50, top_k)),
                kwargs.get("final_top_k", top_k),
            )
            return self._format_hybrid_results(
                raw_results,
                top_k=top_k,
                threshold=effective_threshold,
            )

        raw_results = await executor.run_in_thread(
            self.hybrid_matcher.match_bulk,
            texts,
            kwargs.get("blocking_top_k", 1000),
            kwargs.get("retrieval_top_k", max(50, top_k)),
            kwargs.get("final_top_k", top_k),
            kwargs.get("n_jobs", -1),
            kwargs.get("chunk_size"),
        )
        return [
            self._format_hybrid_results(
                results,
                top_k=top_k,
                threshold=effective_threshold,
            )
            for results in raw_results
        ]

    async def match_batch_async(
        self,
        queries: List[str],
        threshold: Optional[float] = None,
        top_k: int = 1,
        batch_size: int = 32,
        on_progress: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ) -> List[Any]:
        if self._active_matcher is None:
            await self.fit_async()

        executor = self._ensure_async_executor()
        return await self._match_batch_async_impl(
            executor,
            queries,
            top_k,
            batch_size,
            on_progress,
            threshold_override=threshold,
            **kwargs,
        )

    async def _match_batch_async_impl(
        self,
        executor: Any,
        queries: List[str],
        top_k: int,
        batch_size: int,
        on_progress: Optional[Callable[[int, int], None]],
        threshold_override: Optional[float] = None,
        **kwargs,
    ) -> List[Any]:
        total = len(queries)
        results = []
        completed = 0

        for index in range(0, total, batch_size):
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelled():
                raise asyncio.CancelledError()

            batch = queries[index : index + batch_size]
            batch_results = await executor.run_in_thread(
                self.match,
                batch,
                top_k,
                _threshold_override=threshold_override,
                **kwargs,
            )

            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            elif not isinstance(batch_results, list):
                batch_results = list(batch_results)

            results.extend(batch_results)
            completed += len(batch)

            if on_progress:
                if inspect.iscoroutinefunction(on_progress):
                    await on_progress(completed, total)
                else:
                    on_progress(completed, total)

        return results

    def _format_hybrid_results(
        self,
        results: Optional[List[Dict[str, Any]]],
        top_k: int,
        threshold: Optional[float] = None,
    ) -> Any:
        effective_threshold = self._resolve_threshold(threshold, self.threshold)
        filtered = [
            result
            for result in (results or [])
            if result.get("score", 0.0) >= effective_threshold
        ]
        if top_k == 1:
            return filtered[0] if filtered else None
        return filtered[:top_k]

    def get_training_info(self) -> Dict[str, Any]:
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
        stats = {
            "num_entities": len(self.entities),
            "model_name": self.model_name,
            "threshold": self.threshold,
            "normalize": self.normalize,
            "training_mode": self._training_mode,
            "is_trained": self._active_matcher is not None,
        }

        if self._components._embedding_matcher:
            stats["has_embeddings"] = (
                self._components._embedding_matcher.embeddings is not None
            )

        if self._components._entity_matcher:
            classifier = getattr(self._components._entity_matcher, "classifier", None)
            stats["classifier_trained"] = (
                getattr(
                    classifier,
                    "is_trained",
                    self._components._entity_matcher.is_trained,
                )
                if classifier is not None
                else False
            )

        if self._components._bert_matcher:
            classifier = getattr(self._components._bert_matcher, "classifier", None)
            stats["bert_classifier_trained"] = (
                getattr(
                    classifier,
                    "is_trained",
                    self._components._bert_matcher.is_trained,
                )
                if classifier is not None
                else False
            )

        return stats

    def _build_explanation(
        self, query: str, results: Any, query_normalized: Optional[str]
    ) -> Dict[str, Any]:
        evaluation_threshold = self.threshold

        if results is None:
            result_list = []
        elif isinstance(results, list):
            result_list = results
        else:
            result_list = [results]

        best = result_list[0] if result_list else None
        matched = bool(best and best.get("score", 0) >= evaluation_threshold)

        return {
            "query": query,
            "query_normalized": query_normalized,
            "matched": matched,
            "best_match": best,
            "top_k": result_list,
            "threshold": evaluation_threshold,
            "mode": self._training_mode,
        }

    def _explain_match_impl(self, query: str, top_k: int) -> Dict[str, Any]:
        if not self._active_matcher:
            raise TrainingError(
                "Matcher not ready. Call fit() first.",
                details={"mode": self._training_mode},
            )

        results = self.match(query, top_k=top_k, _threshold_override=0.0)

        query_normalized = None
        if self.normalize:
            normalizer = TextNormalizer()
            query_normalized = normalizer.normalize(query)

        return self._build_explanation(query, results, query_normalized)

    def explain_match(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return self._explain_match_impl(query, top_k)

    async def explain_match_async(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        executor = self._ensure_async_executor()

        if not self._active_matcher:
            raise TrainingError(
                "Matcher not ready. Call fit() or fit_async() first.",
                details={"mode": self._training_mode},
            )

        results = await self.match_async(query, top_k=top_k, _threshold_override=0.0)

        query_normalized = None
        if self.normalize:
            normalizer = TextNormalizer()
            query_normalized = await executor.run_in_thread(normalizer.normalize, query)

        return self._build_explanation(query, results, query_normalized)

    def diagnose(self, query: str) -> Dict[str, Any]:
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

        try:
            explanation = self.explain_match(query, top_k=3)
            diagnosis.update(explanation)

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
        except (ValueError, TypeError, RuntimeError, KeyError) as exc:
            diagnosis["error"] = str(exc)
            diagnosis["suggestion"] = "Check input format and entity configuration"

        return diagnosis

    async def diagnose_async(self, query: str) -> Dict[str, Any]:
        diagnosis = {
            "query": query,
            "matcher_ready": self._active_matcher is not None,
            "active_matcher": (
                type(self._active_matcher).__name__ if self._active_matcher else None
            ),
        }

        if not self._active_matcher:
            diagnosis["issue"] = "Matcher not ready"
            diagnosis["suggestion"] = (
                "Call matcher.fit() or matcher.fit_async() to initialize"
            )
            return diagnosis

        try:
            explanation = await self.explain_match_async(query, top_k=3)
            diagnosis.update(explanation)

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
        except (ValueError, TypeError, RuntimeError, KeyError) as exc:
            diagnosis["error"] = str(exc)
            diagnosis["suggestion"] = "Check input format and entity configuration"

        return diagnosis

    def __repr__(self) -> str:
        status = "trained" if self._active_matcher else "untrained"
        return f"Matcher(mode={self._training_mode}, status={status})"

    @property
    def model(self):
        if self._training_mode == "zero-shot":
            return self.embedding_matcher.model
        if self._training_mode == "bert":
            if self.bert_matcher.classifier:
                return self.bert_matcher.classifier.model
            return None
        if self._training_mode == "hybrid":
            return self.embedding_matcher.model
        if self.entity_matcher.classifier:
            return self.entity_matcher.classifier.model
        return None

    async def aclose(self) -> None:
        if self._async_executor:
            self._async_executor.shutdown()
            self._async_executor = None

    async def __aenter__(self):
        self._ensure_async_executor()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._async_executor is not None:
            await self.aclose()
