import asyncio
import os
import platform
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .matching_strategy import MatchingStrategy

from sentence_transformers import SentenceTransformer

# Enable CPU fallback for unsupported MPS ops before torch/sentence-transformers import.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


from ..config import (
    is_bert_model,
    supports_training_model,
)
from ..exceptions import ModeError, ValidationError
from ..pipeline.match_result import build_match_result_with_metadata
from ..utils.logging_config import configure_logging, get_logger
from ..utils.validation import (
    validate_entities,
    validate_threshold,
)
from .embedding_matcher import EmbeddingMatcher
from .matcher_components import MatcherComponentFactory
from .matcher_engines import _BatchEngine, _DiagnosisEngine, _HybridEngine
from .matcher_entity import _EntityMatcher
from .matcher_runtime import MatcherRuntimeState
from .matcher_shared import (
    TextInput,
    coerce_texts,
    extract_top_prediction_metadata,
    resolve_threshold,
)
from .matching_strategy import MatcherFacade

if TYPE_CHECKING:
    from .async_utils import AsyncExecutor

EmbeddingModel = SentenceTransformer
__all__ = ["EmbeddingMatcher", "EmbeddingModel", "Matcher", "_EntityMatcher"]

# Backwards-compatible aliases for internal helpers that some callers may import.
_coerce_texts = coerce_texts
_extract_top_prediction_metadata = extract_top_prediction_metadata
_resolve_threshold = resolve_threshold


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
        entities: list[dict[str, Any]],
        model: str = "default",
        threshold: float = 0.7,
        normalize: bool = True,
        mode: str | None = None,
        blocking_strategy: Any | None = None,
        reranker_model: str = "default",
        verbose: bool = False,
        metrics_callback: Callable | None = None,
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

        self._async_executor: AsyncExecutor | None = None
        self._async_fit_lock = asyncio.Lock()

        self._training_mode = self._runtime_state.training_mode
        self._components = MatcherComponentFactory(self)
        self._has_training_data = self._runtime_state.has_training_data
        self._active_matcher: Any | None = None
        self._detected_mode: str | None = self._runtime_state.detected_mode

        self._hybrid_engine = _HybridEngine(self)
        self._batch_engine = _BatchEngine(self)
        self._diagnosis_engine = _DiagnosisEngine(self)

    def _emit_metric(
        self,
        name: str,
        value: float,
        unit: str,
        labels: dict[str, str] | None = None,
    ) -> None:
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
    def _resolve_threshold(threshold_override: float | None, default: float) -> float:
        return resolve_threshold(threshold_override, default)

    def _get_strategy(self) -> "MatchingStrategy":
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
        threshold_override: float | None = None,
        **kwargs,
    ) -> Any:
        if self._training_mode == "hybrid":
            return self._hybrid_engine.match(
                texts, top_k=top_k, threshold_override=threshold_override, **kwargs
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
        threshold_override: float | None = None,
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
            match_method=self._training_mode,
        )

    async def _match_with_metadata_async(
        self,
        texts: TextInput,
        top_k: int = 1,
        threshold_override: float | None = None,
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
            match_method=self._training_mode,
        )

    async def _match_async_impl(
        self,
        texts: TextInput,
        top_k: int = 1,
        threshold_override: float | None = None,
        **kwargs,
    ) -> Any:
        if self._training_mode == "hybrid":
            return await self._hybrid_engine.match_async(
                texts, top_k=top_k, threshold_override=threshold_override, **kwargs
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

    def get_reference_corpus(self) -> dict[str, Any]:
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

    def _detect_training_mode(self, training_data: list[dict] | None) -> str:
        if training_data is None:
            detected = "zero-shot"
        else:
            entity_counts: dict[str, int] = defaultdict(int)
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
        training_data: list[dict] | None = None,
        mode: str | None = None,
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
        training_data: list[dict] | None = None,
        mode: str | None = None,
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
    ) -> str | None | list[str | None]:
        results = self.match(texts, top_k=1, **kwargs)
        if isinstance(results, list):
            return [result["id"] if result else None for result in results]
        return results["id"] if results else None

    def set_threshold(self, threshold: float) -> "Matcher":
        self._apply_threshold(threshold)
        return self

    async def match_batch_async(
        self,
        queries: list[str],
        threshold: float | None = None,
        top_k: int = 1,
        batch_size: int = 32,
        on_progress: Callable[[int, int], None] | None = None,
        **kwargs,
    ) -> list[Any]:
        if self._active_matcher is None:
            await self.fit_async()

        return await self._batch_engine.match_batch(
            queries,
            threshold=threshold,
            top_k=top_k,
            batch_size=batch_size,
            on_progress=on_progress,
            **kwargs,
        )

    def get_training_info(self) -> dict[str, Any]:
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

    def get_statistics(self) -> dict[str, Any]:
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

    def explain_match(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return self._diagnosis_engine.explain(query, top_k)

    async def explain_match_async(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return await self._diagnosis_engine.explain_async(query, top_k)

    def diagnose(self, query: str) -> dict[str, Any]:
        return self._diagnosis_engine.diagnose(query)

    async def diagnose_async(self, query: str) -> dict[str, Any]:
        return await self._diagnosis_engine.diagnose_async(query)

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
