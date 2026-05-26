from __future__ import annotations

from typing import Any, Protocol

from ..config import is_bert_model, supports_training_model
from .matcher_shared import TextInput, coerce_texts, resolve_threshold


class _ModeContext(Protocol):
    threshold: float
    logger: Any
    _training_mode: str
    _requested_model: str
    _training_model_name: str
    _has_training_data: bool

    @property
    def embedding_matcher(self) -> Any: ...

    @property
    def entity_matcher(self) -> Any: ...

    @property
    def bert_matcher(self) -> Any: ...

    @property
    def hybrid_matcher(self) -> Any: ...

    def _ensure_async_executor(self) -> Any: ...


class ZeroShotMode:
    """Zero-shot matching using embedding similarity."""

    def __init__(self, ctx: _ModeContext) -> None:
        self._ctx = ctx

    def fit(
        self,
        training_data: list[dict] | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Any:
        self._ctx.logger.info("Building zero-shot index (no training required)")
        self._ctx.embedding_matcher.build_index()
        return self._ctx.embedding_matcher

    def match(
        self,
        texts: TextInput,
        top_k: int,
        threshold_override: float | None,
        **kwargs,
    ) -> Any:
        effective_threshold = resolve_threshold(threshold_override, self._ctx.threshold)
        return self._ctx.embedding_matcher.match(
            texts,
            top_k=top_k,
            threshold_override=effective_threshold,
            **kwargs,
        )

    async def match_async(
        self,
        texts: TextInput,
        top_k: int,
        threshold_override: float | None,
        **kwargs,
    ) -> Any:
        effective_threshold = resolve_threshold(threshold_override, self._ctx.threshold)
        return await self._ctx.embedding_matcher.match_async(
            texts,
            top_k=top_k,
            threshold_override=effective_threshold,
            **kwargs,
        )


class TrainingMode:
    """Trained matching using SetFit or BERT classifiers."""

    def __init__(self, ctx: _ModeContext, classifier_type: str) -> None:
        self._ctx = ctx
        self._classifier_type = classifier_type

    def _resolve_matcher(self) -> Any:
        if self._classifier_type == "bert":
            return self._ctx.bert_matcher
        return self._ctx.entity_matcher

    def fit(
        self,
        training_data: list[dict] | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Any:
        ctx = self._ctx
        ctx.logger.info(f"Training in {ctx._training_mode} mode")

        if self._classifier_type == "bert" and not is_bert_model(ctx._requested_model):
            ctx.logger.warning(
                f"Using non-BERT model '{ctx._requested_model}' with bert mode. "
                "For optimal results, use a BERT-based model."
            )
        elif self._classifier_type == "setfit" and not supports_training_model(
            ctx._requested_model
        ):
            ctx.logger.warning(
                "Requested model is retrieval-only; falling back to "
                f"{ctx._training_model_name} for training"
            )

        resolved = self._resolve_matcher()
        resolved.train(training_data, show_progress=show_progress, **kwargs)
        ctx._has_training_data = True
        ctx.logger.info("Training complete")
        return resolved

    def match(
        self,
        texts: TextInput,
        top_k: int,
        threshold_override: float | None,
        **kwargs,
    ) -> Any:
        effective_threshold = resolve_threshold(threshold_override, self._ctx.threshold)
        return self._resolve_matcher().match(
            texts,
            candidates=kwargs.get("candidates"),
            top_k=top_k,
            threshold_override=effective_threshold,
        )

    async def match_async(
        self,
        texts: TextInput,
        top_k: int,
        threshold_override: float | None,
        **kwargs,
    ) -> Any:
        effective_threshold = resolve_threshold(threshold_override, self._ctx.threshold)
        return await self._resolve_matcher().match_async(
            texts,
            candidates=kwargs.get("candidates"),
            top_k=top_k,
            threshold_override=effective_threshold,
        )


class HybridMode:
    """Hybrid matching combining blocking, retrieval, and reranking."""

    def __init__(self, ctx: _ModeContext) -> None:
        self._ctx = ctx

    def fit(
        self,
        training_data: list[dict] | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Any:
        ctx = self._ctx
        ctx.logger.info("Initializing hybrid pipeline")
        ctx._has_training_data = False
        return ctx.hybrid_matcher

    @staticmethod
    def _extract_hybrid_kwargs(
        kwargs: dict[str, Any], top_k: int
    ) -> tuple[int, int, int, int, Any]:
        return (
            kwargs.get("blocking_top_k", 1000),
            kwargs.get("retrieval_top_k", max(50, top_k)),
            kwargs.get("final_top_k", top_k),
            kwargs.get("n_jobs", -1),
            kwargs.get("chunk_size"),
        )

    def match(
        self,
        texts: TextInput,
        top_k: int,
        threshold_override: float | None,
        **kwargs,
    ) -> Any:
        ctx = self._ctx
        effective_threshold = resolve_threshold(threshold_override, ctx.threshold)
        blocking_top_k, retrieval_top_k, final_top_k, n_jobs, chunk_size = (
            self._extract_hybrid_kwargs(kwargs, top_k)
        )

        texts, single_input = coerce_texts(texts)
        if single_input:
            raw_results = ctx.hybrid_matcher.match(
                texts[0],
                blocking_top_k=blocking_top_k,
                retrieval_top_k=retrieval_top_k,
                final_top_k=final_top_k,
            )
            return self._format_results(raw_results, top_k, effective_threshold)

        raw_results = ctx.hybrid_matcher.match_bulk(
            texts,
            blocking_top_k=blocking_top_k,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )
        return [
            self._format_results(r, top_k, effective_threshold) for r in raw_results
        ]

    async def match_async(
        self,
        texts: TextInput,
        top_k: int,
        threshold_override: float | None,
        **kwargs,
    ) -> Any:
        ctx = self._ctx
        executor = ctx._ensure_async_executor()
        effective_threshold = resolve_threshold(threshold_override, ctx.threshold)
        blocking_top_k, retrieval_top_k, final_top_k, n_jobs, chunk_size = (
            self._extract_hybrid_kwargs(kwargs, top_k)
        )

        texts, single_input = coerce_texts(texts)
        if single_input:
            raw_results = await executor.run_in_thread(
                ctx.hybrid_matcher.match,
                texts[0],
                blocking_top_k,
                retrieval_top_k,
                final_top_k,
            )
            return self._format_results(raw_results, top_k, effective_threshold)

        raw_results = await executor.run_in_thread(
            ctx.hybrid_matcher.match_bulk,
            texts,
            blocking_top_k,
            retrieval_top_k,
            final_top_k,
            n_jobs,
            chunk_size,
        )
        return [
            self._format_results(r, top_k, effective_threshold) for r in raw_results
        ]

    def _format_results(
        self,
        results: list[dict[str, Any]] | None,
        top_k: int,
        threshold: float | None = None,
    ) -> Any:
        effective_threshold = resolve_threshold(threshold, self._ctx.threshold)
        filtered = [
            result
            for result in (results or [])
            if result.get("score", 0.0) >= effective_threshold
        ]
        if top_k == 1:
            return filtered[0] if filtered else None
        return filtered[:top_k]
