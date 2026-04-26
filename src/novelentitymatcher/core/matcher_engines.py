from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..exceptions import TrainingError
from .matcher_shared import (
    TextInput,
    coerce_texts,
    resolve_threshold,
)
from .normalizer import TextNormalizer

if TYPE_CHECKING:
    from .matcher import Matcher


class _HybridEngine:
    """Hybrid matching operations delegated from the Matcher facade."""

    def __init__(self, owner: Matcher):
        self._m = owner

    def match(
        self,
        texts: TextInput,
        top_k: int = 1,
        threshold_override: float | None = None,
        **kwargs,
    ) -> Any:
        blocking_top_k = kwargs.get("blocking_top_k", 1000)
        retrieval_top_k = kwargs.get("retrieval_top_k", max(50, top_k))
        final_top_k = kwargs.get("final_top_k", top_k)
        n_jobs = kwargs.get("n_jobs", -1)
        chunk_size = kwargs.get("chunk_size")
        effective_threshold = resolve_threshold(threshold_override, self._m.threshold)

        texts, single_input = coerce_texts(texts)
        if single_input:
            raw_results = self._m.hybrid_matcher.match(
                texts[0],
                blocking_top_k=blocking_top_k,
                retrieval_top_k=retrieval_top_k,
                final_top_k=final_top_k,
            )
            return self._format_results(raw_results, top_k, effective_threshold)

        raw_results = self._m.hybrid_matcher.match_bulk(
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
        top_k: int = 1,
        threshold_override: float | None = None,
        **kwargs,
    ) -> Any:
        executor = self._m._ensure_async_executor()
        effective_threshold = resolve_threshold(threshold_override, self._m.threshold)

        texts, single_input = coerce_texts(texts)
        if single_input:
            raw_results = await executor.run_in_thread(
                self._m.hybrid_matcher.match,
                texts[0],
                kwargs.get("blocking_top_k", 1000),
                kwargs.get("retrieval_top_k", max(50, top_k)),
                kwargs.get("final_top_k", top_k),
            )
            return self._format_results(raw_results, top_k, effective_threshold)

        raw_results = await executor.run_in_thread(
            self._m.hybrid_matcher.match_bulk,
            texts,
            kwargs.get("blocking_top_k", 1000),
            kwargs.get("retrieval_top_k", max(50, top_k)),
            kwargs.get("final_top_k", top_k),
            kwargs.get("n_jobs", -1),
            kwargs.get("chunk_size"),
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
        effective_threshold = resolve_threshold(threshold, self._m.threshold)
        filtered = [
            result
            for result in (results or [])
            if result.get("score", 0.0) >= effective_threshold
        ]
        if top_k == 1:
            return filtered[0] if filtered else None
        return filtered[:top_k]


class _BatchEngine:
    """Batch async matching operations delegated from the Matcher facade."""

    def __init__(self, owner: Matcher):
        self._m = owner

    async def match_batch(
        self,
        queries: list[str],
        threshold: float | None = None,
        top_k: int = 1,
        batch_size: int = 32,
        on_progress: Callable[[int, int], None] | None = None,
        **kwargs,
    ) -> list[Any]:
        executor = self._m._ensure_async_executor()
        return await self._match_impl(
            executor,
            queries,
            top_k,
            batch_size,
            on_progress,
            threshold_override=threshold,
            **kwargs,
        )

    async def _match_impl(
        self,
        executor: Any,
        queries: list[str],
        top_k: int,
        batch_size: int,
        on_progress: Callable[[int, int], None] | None,
        threshold_override: float | None = None,
        **kwargs,
    ) -> list[Any]:
        total = len(queries)
        results = []
        completed = 0

        for index in range(0, total, batch_size):
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelled():
                raise asyncio.CancelledError()

            batch = queries[index : index + batch_size]
            batch_results = await executor.run_in_thread(
                self._m.match,
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


class _DiagnosisEngine:
    """Explanation and diagnosis operations delegated from the Matcher facade."""

    def __init__(self, owner: Matcher):
        self._m = owner

    def build_explanation(
        self, query: str, results: Any, query_normalized: str | None
    ) -> dict[str, Any]:
        evaluation_threshold = self._m.threshold

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
            "mode": self._m._training_mode,
        }

    def explain(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not self._m._active_matcher:
            raise TrainingError(
                "Matcher not ready. Call fit() first.",
                details={"mode": self._m._training_mode},
            )

        results = self._m.match(query, top_k=top_k, _threshold_override=0.0)

        query_normalized = None
        if self._m.normalize:
            normalizer = TextNormalizer()
            query_normalized = normalizer.normalize(query)

        return self.build_explanation(query, results, query_normalized)

    async def explain_async(self, query: str, top_k: int = 5) -> dict[str, Any]:
        executor = self._m._ensure_async_executor()

        if not self._m._active_matcher:
            raise TrainingError(
                "Matcher not ready. Call fit() or fit_async() first.",
                details={"mode": self._m._training_mode},
            )

        results = await self._m.match_async(query, top_k=top_k, _threshold_override=0.0)

        query_normalized = None
        if self._m.normalize:
            normalizer = TextNormalizer()
            query_normalized = await executor.run_in_thread(normalizer.normalize, query)

        return self.build_explanation(query, results, query_normalized)

    def diagnose(self, query: str) -> dict[str, Any]:
        diagnosis: dict[str, Any] = {
            "query": query,
            "matcher_ready": self._m._active_matcher is not None,
            "active_matcher": (
                type(self._m._active_matcher).__name__
                if self._m._active_matcher
                else None
            ),
        }

        if not self._m._active_matcher:
            diagnosis["issue"] = "Matcher not ready"
            diagnosis["suggestion"] = "Call matcher.fit() to initialize the matcher"
            return diagnosis

        try:
            explanation = self.explain(query, top_k=3)
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

    async def diagnose_async(self, query: str) -> dict[str, Any]:
        diagnosis: dict[str, Any] = {
            "query": query,
            "matcher_ready": self._m._active_matcher is not None,
            "active_matcher": (
                type(self._m._active_matcher).__name__
                if self._m._active_matcher
                else None
            ),
        }

        if not self._m._active_matcher:
            diagnosis["issue"] = "Matcher not ready"
            diagnosis["suggestion"] = (
                "Call matcher.fit() or matcher.fit_async() to initialize"
            )
            return diagnosis

        try:
            explanation = await self.explain_async(query, top_k=3)
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
