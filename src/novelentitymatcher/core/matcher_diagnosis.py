from typing import TYPE_CHECKING, Any

from ..exceptions import TrainingError
from .normalizer import TextNormalizer

if TYPE_CHECKING:
    from .matcher import Matcher


class DiagnosisEngine:
    """Explanation and diagnosis operations for the Matcher facade."""

    def __init__(self, facade: "Matcher") -> None:
        self._facade = facade

    def build_explanation(
        self, query: str, results: Any, query_normalized: str | None
    ) -> dict[str, Any]:
        facade = self._facade
        evaluation_threshold = facade.threshold

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
            "mode": facade._training_mode,
        }

    def _build_diagnosis_header(self, query: str) -> dict[str, Any]:
        facade = self._facade
        return {
            "query": query,
            "matcher_ready": facade._active_matcher is not None,
            "active_matcher": (
                type(facade._active_matcher).__name__
                if facade._active_matcher
                else None
            ),
        }

    def _finalize_diagnosis(
        self,
        diagnosis: dict[str, Any],
        explanation: dict[str, Any] | None = None,
        exc: Exception | None = None,
    ) -> None:
        if exc is not None:
            diagnosis["error"] = str(exc)
            diagnosis["suggestion"] = "Check input format and entity configuration"
        elif explanation is not None:
            diagnosis.update(explanation)
            self._add_failure_hints(diagnosis, explanation)

    def explain(self, query: str, top_k: int = 5) -> dict[str, Any]:
        facade = self._facade
        if not facade._active_matcher:
            raise TrainingError(
                "Matcher not ready. Call fit() first.",
                details={"mode": facade._training_mode},
            )

        results = facade.match(query, top_k=top_k, _threshold_override=0.0)

        query_normalized = None
        if facade.normalize:
            normalizer = TextNormalizer()
            query_normalized = normalizer.normalize(query)

        return self.build_explanation(query, results, query_normalized)

    async def explain_async(self, query: str, top_k: int = 5) -> dict[str, Any]:
        facade = self._facade
        executor = facade._ensure_async_executor()

        if not facade._active_matcher:
            raise TrainingError(
                "Matcher not ready. Call fit() or fit_async() first.",
                details={"mode": facade._training_mode},
            )

        results = await facade.match_async(query, top_k=top_k, _threshold_override=0.0)

        query_normalized = None
        if facade.normalize:
            normalizer = TextNormalizer()
            query_normalized = await executor.run_in_thread(normalizer.normalize, query)

        return self.build_explanation(query, results, query_normalized)

    def diagnose(self, query: str) -> dict[str, Any]:
        diagnosis = self._build_diagnosis_header(query)

        if not self._facade._active_matcher:
            diagnosis["issue"] = "Matcher not ready"
            diagnosis["suggestion"] = "Call matcher.fit() to initialize the matcher"
            return diagnosis

        try:
            explanation = self.explain(query, top_k=3)
            self._finalize_diagnosis(diagnosis, explanation=explanation)
        except Exception as exc:
            self._finalize_diagnosis(diagnosis, exc=exc)

        return diagnosis

    async def diagnose_async(self, query: str) -> dict[str, Any]:
        diagnosis = self._build_diagnosis_header(query)

        if not self._facade._active_matcher:
            diagnosis["issue"] = "Matcher not ready"
            diagnosis["suggestion"] = (
                "Call matcher.fit() or matcher.fit_async() to initialize"
            )
            return diagnosis

        try:
            explanation = await self.explain_async(query, top_k=3)
            self._finalize_diagnosis(diagnosis, explanation=explanation)
        except Exception as exc:
            self._finalize_diagnosis(diagnosis, exc=exc)

        return diagnosis

    @staticmethod
    def _add_failure_hints(
        diagnosis: dict[str, Any], explanation: dict[str, Any]
    ) -> None:
        if not explanation["matched"]:
            if explanation["best_match"]:
                score = explanation["best_match"].get("score", 0)
                threshold = explanation["threshold"]
                diagnosis["issue"] = f"Score {score:.2f} below threshold {threshold}"
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
