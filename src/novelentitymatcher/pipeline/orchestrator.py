"""
Internal pipeline orchestrator.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from .contracts import (
    PipelineRunResult,
    PipelineStage,
    PipelineStageError,
    StageContext,
    StageResult,
)


class PipelineOrchestrator:
    """Runs an ordered list of internal stages against a shared context."""

    def __init__(self, stages: Iterable[PipelineStage]):
        self.stages: list[PipelineStage] = list(stages)

    @staticmethod
    def _build_stage_error_result(
        stage: PipelineStage,
        exc: Exception,
        elapsed_ms: float,
        context: StageContext,
        errors: list[PipelineStageError],
    ) -> StageResult:
        message = str(exc) or repr(exc)
        stage_error = PipelineStageError(
            stage_name=stage.name,
            error_type=type(exc).__name__,
            message=message,
        )
        errors.append(stage_error)
        context.metadata.setdefault("stage_errors", []).append(
            {
                "stage_name": stage_error.stage_name,
                "error_type": stage_error.error_type,
                "message": stage_error.message,
            }
        )
        result = StageResult(
            stage_name=stage.name,
            metadata={
                "error": {
                    "type": stage_error.error_type,
                    "message": stage_error.message,
                }
            },
            errors=[stage_error.message],
        )
        result.timing_ms = elapsed_ms
        return result

    @staticmethod
    def _finalize_stage(
        stage: PipelineStage,
        result: StageResult,
        elapsed_ms: float,
        context: StageContext,
        timing_breakdown: dict[str, float],
        stage_results: list[StageResult],
    ) -> None:
        result.timing_ms = elapsed_ms
        timing_breakdown[stage.name] = elapsed_ms
        context.artifacts.update(result.artifacts)  # type: ignore[typeddict-item]
        context.metadata[stage.name] = result.metadata
        stage_results.append(result)

    def run(
        self,
        context: StageContext,
        *,
        continue_on_error: bool = False,
    ) -> PipelineRunResult:
        stage_results: list[StageResult] = []
        timing_breakdown: dict[str, float] = {}
        errors: list[PipelineStageError] = []
        for stage in self.stages:
            t0 = time.perf_counter()
            try:
                result = stage.run(context)
            except Exception as exc:
                if not continue_on_error:
                    raise
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                result = self._build_stage_error_result(
                    stage, exc, elapsed_ms, context, errors
                )
                self._finalize_stage(
                    stage, result, elapsed_ms, context, timing_breakdown, stage_results
                )
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._finalize_stage(
                stage, result, elapsed_ms, context, timing_breakdown, stage_results
            )
        return PipelineRunResult(
            context=context,
            stage_results=stage_results,
            timing_breakdown=timing_breakdown,
            errors=errors,
        )

    async def run_async(
        self,
        context: StageContext,
        *,
        continue_on_error: bool = False,
    ) -> PipelineRunResult:
        stage_results: list[StageResult] = []
        timing_breakdown: dict[str, float] = {}
        errors: list[PipelineStageError] = []
        for stage in self.stages:
            t0 = time.perf_counter()
            try:
                result = await stage.run_async(context)
            except Exception as exc:
                if not continue_on_error:
                    raise
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                result = self._build_stage_error_result(
                    stage, exc, elapsed_ms, context, errors
                )
                self._finalize_stage(
                    stage, result, elapsed_ms, context, timing_breakdown, stage_results
                )
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._finalize_stage(
                stage, result, elapsed_ms, context, timing_breakdown, stage_results
            )
        return PipelineRunResult(
            context=context,
            stage_results=stage_results,
            timing_breakdown=timing_breakdown,
            errors=errors,
        )
