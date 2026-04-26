"""
Internal pipeline orchestrator.
"""

from __future__ import annotations

import time
from typing import Iterable, List

from .contracts import PipelineRunResult, PipelineStage, StageContext


class PipelineOrchestrator:
    """Runs an ordered list of internal stages against a shared context."""

    def __init__(self, stages: Iterable[PipelineStage]):
        self.stages: List[PipelineStage] = list(stages)

    def run(self, context: StageContext) -> PipelineRunResult:
        stage_results = []
        timing_breakdown = {}
        for stage in self.stages:
            t0 = time.perf_counter()
            result = stage.run(context)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.timing_ms = elapsed_ms
            timing_breakdown[stage.name] = elapsed_ms
            context.artifacts.update(result.artifacts)
            context.metadata[stage.name] = result.metadata
            stage_results.append(result)
        return PipelineRunResult(
            context=context,
            stage_results=stage_results,
            timing_breakdown=timing_breakdown,
        )

    async def run_async(self, context: StageContext) -> PipelineRunResult:
        stage_results = []
        timing_breakdown = {}
        for stage in self.stages:
            t0 = time.perf_counter()
            result = await stage.run_async(context)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.timing_ms = elapsed_ms
            timing_breakdown[stage.name] = elapsed_ms
            context.artifacts.update(result.artifacts)
            context.metadata[stage.name] = result.metadata
            stage_results.append(result)
        return PipelineRunResult(
            context=context,
            stage_results=stage_results,
            timing_breakdown=timing_breakdown,
        )
