"""
Internal staged discovery pipeline contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageContext:
    """Mutable context passed between internal pipeline stages."""

    inputs: list[str]
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def artifact_summary(self) -> dict[str, str]:
        """Return a summary of artifact keys and their types."""
        return {key: type(value).__name__ for key, value in self.artifacts.items()}


@dataclass
class StageResult:
    """Result returned by a single pipeline stage."""

    stage_name: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = "1.0"
    timing_ms: float | None = None
    stage_config_snapshot: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class PipelineRunResult:
    """Terminal result for an internal pipeline run."""

    context: StageContext
    stage_results: list[StageResult] = field(default_factory=list)
    timing_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def total_time_ms(self) -> float:
        return sum(self.timing_breakdown.values())


class PipelineStage(ABC):
    """Base contract for internal discovery stages."""

    name: str

    @abstractmethod
    def run(self, context: StageContext) -> StageResult:
        """Execute the stage synchronously."""

    async def run_async(self, context: StageContext) -> StageResult:
        """Async entrypoint; stages can override when they have real async work."""
        return self.run(context)
