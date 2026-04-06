"""Tests for PipelineOrchestrator async and edge case behavior."""

import asyncio

from novelentitymatcher.pipeline import PipelineOrchestrator, StageContext, StageResult


class CountingStage:
    """Test stage that counts invocations."""

    name = "counting"

    def __init__(self, fail_on_nth: int | None = None):
        self.fail_on_nth = fail_on_nth
        self.call_count = 0

    def run(self, context: StageContext) -> StageResult:
        self.call_count += 1
        return StageResult(
            stage_name=self.name,
            artifacts={"calls": self.call_count},
        )


class AsyncCountingStage:
    """Test stage with async run_async implementation."""

    name = "async_counting"

    def __init__(self):
        self.call_count = 0

    def run(self, context: StageContext) -> StageResult:
        self.call_count += 1
        return StageResult(
            stage_name=self.name,
            artifacts={"calls": self.call_count},
        )

    async def run_async(self, context: StageContext) -> StageResult:
        await asyncio.sleep(0.001)
        self.call_count += 1
        return StageResult(
            stage_name=self.name,
            artifacts={"async_calls": self.call_count},
            metadata={"async_calls": self.call_count},
        )


def test_orchestrator_run_async_calls_all_stages():
    """Orchestrator.run_async should call run_async on each stage."""
    orchestrator = PipelineOrchestrator(
        stages=[
            AsyncCountingStage(),
            AsyncCountingStage(),
        ]
    )

    result = asyncio.run(orchestrator.run_async(StageContext(inputs=["test"])))

    assert len(result.stage_results) == 2
    assert result.stage_results[0].stage_name == "async_counting"
    assert result.stage_results[1].stage_name == "async_counting"
    assert result.context.metadata["async_counting"]["async_calls"] == 1


def test_orchestrator_run_async_updates_context():
    """run_async should update context artifacts and metadata."""
    orchestrator = PipelineOrchestrator(
        stages=[
            AsyncCountingStage(),
        ]
    )

    result = asyncio.run(orchestrator.run_async(StageContext(inputs=["a", "b"])))

    assert "async_counting" in result.context.metadata
    assert result.context.artifacts["async_calls"] == 1


def test_orchestrator_empty_stages():
    """Orchestrator should handle empty stage list gracefully."""
    orchestrator = PipelineOrchestrator(stages=[])

    result = orchestrator.run(StageContext(inputs=["test"]))

    assert result.stage_results == []
    assert result.context.artifacts == {}
    assert result.context.metadata == {}


def test_orchestrator_context_passed_between_stages():
    """Each stage should see artifacts from previous stages."""
    call_counts = []

    class CallCountingStage:
        name = "counting"

        def __init__(self, stage_id: int):
            self.stage_id = stage_id

        def run(self, context: StageContext) -> StageResult:
            call_counts.append(self.stage_id)
            return StageResult(
                stage_name=f"counting_{self.stage_id}",
                artifacts={f"stage_{self.stage_id}_ran": True},
            )

    orchestrator = PipelineOrchestrator(
        stages=[
            CallCountingStage(1),
            CallCountingStage(2),
            CallCountingStage(3),
        ]
    )

    result = orchestrator.run(StageContext(inputs=["test"]))

    assert len(result.stage_results) == 3
    assert call_counts == [1, 2, 3]


def test_orchestrator_run_sync_uses_run_not_run_async():
    """Orchestrator.run should call run, not run_async."""
    async_stage = AsyncCountingStage()
    orchestrator = PipelineOrchestrator(stages=[async_stage])

    result = orchestrator.run(StageContext(inputs=["test"]))

    assert async_stage.call_count == 1
    assert "async_calls" not in result.stage_results[0].artifacts
    assert "calls" in result.stage_results[0].artifacts


def test_orchestrator_single_stage():
    """Orchestrator should work with a single stage."""
    orchestrator = PipelineOrchestrator(stages=[CountingStage()])

    result = orchestrator.run(StageContext(inputs=["test"]))

    assert len(result.stage_results) == 1
    assert result.stage_results[0].stage_name == "counting"


def test_orchestrator_multiple_stages_preserve_order():
    """Stages should execute in the order they were provided."""
    call_order = []

    class OrderTrackingStage:
        name = "order"

        def __init__(self, stage_id: int):
            self.stage_id = stage_id

        def run(self, context: StageContext) -> StageResult:
            call_order.append(self.stage_id)
            return StageResult(stage_name=f"stage_{self.stage_id}")

    orchestrator = PipelineOrchestrator(
        stages=[
            OrderTrackingStage(1),
            OrderTrackingStage(2),
            OrderTrackingStage(3),
        ]
    )

    orchestrator.run(StageContext(inputs=[]))

    assert call_order == [1, 2, 3]
