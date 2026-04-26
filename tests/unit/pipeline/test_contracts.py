"""Unit tests for pipeline.contracts module."""

import pytest

from novelentitymatcher.pipeline.contracts import (
    PipelineRunResult,
    PipelineStage,
    StageContext,
    StageResult,
)


class TestStageContext:
    def test_init_with_inputs_only(self):
        ctx = StageContext(inputs=["a", "b"])
        assert ctx.inputs == ["a", "b"]
        assert ctx.artifacts == {}
        assert ctx.metadata == {}

    def test_init_with_artifacts_and_metadata(self):
        ctx = StageContext(
            inputs=["x"],
            artifacts={"key": "value"},
            metadata={"stage": 1},
        )
        assert ctx.artifacts == {"key": "value"}
        assert ctx.metadata == {"stage": 1}

    def test_artifact_summary_empty(self):
        ctx = StageContext(inputs=[])
        assert ctx.artifact_summary() == {}

    def test_artifact_summary_with_types(self):
        ctx = StageContext(
            inputs=[],
            artifacts={"str_val": "hello", "int_val": 42, "list_val": [1, 2]},
        )
        summary = ctx.artifact_summary()
        assert summary["str_val"] == "str"
        assert summary["int_val"] == "int"
        assert summary["list_val"] == "list"


class TestStageResult:
    def test_defaults(self):
        result = StageResult(stage_name="test")
        assert result.stage_name == "test"
        assert result.artifacts == {}
        assert result.metadata == {}
        assert result.contract_version == "1.0"
        assert result.timing_ms is None
        assert result.stage_config_snapshot == {}
        assert result.errors == []

    def test_custom_values(self):
        result = StageResult(
            stage_name="test",
            artifacts={"k": "v"},
            metadata={"m": 1},
            contract_version="2.0",
            timing_ms=12.5,
            stage_config_snapshot={"opt": True},
            errors=["err1"],
        )
        assert result.timing_ms == 12.5
        assert result.errors == ["err1"]


class TestPipelineRunResult:
    def test_total_time_ms_empty(self):
        result = PipelineRunResult(context=StageContext(inputs=[]))
        assert result.total_time_ms == 0.0

    def test_total_time_ms_with_breakdown(self):
        result = PipelineRunResult(
            context=StageContext(inputs=[]),
            timing_breakdown={"stage1": 10.0, "stage2": 20.5},
        )
        assert result.total_time_ms == pytest.approx(30.5)


class DummyStage(PipelineStage):
    name = "dummy"

    def run(self, context: StageContext) -> StageResult:
        return StageResult(stage_name=self.name)


class TestPipelineStage:
    def test_abstract_run(self):
        stage = DummyStage()
        ctx = StageContext(inputs=[])
        result = stage.run(ctx)
        assert result.stage_name == "dummy"

    async def test_default_async_run(self):
        stage = DummyStage()
        ctx = StageContext(inputs=[])
        result = await stage.run_async(ctx)
        assert isinstance(result, StageResult)
        assert result.stage_name == "dummy"
