"""Tests verifying async and sync paths produce equivalent results."""

import asyncio

from novelentitymatcher import DiscoveryPipeline, Matcher
from novelentitymatcher.novelty.schemas import (
    NovelClassAnalysis,
)
from novelentitymatcher.pipeline import (
    PipelineOrchestrator,
    StageContext,
    StageResult,
)


def _build_trained_matcher() -> Matcher:
    entities = [
        {"id": "physics", "name": "Quantum Physics"},
        {"id": "biology", "name": "Molecular Biology"},
    ]
    matcher = Matcher(entities=entities, model="minilm", threshold=0.6)
    matcher.fit(
        texts=[
            "quantum mechanics",
            "wave function",
            "gene expression",
            "DNA replication",
        ],
        labels=["physics", "physics", "biology", "biology"],
    )
    return matcher


def test_matcher_sync_async_parity():
    """matcher.match and matcher.match_async should produce equivalent results."""
    matcher = _build_trained_matcher()
    queries = ["quantum physics", "molecular biology"]

    sync_result = matcher.match(queries)
    async_result = asyncio.run(matcher.match_async(queries))

    assert len(sync_result) == len(async_result)

    for sync_match, async_match in zip(sync_result, async_result):
        assert sync_match["id"] == async_match["id"]
        assert abs(sync_match["score"] - async_match["score"]) < 1e-9


def test_discovery_pipeline_match_sync_async_parity(tmp_path):
    """pipeline.match and pipeline.match_async should produce equivalent results."""
    pipeline = DiscoveryPipeline(
        matcher=_build_trained_matcher(),
        auto_save=False,
        review_storage_path=tmp_path / "review_records.json",
    )

    pipeline.novel_entity_matcher.detector.detect_novel_samples = lambda **kwargs: type(
        "Report",
        (),
        {
            "novel_samples": [],
            "detection_strategies": ["confidence"],
        },
    )()

    text = "quantum mechanics"

    sync_result = pipeline.match(text)
    async_result = asyncio.run(pipeline.match_async(text))

    assert sync_result.is_match == async_result.is_match
    assert sync_result.is_novel == async_result.is_novel
    assert abs(sync_result.score - async_result.score) < 1e-9
    assert sync_result.predicted_id == async_result.predicted_id


def test_discovery_pipeline_discover_sync_async_parity(tmp_path):
    """pipeline.discover and pipeline.discover_async should produce equivalent results."""
    pipeline = DiscoveryPipeline(
        matcher=_build_trained_matcher(),
        auto_save=False,
        review_storage_path=tmp_path / "review_records.json",
    )

    pipeline.novel_entity_matcher.detector.detect_novel_samples = lambda **kwargs: type(
        "Report",
        (),
        {
            "novel_samples": [],
            "detection_strategies": ["confidence"],
        },
    )()

    pipeline.novel_entity_matcher.llm_proposer.propose_from_clusters = lambda **kwargs: (
        NovelClassAnalysis(
            proposed_classes=[],
            rejected_as_noise=[],
            analysis_summary="No clusters found.",
            cluster_count=0,
            model_used="test-model",
        )
    )

    queries = ["quantum physics", "molecular biology"]

    sync_result = asyncio.run(pipeline.discover(queries, run_llm_proposal=False))
    async_result = asyncio.run(pipeline.discover_async(queries, run_llm_proposal=False))

    assert (
        sync_result.metadata["num_novel_samples"]
        == async_result.metadata["num_novel_samples"]
    )
    assert (
        sync_result.metadata["num_discovery_clusters"]
        == async_result.metadata["num_discovery_clusters"]
    )


class MockSyncStage:
    name = "mock_sync"

    def run(self, context: StageContext) -> StageResult:
        combined = ", ".join(context.inputs)
        return StageResult(
            stage_name=self.name,
            artifacts={"processed": combined},
            metadata={"mode": "sync"},
        )

    async def run_async(self, context: StageContext) -> StageResult:
        return self.run(context)


class MockAsyncStage:
    name = "mock_async"

    def run(self, context: StageContext) -> StageResult:
        combined = ", ".join(context.inputs)
        return StageResult(
            stage_name=self.name,
            artifacts={"processed": combined},
            metadata={"mode": "sync"},
        )

    async def run_async(self, context: StageContext) -> StageResult:
        await asyncio.sleep(0.001)
        combined = ", ".join(context.inputs)
        return StageResult(
            stage_name=self.name,
            artifacts={"processed": combined},
            metadata={"mode": "async"},
        )


def test_pipeline_orchestrator_sync_async_parity():
    """orchestrator.run and orchestrator.run_async should produce equivalent results."""
    orchestrator = PipelineOrchestrator(stages=[MockSyncStage(), MockAsyncStage()])

    context = "test context"
    sync_result = orchestrator.run(StageContext(inputs=[context]))
    async_result = asyncio.run(orchestrator.run_async(StageContext(inputs=[context])))

    assert len(sync_result.stage_results) == len(async_result.stage_results)

    for sync_sr, async_sr in zip(sync_result.stage_results, async_result.stage_results):
        assert sync_sr.stage_name == async_sr.stage_name
        assert sync_sr.artifacts == async_sr.artifacts

    assert sync_result.context.artifacts == async_result.context.artifacts
