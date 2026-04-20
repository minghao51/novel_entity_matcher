"""Extended tests for DiscoveryPipeline edge cases and from_config."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from novelentitymatcher import DiscoveryPipeline, Matcher
from novelentitymatcher.novelty.schemas import (
    ClassProposal,
    NovelClassAnalysis,
)


def _build_trained_matcher() -> Matcher:
    """Create a trained matcher for testing."""
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


class TestDiscoveryPipelineMatch:
    """Tests for DiscoveryPipeline.match method."""

    def test_match_returns_novel_entity_match_result(self):
        """match should return a NovelEntityMatchResult."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
        )

        result = pipeline.match("quantum wave")

        assert result is not None
        assert hasattr(result, "is_match")
        assert hasattr(result, "id")
        assert hasattr(result, "score")


class TestDiscoveryPipelineFit:
    """Tests for DiscoveryPipeline fit methods."""

    def test_fit_returns_self(self):
        """fit should return the pipeline for chaining."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
        )

        result = pipeline.fit(
            texts=["quantum", "biology"],
            labels=["physics", "biology"],
        )

        assert result is pipeline

    @pytest.mark.asyncio
    async def test_fit_async_returns_self(self):
        """fit_async should return the pipeline for chaining."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
        )

        result = await pipeline.fit_async(
            texts=["quantum", "biology"],
            labels=["physics", "biology"],
        )

        assert result is pipeline


class TestDiscoveryPipelineReviewLifecycle:
    """Tests for DiscoveryPipeline reject and approve workflows."""

    def test_reject_proposal_sets_state_to_rejected(self, tmp_path: Path):
        """reject_proposal should set record state to rejected."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
            review_storage_path=tmp_path / "records.json",
        )
        manager = pipeline.review_manager
        records = manager.create_records(
            type(
                "Report",
                (),
                {
                    "discovery_id": "disc123",
                    "timestamp": datetime.now(),
                    "class_proposals": NovelClassAnalysis(
                        proposed_classes=[
                            ClassProposal(
                                name="Quantum Biology",
                                description="desc",
                                confidence=0.9,
                                sample_count=2,
                                example_samples=["a", "b"],
                                justification="coherent",
                            )
                        ],
                        rejected_as_noise=[],
                        analysis_summary="One cluster",
                        cluster_count=1,
                        model_used="test",
                    ),
                    "diagnostics": {},
                },
            )()
        )

        rejected = pipeline.reject_proposal(
            records[0].review_id, notes="not convincing"
        )

        assert rejected.state == "rejected"
        assert rejected.reviewed_at is not None
        assert rejected.notes == "not convincing"

    def test_reject_proposal_unknown_id_raises_key_error(self, tmp_path: Path):
        """reject_proposal should raise KeyError for unknown review_id."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
            review_storage_path=tmp_path / "records.json",
        )

        with pytest.raises(KeyError):
            pipeline.reject_proposal("nonexistent_id")

    def test_list_review_records_returns_all_when_no_discovery_id(self, tmp_path: Path):
        """list_review_records should return all records when discovery_id is None."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
            review_storage_path=tmp_path / "records.json",
        )
        manager = pipeline.review_manager
        manager.create_records(
            type(
                "Report",
                (),
                {
                    "discovery_id": "disc1",
                    "timestamp": datetime.now(),
                    "class_proposals": NovelClassAnalysis(
                        proposed_classes=[
                            ClassProposal(
                                name="A",
                                description="d",
                                confidence=0.9,
                                sample_count=1,
                                example_samples=["a"],
                                justification="j",
                            )
                        ],
                        rejected_as_noise=[],
                        analysis_summary="",
                        cluster_count=1,
                        model_used="test",
                    ),
                    "diagnostics": {},
                },
            )()
        )
        manager.create_records(
            type(
                "Report",
                (),
                {
                    "discovery_id": "disc2",
                    "timestamp": datetime.now(),
                    "class_proposals": NovelClassAnalysis(
                        proposed_classes=[
                            ClassProposal(
                                name="B",
                                description="d",
                                confidence=0.9,
                                sample_count=1,
                                example_samples=["a"],
                                justification="j",
                            )
                        ],
                        rejected_as_noise=[],
                        analysis_summary="",
                        cluster_count=1,
                        model_used="test",
                    ),
                    "diagnostics": {},
                },
            )()
        )

        records = pipeline.list_review_records()

        assert len(records) == 2

    def test_list_review_records_filters_by_discovery_id(self, tmp_path: Path):
        """list_review_records should filter by discovery_id when provided."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
            review_storage_path=tmp_path / "records.json",
        )
        manager = pipeline.review_manager
        manager.create_records(
            type(
                "Report",
                (),
                {
                    "discovery_id": "disc1",
                    "timestamp": datetime.now(),
                    "class_proposals": NovelClassAnalysis(
                        proposed_classes=[
                            ClassProposal(
                                name="A",
                                description="d",
                                confidence=0.9,
                                sample_count=1,
                                example_samples=["a"],
                                justification="j",
                            )
                        ],
                        rejected_as_noise=[],
                        analysis_summary="",
                        cluster_count=1,
                        model_used="test",
                    ),
                    "diagnostics": {},
                },
            )()
        )
        manager.create_records(
            type(
                "Report",
                (),
                {
                    "discovery_id": "disc2",
                    "timestamp": datetime.now(),
                    "class_proposals": NovelClassAnalysis(
                        proposed_classes=[
                            ClassProposal(
                                name="B",
                                description="d",
                                confidence=0.9,
                                sample_count=1,
                                example_samples=["a"],
                                justification="j",
                            )
                        ],
                        rejected_as_noise=[],
                        analysis_summary="",
                        cluster_count=1,
                        model_used="test",
                    ),
                    "diagnostics": {},
                },
            )()
        )

        disc1_records = pipeline.list_review_records("disc1")

        assert len(disc1_records) == 1
        assert disc1_records[0].discovery_id == "disc1"


class TestDiscoveryPipelinePromote:
    """Tests for DiscoveryPipeline promote_proposal."""

    def test_promote_proposal_calls_promoter_callback(self, tmp_path: Path):
        """promote_proposal should call promoter callback with approved record."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
            review_storage_path=tmp_path / "records.json",
        )
        manager = pipeline.review_manager
        records = manager.create_records(
            type(
                "Report",
                (),
                {
                    "discovery_id": "disc123",
                    "timestamp": datetime.now(),
                    "class_proposals": NovelClassAnalysis(
                        proposed_classes=[
                            ClassProposal(
                                name="X",
                                description="d",
                                confidence=0.9,
                                sample_count=1,
                                example_samples=["a"],
                                justification="j",
                            )
                        ],
                        rejected_as_noise=[],
                        analysis_summary="",
                        cluster_count=1,
                        model_used="test",
                    ),
                    "diagnostics": {},
                },
            )()
        )
        promoter_mock = MagicMock()

        pipeline.promote_proposal(records[0].review_id, promoter=promoter_mock)

        promoter_mock.assert_called_once()

    def test_promote_proposal_without_callback(self, tmp_path: Path):
        """promote_proposal should work without a promoter callback."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
            review_storage_path=tmp_path / "records.json",
        )
        manager = pipeline.review_manager
        records = manager.create_records(
            type(
                "Report",
                (),
                {
                    "discovery_id": "disc123",
                    "timestamp": datetime.now(),
                    "class_proposals": NovelClassAnalysis(
                        proposed_classes=[
                            ClassProposal(
                                name="X",
                                description="d",
                                confidence=0.9,
                                sample_count=1,
                                example_samples=["a"],
                                justification="j",
                            )
                        ],
                        rejected_as_noise=[],
                        analysis_summary="",
                        cluster_count=1,
                        model_used="test",
                    ),
                    "diagnostics": {},
                },
            )()
        )

        promoted = pipeline.promote_proposal(records[0].review_id)

        assert promoted.state == "promoted"
        assert promoted.promoted_at is not None


class TestDiscoveryPipelineApprove:
    """Tests for DiscoveryPipeline approve_proposal."""

    def test_approve_proposal_with_notes(self, tmp_path: Path):
        """approve_proposal should set notes when provided."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
            review_storage_path=tmp_path / "records.json",
        )
        manager = pipeline.review_manager
        records = manager.create_records(
            type(
                "Report",
                (),
                {
                    "discovery_id": "disc123",
                    "timestamp": datetime.now(),
                    "class_proposals": NovelClassAnalysis(
                        proposed_classes=[
                            ClassProposal(
                                name="X",
                                description="d",
                                confidence=0.9,
                                sample_count=1,
                                example_samples=["a"],
                                justification="j",
                            )
                        ],
                        rejected_as_noise=[],
                        analysis_summary="",
                        cluster_count=1,
                        model_used="test",
                    ),
                    "diagnostics": {},
                },
            )()
        )

        approved = pipeline.approve_proposal(records[0].review_id, notes="looks good")

        assert approved.state == "approved"
        assert approved.notes == "looks good"

    def test_approve_proposal_unknown_id_raises_key_error(self, tmp_path: Path):
        """approve_proposal should raise KeyError for unknown review_id."""
        pipeline = DiscoveryPipeline(
            matcher=_build_trained_matcher(),
            auto_save=False,
            review_storage_path=tmp_path / "records.json",
        )

        with pytest.raises(KeyError):
            pipeline.approve_proposal("nonexistent_id")
