"""Tests for ProposalReviewManager lifecycle and error handling."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from novelentitymatcher.novelty.schemas import (
    ClassProposal,
    NovelClassAnalysis,
    ProposalReviewRecord,
)
from novelentitymatcher.novelty.storage.review import ProposalReviewManager


def _make_report(discovery_id: str = "disc123", with_proposals: bool = True):
    """Helper to create a mock discovery report."""
    if with_proposals:
        class_proposals = NovelClassAnalysis(
            proposed_classes=[
                ClassProposal(
                    name="Quantum Biology",
                    description="Quantum effects in biological systems",
                    confidence=0.91,
                    sample_count=2,
                    example_samples=["a", "b"],
                    justification="coherent",
                )
            ],
            rejected_as_noise=[],
            analysis_summary="One cluster",
            cluster_count=1,
            model_used="test-model",
        )
    else:
        class_proposals = NovelClassAnalysis(
            proposed_classes=[],
            rejected_as_noise=[],
            analysis_summary="",
            cluster_count=0,
            model_used="",
        )

    class MockReport:
        pass

    MockReport.discovery_id = discovery_id
    MockReport.timestamp = datetime.now()
    MockReport.diagnostics = {}
    MockReport.class_proposals = class_proposals

    return MockReport()


def test_create_records_empty_proposals(tmp_path: Path):
    """create_records should return empty list when no proposals."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    report = _make_report(with_proposals=False)

    records = manager.create_records(report)

    assert records == []


def test_create_records_saves_to_storage(tmp_path: Path):
    """create_records should persist records to storage."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    report = _make_report()

    records = manager.create_records(report)

    assert len(records) == 1
    assert manager.storage_path.exists()


def test_create_records_assigns_unique_review_ids(tmp_path: Path):
    """Each record should get a unique review_id."""
    manager = ProposalReviewManager(tmp_path / "records.json")

    class MultiProposalReport:
        discovery_id = "disc1"
        timestamp = datetime.now()
        diagnostics = {}
        class_proposals = NovelClassAnalysis(
            proposed_classes=[
                ClassProposal(
                    name=f"Class{i}",
                    description="desc",
                    confidence=0.9,
                    sample_count=1,
                    example_samples=["a"],
                    justification="j",
                )
                for i in range(3)
            ],
            rejected_as_noise=[],
            analysis_summary="",
            cluster_count=1,
            model_used="test",
        )

    records = manager.create_records(MultiProposalReport())

    review_ids = [r.review_id for r in records]
    assert len(review_ids) == len(set(review_ids))


def test_list_records_nonexistent_storage(tmp_path: Path):
    """list_records should return empty list when storage doesn't exist."""
    manager = ProposalReviewManager(tmp_path / "nonexistent.json")

    records = manager.list_records()

    assert records == []


def test_list_records_filter_by_discovery_id(tmp_path: Path):
    """list_records should filter by discovery_id when provided."""
    manager = ProposalReviewManager(tmp_path / "records.json")

    manager.create_records(_make_report("disc1"))
    manager.create_records(_make_report("disc2"))

    disc1_records = manager.list_records("disc1")
    disc2_records = manager.list_records("disc2")
    all_records = manager.list_records()

    assert len(disc1_records) == 1
    assert len(disc2_records) == 1
    assert disc1_records[0].discovery_id == "disc1"
    assert disc2_records[0].discovery_id == "disc2"
    assert len(all_records) == 2


def test_update_state_to_approved_sets_reviewed_at(tmp_path: Path):
    """update_state to approved should set reviewed_at timestamp."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())

    updated = manager.update_state(records[0].review_id, "approved")

    assert updated.state == "approved"
    assert updated.reviewed_at is not None


def test_update_state_to_rejected_sets_reviewed_at(tmp_path: Path):
    """update_state to rejected should set reviewed_at timestamp."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())

    updated = manager.update_state(records[0].review_id, "rejected")

    assert updated.state == "rejected"
    assert updated.reviewed_at is not None


def test_update_state_to_promoted_sets_promoted_at(tmp_path: Path):
    """update_state to promoted should work after approval."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())
    manager.update_state(records[0].review_id, "approved")

    updated = manager.update_state(records[0].review_id, "promoted")

    assert updated.state == "promoted"
    assert updated.promoted_at is not None
    assert updated.reviewed_at is not None


def test_update_state_unknown_review_id_raises_key_error(tmp_path: Path):
    """update_state should raise KeyError for unknown review_id."""
    manager = ProposalReviewManager(tmp_path / "records.json")

    with pytest.raises(KeyError, match="Unknown review_id"):
        manager.update_state("nonexistent_id", "approved")


def test_update_state_preserves_notes_when_not_provided(tmp_path: Path):
    """update_state should preserve existing notes when not overridden."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())
    review_id = records[0].review_id

    manager.update_state(review_id, "approved", notes="first review")
    updated = manager.update_state(review_id, "approved")

    assert updated.notes == "first review"


def test_update_state_overwrites_notes_when_provided(tmp_path: Path):
    """update_state should overwrite notes when explicitly provided."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())
    review_id = records[0].review_id

    manager.update_state(review_id, "approved", notes="first review")
    updated = manager.update_state(review_id, "approved", notes="updated review")

    assert updated.notes == "updated review"


def test_promote_calls_promoter_callback(tmp_path: Path):
    """promote should call the promoter callback with approved record."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())
    review_id = records[0].review_id

    promoter_mock = MagicMock()

    result = manager.promote(review_id, promoter=promoter_mock)

    promoter_mock.assert_called_once()
    approved_record = promoter_mock.call_args[0][0]
    assert approved_record.state == "approved"
    assert result.review_record.state == "promoted"


def test_promote_without_callback(tmp_path: Path):
    """promote should work without a promoter callback."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())

    result = manager.promote(records[0].review_id)

    assert result.review_record.state == "promoted"
    assert len(result.entities_added) == 1
    assert result.entities_added[0]["name"] == "Quantum Biology"
    assert result.index_updated is False
    assert result.retrain_required is False


def test_promote_rejected_record_raises_value_error(tmp_path: Path):
    """Rejected proposals should not silently re-enter promotion."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())
    review_id = records[0].review_id

    manager.update_state(review_id, "rejected")

    with pytest.raises(ValueError, match="rejected -> promoted"):
        manager.promote(review_id)


def test_promote_with_index_updater(tmp_path: Path):
    """promote should call index_updater with new entities."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())

    updater_mock = MagicMock()
    result = manager.promote(records[0].review_id, index_updater=updater_mock)

    updater_mock.assert_called_once()
    assert result.index_updated


def test_promote_with_retrain_callback(tmp_path: Path):
    """promote should call retrain_callback when provided."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())

    retrain_mock = MagicMock()
    result = manager.promote(records[0].review_id, retrain_callback=retrain_mock)

    retrain_mock.assert_called_once()
    assert result.retrain_required


def test_promote_with_entities_list(tmp_path: Path):
    """promote should extend the provided entities list."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())

    entities: list[dict] = []
    manager.promote(records[0].review_id, entities=entities)

    assert len(entities) == 1
    assert entities[0]["name"] == "Quantum Biology"


def test_promote_with_index_update(tmp_path: Path):
    """promote_with_index_update should update matcher entities."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())

    class FakeMatcher:
        def __init__(self):
            self.entities: list[dict] = []
            self.reindex_called = False

        def reindex(self):
            self.reindex_called = True

    matcher = FakeMatcher()
    result = manager.promote_with_index_update(records[0].review_id, matcher)

    assert result.review_record.state == "promoted"
    assert len(matcher.entities) == 1
    assert matcher.entities[0]["name"] == "Quantum Biology"
    assert matcher.reindex_called


def test_promote_with_index_update_fallback_calls_fit_without_entities(tmp_path: Path):
    """Fallback path should not pass entity dictionaries to fit()."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())

    class FakeMatcher:
        def __init__(self):
            self.entities: list[dict] = []
            self.fit_called = False

        def fit(self):
            self.fit_called = True

    matcher = FakeMatcher()
    result = manager.promote_with_index_update(records[0].review_id, matcher)

    assert result.review_record.state == "promoted"
    assert len(matcher.entities) == 1
    assert matcher.entities[0]["name"] == "Quantum Biology"
    assert matcher.fit_called


def test_update_state_pending_to_promoted_raises_value_error(tmp_path: Path):
    """Promotion should require approval flow instead of skipping review."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())

    with pytest.raises(ValueError, match="pending_review -> promoted"):
        manager.update_state(records[0].review_id, "promoted")


def test_update_state_rejected_to_approved_raises_value_error(tmp_path: Path):
    """Rejected proposals should remain terminal unless recreated explicitly."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    records = manager.create_records(_make_report())
    review_id = records[0].review_id

    manager.update_state(review_id, "rejected")

    with pytest.raises(ValueError, match="rejected -> approved"):
        manager.update_state(review_id, "approved")


def test_save_records_merges_with_existing(tmp_path: Path):
    """save_records should merge with existing records by review_id."""
    manager = ProposalReviewManager(tmp_path / "records.json")

    record1 = ProposalReviewRecord(
        review_id="abc123",
        discovery_id="disc1",
        proposal_index=0,
        proposal_name="Class1",
        proposal=ClassProposal(
            name="Class1",
            description="d",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        ),
    )
    record2 = ProposalReviewRecord(
        review_id="def456",
        discovery_id="disc2",
        proposal_index=0,
        proposal_name="Class2",
        proposal=ClassProposal(
            name="Class2",
            description="d",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        ),
    )

    manager.save_records([record1])
    manager.save_records([record2])

    all_records = manager.list_records()
    assert len(all_records) == 2


def test_save_records_overwrites_existing_by_review_id(tmp_path: Path):
    """save_records should overwrite existing record with same review_id."""
    manager = ProposalReviewManager(tmp_path / "records.json")

    record1 = ProposalReviewRecord(
        review_id="abc123",
        discovery_id="disc1",
        proposal_index=0,
        proposal_name="Original",
        proposal=ClassProposal(
            name="Original",
            description="d",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        ),
    )
    record1_updated = ProposalReviewRecord(
        review_id="abc123",
        discovery_id="disc1",
        proposal_index=0,
        proposal_name="Updated",
        state="approved",
        proposal=ClassProposal(
            name="Updated",
            description="d",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        ),
    )

    manager.save_records([record1])
    manager.save_records([record1_updated])

    records = manager.list_records()
    assert len(records) == 1
    assert records[0].proposal_name == "Updated"
    assert records[0].state == "approved"


def test_read_storage_invalid_json_raises_value_error(tmp_path: Path):
    """_read_storage should raise ValueError for invalid JSON."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    manager.storage_path.parent.mkdir(parents=True, exist_ok=True)
    manager.storage_path.write_text("not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        manager.list_records()


def test_read_storage_non_list_raises_value_error(tmp_path: Path):
    """_read_storage should raise ValueError when storage is not a list."""
    manager = ProposalReviewManager(tmp_path / "records.json")
    manager.storage_path.parent.mkdir(parents=True, exist_ok=True)
    manager.storage_path.write_text('{"key": "value"}', encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON list"):
        manager.list_records()
