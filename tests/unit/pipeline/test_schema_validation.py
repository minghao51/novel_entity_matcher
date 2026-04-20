"""Tests for schema model validation and state transitions."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from novelentitymatcher.novelty.schemas import (
    ClassProposal,
    ClusterEvidence,
    DiscoveryCluster,
    NovelClassAnalysis,
    NovelClassDiscoveryReport,
    NovelSampleMetadata,
    NovelSampleReport,
    ProposalReviewRecord,
)


class TestNovelSampleMetadata:
    """Validation tests for NovelSampleMetadata."""

    def test_valid_sample(self):
        """Valid sample should create without errors."""
        sample = NovelSampleMetadata(
            text="quantum physics",
            index=0,
            confidence=0.85,
            predicted_class="physics",
            novelty_score=0.92,
        )
        assert sample.text == "quantum physics"
        assert sample.confidence == 0.85

    def test_confidence_not_clamped_to_valid_range(self):
        """Confidence should accept any float value (model_config allows arbitrary)."""
        sample = NovelSampleMetadata(
            text="test",
            index=0,
            confidence=1.5,
            predicted_class="physics",
        )
        assert sample.confidence == 1.5

    def test_optional_fields_default_to_none(self):
        """Optional score fields should default to None."""
        sample = NovelSampleMetadata(
            text="test",
            index=0,
            confidence=0.5,
            predicted_class="x",
        )
        assert sample.novelty_score is None
        assert sample.margin_score is None
        assert sample.entropy_score is None
        assert sample.uncertainty_score is None

    def test_signals_default_to_empty_dict(self):
        """signals should default to empty dict."""
        sample = NovelSampleMetadata(
            text="test",
            index=0,
            confidence=0.5,
            predicted_class="x",
        )
        assert sample.signals == {}

    def test_metrics_default_to_empty_dict(self):
        """metrics should default to empty dict."""
        sample = NovelSampleMetadata(
            text="test",
            index=0,
            confidence=0.5,
            predicted_class="x",
        )
        assert sample.metrics == {}


class TestClassProposal:
    """Validation tests for ClassProposal."""

    def test_valid_proposal(self):
        """Valid proposal should create without errors."""
        proposal = ClassProposal(
            name="Quantum Biology",
            description="Quantum effects in biological systems",
            confidence=0.91,
            sample_count=5,
            example_samples=["sample1", "sample2"],
            justification="coherent concept",
        )
        assert proposal.name == "Quantum Biology"

    def test_confidence_clamped_to_valid_range(self):
        """Confidence with ge=0.0 and le=1.0 should enforce range."""
        with pytest.raises(ValidationError):
            ClassProposal(
                name="Test",
                description="desc",
                confidence=1.5,
                sample_count=1,
                example_samples=["a"],
                justification="j",
            )

    def test_confidence_negative_raises_error(self):
        """Negative confidence should raise ValidationError."""
        with pytest.raises(ValidationError):
            ClassProposal(
                name="Test",
                description="desc",
                confidence=-0.1,
                sample_count=1,
                example_samples=["a"],
                justification="j",
            )

    def test_sample_count_must_be_non_negative(self):
        """sample_count with ge=0 should enforce non-negative."""
        with pytest.raises(ValidationError):
            ClassProposal(
                name="Test",
                description="desc",
                confidence=0.9,
                sample_count=-1,
                example_samples=["a"],
                justification="j",
            )

    def test_source_cluster_ids_default_to_empty_list(self):
        """source_cluster_ids should default to empty list."""
        proposal = ClassProposal(
            name="Test",
            description="desc",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        )
        assert proposal.source_cluster_ids == []

    def test_provenance_default_to_empty_dict(self):
        """provenance should default to empty dict."""
        proposal = ClassProposal(
            name="Test",
            description="desc",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        )
        assert proposal.provenance == {}


class TestDiscoveryCluster:
    """Validation tests for DiscoveryCluster."""

    def test_valid_cluster(self):
        """Valid cluster should create without errors."""
        cluster = DiscoveryCluster(
            cluster_id=0,
            sample_indices=[0, 1, 2],
            sample_count=3,
            example_texts=["a", "b", "c"],
            keywords=["quantum", "biology"],
        )
        assert cluster.cluster_id == 0
        assert cluster.sample_count == 3

    def test_sample_count_must_be_non_negative(self):
        """sample_count with ge=0 should enforce non-negative."""
        with pytest.raises(ValidationError):
            DiscoveryCluster(
                cluster_id=0,
                sample_indices=[],
                sample_count=-1,
                example_texts=[],
                keywords=[],
            )

    def test_evidence_default_to_none(self):
        """evidence should default to None."""
        cluster = DiscoveryCluster(
            cluster_id=0,
            sample_indices=[],
            sample_count=0,
            example_texts=[],
            keywords=[],
        )
        assert cluster.evidence is None


class TestClusterEvidence:
    """Validation tests for ClusterEvidence."""

    def test_valid_evidence(self):
        """Valid evidence should create without errors."""
        evidence = ClusterEvidence(
            keywords=["quantum", "bio"],
            representative_examples=["example1"],
            sample_indices=[0, 1],
            predicted_classes=["physics"],
            confidence_summary={"mean": 0.85},
        )
        assert evidence.keywords == ["quantum", "bio"]

    def test_all_fields_optional(self):
        """All fields should have defaults."""
        evidence = ClusterEvidence()
        assert evidence.keywords == []
        assert evidence.representative_examples == []


class TestProposalReviewRecord:
    """Validation tests for ProposalReviewRecord."""

    def test_valid_record(self):
        """Valid record should create without errors."""
        proposal = ClassProposal(
            name="Test",
            description="desc",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        )
        record = ProposalReviewRecord(
            review_id="abc123",
            discovery_id="disc456",
            proposal_index=0,
            proposal_name="Test",
            proposal=proposal,
        )
        assert record.review_id == "abc123"
        assert record.state == "pending_review"

    def test_state_defaults_to_pending_review(self):
        """state should default to pending_review."""
        proposal = ClassProposal(
            name="Test",
            description="desc",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        )
        record = ProposalReviewRecord(
            review_id="abc",
            discovery_id="disc",
            proposal_index=0,
            proposal_name="Test",
            proposal=proposal,
        )
        assert record.state == "pending_review"

    def test_proposal_index_must_be_non_negative(self):
        """proposal_index with ge=0 should enforce non-negative."""
        proposal = ClassProposal(
            name="Test",
            description="desc",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        )
        with pytest.raises(ValidationError):
            ProposalReviewRecord(
                review_id="abc",
                discovery_id="disc",
                proposal_index=-1,
                proposal_name="Test",
                proposal=proposal,
            )

    def test_invalid_state_raises_validation_error(self):
        """Invalid state value should raise ValidationError."""
        proposal = ClassProposal(
            name="Test",
            description="desc",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        )
        with pytest.raises(ValidationError):
            ProposalReviewRecord(
                review_id="abc",
                discovery_id="disc",
                proposal_index=0,
                proposal_name="Test",
                proposal=proposal,
                state="invalid_state",
            )

    def test_valid_state_values(self):
        """All valid ReviewState values should be accepted."""
        proposal = ClassProposal(
            name="Test",
            description="desc",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        )
        for state in ["pending_review", "approved", "rejected", "promoted"]:
            record = ProposalReviewRecord(
                review_id=f"abc_{state}",
                discovery_id="disc",
                proposal_index=0,
                proposal_name="Test",
                proposal=proposal,
                state=state,
            )
            assert record.state == state


class TestNovelClassAnalysis:
    """Validation tests for NovelClassAnalysis."""

    def test_valid_analysis(self):
        """Valid analysis should create without errors."""
        analysis = NovelClassAnalysis(
            proposed_classes=[
                ClassProposal(
                    name="Test",
                    description="desc",
                    confidence=0.9,
                    sample_count=1,
                    example_samples=["a"],
                    justification="j",
                )
            ],
            rejected_as_noise=["noise1"],
            analysis_summary="One coherent cluster",
            cluster_count=1,
            model_used="test-model",
        )
        assert len(analysis.proposed_classes) == 1
        assert analysis.cluster_count == 1

    def test_cluster_count_must_be_non_negative(self):
        """cluster_count with ge=0 should enforce non-negative."""
        with pytest.raises(ValidationError):
            NovelClassAnalysis(
                proposed_classes=[],
                rejected_as_noise=[],
                analysis_summary="",
                cluster_count=-1,
                model_used="test",
            )

    def test_validation_errors_default_to_empty_list(self):
        """validation_errors should default to empty list."""
        analysis = NovelClassAnalysis(
            proposed_classes=[],
            rejected_as_noise=[],
            analysis_summary="",
            cluster_count=0,
            model_used="test",
        )
        assert analysis.validation_errors == []


class TestNovelSampleReport:
    """Validation tests for NovelSampleReport."""

    def test_valid_report(self):
        """Valid report should create without errors."""
        report = NovelSampleReport(
            novel_samples=[
                NovelSampleMetadata(
                    text="test",
                    index=0,
                    confidence=0.5,
                    predicted_class="x",
                )
            ],
            detection_strategies=["confidence"],
            config={"threshold": 0.5},
            signal_counts={"confidence": 1},
        )
        assert len(report.novel_samples) == 1
        assert report.detection_strategies == ["confidence"]

    def test_defaults(self):
        """Default values should be applied."""
        report = NovelSampleReport()
        assert report.novel_samples == []
        assert report.detection_strategies == []
        assert report.config == {}
        assert report.signal_counts == {}


class TestNovelClassDiscoveryReport:
    """Validation tests for NovelClassDiscoveryReport."""

    def test_valid_report(self):
        """Valid report should create without errors."""
        report = NovelClassDiscoveryReport(
            discovery_id="disc123",
            novel_sample_report=NovelSampleReport(
                novel_samples=[
                    NovelSampleMetadata(
                        text="test",
                        index=0,
                        confidence=0.5,
                        predicted_class="x",
                    )
                ],
                detection_strategies=["confidence"],
            ),
        )
        assert report.discovery_id == "disc123"
        assert len(report.novel_sample_report.novel_samples) == 1

    def test_discovery_clusters_default_to_empty_list(self):
        """discovery_clusters should default to empty list."""
        report = NovelClassDiscoveryReport(
            discovery_id="disc123",
            novel_sample_report=NovelSampleReport(),
        )
        assert report.discovery_clusters == []

    def test_class_proposals_optional(self):
        """class_proposals should be optional (None by default)."""
        report = NovelClassDiscoveryReport(
            discovery_id="disc123",
            novel_sample_report=NovelSampleReport(),
        )
        assert report.class_proposals is None

    def test_review_records_default_to_empty_list(self):
        """review_records should default to empty list."""
        report = NovelClassDiscoveryReport(
            discovery_id="disc123",
            novel_sample_report=NovelSampleReport(),
        )
        assert report.review_records == []

    def test_diagnostics_default_to_empty_dict(self):
        """diagnostics should default to empty dict."""
        report = NovelClassDiscoveryReport(
            discovery_id="disc123",
            novel_sample_report=NovelSampleReport(),
        )
        assert report.diagnostics == {}

    def test_metadata_default_to_empty_dict(self):
        """metadata should default to empty dict."""
        report = NovelClassDiscoveryReport(
            discovery_id="disc123",
            novel_sample_report=NovelSampleReport(),
        )
        assert report.metadata == {}

    def test_timestamp_defaults_to_now(self):
        """timestamp should default to current time."""
        report = NovelClassDiscoveryReport(
            discovery_id="disc123",
            novel_sample_report=NovelSampleReport(),
        )
        assert isinstance(report.timestamp, datetime)


class TestReviewStateTransitions:
    """Tests for valid state transition sequences in ProposalReviewRecord."""

    def test_pending_to_approved_transition(self):
        """pending_review to approved should be valid."""
        proposal = ClassProposal(
            name="Test",
            description="desc",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        )
        record = ProposalReviewRecord(
            review_id="abc",
            discovery_id="disc",
            proposal_index=0,
            proposal_name="Test",
            proposal=proposal,
            state="pending_review",
        )
        record.state = "approved"
        assert record.state == "approved"

    def test_approved_to_promoted_transition(self):
        """approved to promoted should be valid."""
        proposal = ClassProposal(
            name="Test",
            description="desc",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        )
        record = ProposalReviewRecord(
            review_id="abc",
            discovery_id="disc",
            proposal_index=0,
            proposal_name="Test",
            proposal=proposal,
            state="approved",
        )
        record.state = "promoted"
        assert record.state == "promoted"

    def test_pending_to_rejected_transition(self):
        """pending_review to rejected should be valid."""
        proposal = ClassProposal(
            name="Test",
            description="desc",
            confidence=0.9,
            sample_count=1,
            example_samples=["a"],
            justification="j",
        )
        record = ProposalReviewRecord(
            review_id="abc",
            discovery_id="disc",
            proposal_index=0,
            proposal_name="Test",
            proposal=proposal,
            state="pending_review",
        )
        record.state = "rejected"
        assert record.state == "rejected"
