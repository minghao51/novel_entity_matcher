"""Tests for ProposalConflictResolver."""

from novelentitymatcher.novelty.proposal.conflict_resolver import (
    ProposalConflictResolver,
)
from novelentitymatcher.novelty.schemas.models import ClassProposal


def _make_proposal(
    name="Test",
    confidence=0.8,
    cluster_ids=None,
    examples=None,
    sample_count=10,
):
    return ClassProposal(
        name=name,
        description="",
        confidence=confidence,
        sample_count=sample_count,
        example_samples=examples or ["example"],
        justification="",
        source_cluster_ids=cluster_ids or [1, 2, 3],
    )


class TestProposalConflictResolver:
    def test_no_conflicts(self):
        resolver = ProposalConflictResolver()
        a = _make_proposal(name="Technology", cluster_ids=[1, 2])
        b = _make_proposal(name="Healthcare", cluster_ids=[3, 4])
        conflicts = resolver.detect_conflicts([a, b])
        assert len(conflicts) == 0

    def test_conflict_detected_via_cluster_overlap(self):
        resolver = ProposalConflictResolver()
        a = _make_proposal(name="Tech Startup", cluster_ids=[1, 2, 3])
        b = _make_proposal(name="SaaS Company", cluster_ids=[2, 3, 4])
        conflicts = resolver.detect_conflicts([a, b])
        assert len(conflicts) == 1
        assert conflicts[0].overlap_type in ("partial", "duplicate")

    def test_duplicate_via_name_similarity(self):
        resolver = ProposalConflictResolver()
        a = _make_proposal(name="Tech Startup", cluster_ids=[1], examples=["a"])
        b = _make_proposal(name="techstartup", cluster_ids=[2], examples=["b"])
        conflicts = resolver.detect_conflicts([a, b])
        assert len(conflicts) == 1
        assert conflicts[0].overlap_type == "duplicate"

    def test_subset_detected(self):
        resolver = ProposalConflictResolver()
        a = _make_proposal(name="Tech", cluster_ids=[1, 2, 3])
        b = _make_proposal(name="Tech AI", cluster_ids=[1, 2])
        conflicts = resolver.detect_conflicts([a, b])
        assert len(conflicts) == 1
        assert conflicts[0].overlap_type == "nested"

    def test_superset_detected(self):
        resolver = ProposalConflictResolver()
        a = _make_proposal(name="Tech AI", cluster_ids=[1, 2])
        b = _make_proposal(name="Tech", cluster_ids=[1, 2, 3])
        conflicts = resolver.detect_conflicts([a, b])
        assert len(conflicts) == 1
        assert conflicts[0].overlap_type == "nested"

    def test_resolve_removes_lower_confidence_duplicate(self):
        resolver = ProposalConflictResolver()
        high = _make_proposal(name="Tech", confidence=0.9, cluster_ids=[1, 2])
        low = _make_proposal(name="tech", confidence=0.4, cluster_ids=[1, 2])
        resolved = resolver.resolve([high, low])
        assert len(resolved) == 1
        assert resolved[0].confidence == 0.9

    def test_resolve_no_overlap_returns_all(self):
        resolver = ProposalConflictResolver()
        a = _make_proposal(name="Tech", cluster_ids=[1])
        b = _make_proposal(name="Health", cluster_ids=[2])
        resolved = resolver.resolve([a, b])
        assert len(resolved) == 2

    def test_resolve_nested_keeps_larger(self):
        resolver = ProposalConflictResolver()
        larger = _make_proposal(name="Tech", cluster_ids=[1, 2, 3])
        smaller = _make_proposal(name="Tech AI", cluster_ids=[1, 2])
        resolved = resolver.resolve([larger, smaller])
        assert larger in resolved
        assert smaller not in resolved

    def test_resolve_partial_keeps_higher_confidence(self):
        resolver = ProposalConflictResolver()
        high = _make_proposal(name="Tech A", confidence=0.9, cluster_ids=[1, 2])
        low = _make_proposal(name="Tech B", confidence=0.5, cluster_ids=[2, 3])
        resolved = resolver.resolve([high, low])
        assert len(resolved) == 1
        assert resolved[0].name == "Tech A"

    def test_empty_proposals(self):
        resolver = ProposalConflictResolver()
        assert resolver.resolve([]) == []
