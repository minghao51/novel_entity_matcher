"""Tests for DSPy proposal module."""

import json
from typing import ClassVar

from novelentitymatcher.novelty.proposal.dspy_module import (
    _extract_domain_context,
    _extract_existing_classes,
    _record_to_analysis_json,
    _record_to_cluster_json,
)


class TestRecordHelpers:
    """Tests for internal helper functions that don't require dspy."""

    SAMPLE_RECORD: ClassVar[dict] = {
        "review_id": "abc123",
        "state": "approved",
        "proposal": {
            "name": "Quantum Biology",
            "description": "Intersection of quantum physics and biological systems",
            "confidence": 0.92,
            "sample_count": 2,
            "example_samples": [
                "quantum entanglement in photosynthesis",
                "quantum computing applications",
            ],
            "justification": "Coherent cluster exploring quantum phenomena",
            "source_cluster_ids": [0, 1],
        },
        "provenance": {
            "cluster_ids": [0, 1],
            "keywords": ["quantum", "biology", "physics"],
            "diagnostics": {
                "existing_classes": ["physics", "cs", "biology"],
                "domain_context": "Scientific research papers",
            },
        },
    }

    def test_record_to_cluster_json(self):
        result = _record_to_cluster_json(self.SAMPLE_RECORD)
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["cluster_id"] == 0
        assert data[0]["sample_count"] == 2
        assert data[0]["keywords"] == ["quantum", "biology", "physics"]

    def test_record_to_cluster_json_empty_cluster_ids(self):
        record = {
            "proposal": {
                "name": "Test",
                "sample_count": 3,
                "example_samples": ["a", "b", "c"],
            },
            "provenance": {},
        }
        result = _record_to_cluster_json(record)
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["cluster_id"] == 0
        assert data[0]["sample_count"] == 3

    def test_record_to_cluster_json_missing_proposal(self):
        record = {"provenance": {"cluster_ids": [5]}}
        result = _record_to_cluster_json(record)
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["cluster_id"] == 5
        assert data[0]["sample_count"] == 0

    def test_record_to_analysis_json(self):
        result = _record_to_analysis_json(self.SAMPLE_RECORD)
        data = json.loads(result)
        assert len(data["proposed_classes"]) == 1
        assert data["proposed_classes"][0]["name"] == "Quantum Biology"
        assert data["proposed_classes"][0]["confidence"] == 0.92
        assert data["cluster_count"] == 1

    def test_extract_existing_classes(self):
        result = _extract_existing_classes(self.SAMPLE_RECORD)
        assert result == "physics, cs, biology"

    def test_extract_existing_classes_empty(self):
        record = {"provenance": {}}
        assert _extract_existing_classes(record) == "unknown"

    def test_extract_domain_context(self):
        result = _extract_domain_context(self.SAMPLE_RECORD)
        assert result == "Scientific research papers"

    def test_extract_domain_context_empty(self):
        record = {"provenance": {}}
        assert _extract_domain_context(record) == ""


class TestReviewRecordsToExamples:
    """Tests that require dspy to be importable."""

    SAMPLE_APPROVED: ClassVar[dict] = {
        "review_id": "abc",
        "state": "approved",
        "proposal": {
            "name": "Quantum Biology",
            "description": "Quantum + biology",
            "confidence": 0.9,
            "sample_count": 3,
            "example_samples": ["sample1", "sample2"],
            "justification": "coherent",
            "source_cluster_ids": [0],
        },
        "provenance": {
            "cluster_ids": [0],
            "keywords": ["quantum"],
            "diagnostics": {
                "existing_classes": ["physics"],
                "domain_context": "science",
            },
        },
    }
    SAMPLE_REJECTED: ClassVar[dict] = {
        "review_id": "def",
        "state": "rejected",
        "proposal": {
            "name": "Noise Cluster",
            "description": "Too diverse",
            "confidence": 0.2,
            "sample_count": 2,
            "example_samples": ["noise1", "noise2"],
            "justification": "not coherent",
            "source_cluster_ids": [5],
        },
        "provenance": {
            "cluster_ids": [5],
            "keywords": [],
            "diagnostics": {
                "existing_classes": ["biology"],
                "domain_context": "",
            },
        },
    }

    def test_approved_records_become_examples(self):
        from novelentitymatcher.novelty.proposal.dspy_module import (
            review_records_to_examples,
        )

        examples = review_records_to_examples(
            approved_records=[self.SAMPLE_APPROVED],
            rejected_records=None,
        )
        assert len(examples) == 1
        example = examples[0]
        assert example.clusters_json is not None
        assert example.analysis_json is not None
        data = json.loads(example.analysis_json)
        assert data["proposed_classes"][0]["name"] == "Quantum Biology"

    def test_rejected_records_become_empty_examples(self):
        from novelentitymatcher.novelty.proposal.dspy_module import (
            review_records_to_examples,
        )

        examples = review_records_to_examples(
            approved_records=[self.SAMPLE_APPROVED],
            rejected_records=[self.SAMPLE_REJECTED],
        )
        assert len(examples) == 2
        rejected_example = examples[1]
        data = json.loads(rejected_example.analysis_json)
        assert data["proposed_classes"] == []
        assert data["rejected_as_noise"] == ["all"]


class TestProposalMetric:
    """Tests for the GEPA proposal metric."""

    def test_exact_match(self):
        from novelentitymatcher.novelty.proposal.dspy_module import (
            proposal_metric,
            review_records_to_examples,
        )

        approved = {
            "review_id": "abc",
            "state": "approved",
            "proposal": {
                "name": "Quantum Biology",
                "description": "test",
                "confidence": 0.9,
                "sample_count": 3,
                "example_samples": ["a"],
                "justification": "test",
                "source_cluster_ids": [0],
            },
            "provenance": {
                "cluster_ids": [0],
                "keywords": [],
                "diagnostics": {"existing_classes": [], "domain_context": ""},
            },
        }
        examples = review_records_to_examples(approved_records=[approved])

        # expected = the example itself (perfect match)
        expected = examples[0]

        class MockPrediction:
            analysis_json = expected.analysis_json

        score = proposal_metric(MockPrediction(), expected)
        assert score == 1.0

    def test_no_match(self):
        from novelentitymatcher.novelty.proposal.dspy_module import (
            proposal_metric,
            review_records_to_examples,
        )

        approved = {
            "review_id": "abc",
            "state": "approved",
            "proposal": {
                "name": "Quantum Biology",
                "description": "test",
                "confidence": 0.9,
                "sample_count": 3,
                "example_samples": ["a"],
                "justification": "test",
                "source_cluster_ids": [0],
            },
            "provenance": {
                "cluster_ids": [0],
                "keywords": [],
                "diagnostics": {"existing_classes": [], "domain_context": ""},
            },
        }
        examples = review_records_to_examples(approved_records=[approved])
        expected = examples[0]

        class MockPrediction:
            analysis_json = json.dumps(
                {
                    "proposed_classes": [
                        {
                            "name": "Something Completely Different",
                            "description": "wrong",
                            "confidence": 0.5,
                            "sample_count": 1,
                            "example_samples": ["x"],
                            "justification": "wrong",
                        }
                    ],
                    "rejected_as_noise": [],
                    "analysis_summary": "different",
                    "cluster_count": 1,
                }
            )

        score = proposal_metric(MockPrediction(), expected)
        assert score == 0.0

    def test_partial_match(self):
        from novelentitymatcher.novelty.proposal.dspy_module import (
            proposal_metric,
            review_records_to_examples,
        )

        approved = {
            "review_id": "abc",
            "state": "approved",
            "proposal": {
                "name": "Quantum Biology",
                "description": "test",
                "confidence": 0.9,
                "sample_count": 3,
                "example_samples": ["a"],
                "justification": "test",
                "source_cluster_ids": [0],
            },
            "provenance": {
                "cluster_ids": [0],
                "keywords": [],
                "diagnostics": {"existing_classes": [], "domain_context": ""},
            },
        }
        examples = review_records_to_examples(approved_records=[approved])
        expected = examples[0]

        # Predict both the correct class + an extra one
        expected_data = json.loads(expected.analysis_json)
        expected_data["proposed_classes"].append(
            {
                "name": "Extra Class",
                "description": "extra",
                "confidence": 0.5,
                "sample_count": 1,
                "example_samples": ["x"],
                "justification": "extra",
            }
        )

        class MockPrediction:
            analysis_json = json.dumps(expected_data)

        score = proposal_metric(MockPrediction(), expected)
        # Jaccard: {Quantum Biology, Extra Class} ∩ {Quantum Biology} = 1, union = 2 => 0.5
        assert score == 0.5

    def test_both_empty(self):
        from novelentitymatcher.novelty.proposal.dspy_module import (
            proposal_metric,
        )

        class MockPrediction:
            analysis_json = json.dumps(
                {
                    "proposed_classes": [],
                    "rejected_as_noise": [],
                    "analysis_summary": "empty",
                    "cluster_count": 0,
                }
            )

        expected_data = json.dumps(
            {
                "proposed_classes": [],
                "rejected_as_noise": [],
                "analysis_summary": "empty",
                "cluster_count": 0,
            }
        )

        class MockExample:
            analysis_json = expected_data

        score = proposal_metric(MockPrediction(), MockExample())
        assert score == 1.0

    def test_invalid_json_returns_zero(self):
        from novelentitymatcher.novelty.proposal.dspy_module import (
            proposal_metric,
        )

        class MockPrediction:
            analysis_json = "not valid json"

        class MockExample:
            analysis_json = "also not valid"

        score = proposal_metric(MockPrediction(), MockExample())
        assert score == 0.0
