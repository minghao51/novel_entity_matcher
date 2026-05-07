"""Tests for DSPy integration in LLMClassProposer."""

import json
from unittest.mock import MagicMock, patch

import pytest

from novelentitymatcher.novelty.proposal.llm import LLMClassProposer
from novelentitymatcher.novelty.schemas import DiscoveryCluster


class TestLLMClassProposerDSPyIntegration:
    """Tests for DSPy module integration in LLMClassProposer."""

    @pytest.fixture
    def clusters(self):
        return [
            DiscoveryCluster(
                cluster_id=0,
                sample_indices=[0, 1],
                sample_count=2,
                example_texts=[
                    "quantum entanglement in photosynthesis",
                    "quantum computing applications",
                ],
                keywords=["quantum", "biology"],
            ),
            DiscoveryCluster(
                cluster_id=1,
                sample_indices=[2],
                sample_count=1,
                example_texts=["CRISPR gene editing efficiency"],
                keywords=["crispr", "gene"],
            ),
        ]

    def test_propose_from_clusters_uses_dspy_module(self, clusters):
        """When dspy_module is set, propose_from_clusters should use it."""
        mock_module = MagicMock()
        mock_module.predict.return_value = json.dumps(
            {
                "proposed_classes": [
                    {
                        "name": "Quantum Biology",
                        "description": "Intersection of quantum and biology",
                        "confidence": 0.92,
                        "sample_count": 2,
                        "example_samples": [
                            "quantum entanglement in photosynthesis",
                            "quantum computing applications",
                        ],
                        "justification": "Coherent cluster",
                        "source_cluster_ids": [0],
                    },
                    {
                        "name": "Gene Editing",
                        "description": "Gene editing techniques",
                        "confidence": 0.88,
                        "sample_count": 1,
                        "example_samples": ["CRISPR gene editing efficiency"],
                        "justification": "Clear gene editing focus",
                        "source_cluster_ids": [1],
                    },
                ],
                "rejected_as_noise": [],
                "analysis_summary": "Identified 2 novel classes",
                "cluster_count": 2,
            }
        )

        proposer = LLMClassProposer(
            primary_model="test-model",
            dspy_module=mock_module,
        )

        result = proposer.propose_from_clusters(
            discovery_clusters=clusters,
            existing_classes=["physics", "biology"],
            context="Scientific research",
        )

        assert len(result.proposed_classes) == 2
        assert result.proposed_classes[0].name == "Quantum Biology"
        assert result.proposed_classes[1].name == "Gene Editing"
        mock_module.predict.assert_called_once()

    def test_dspy_call_receives_correct_json(self, clusters):
        """Verify the DSPy module receives serialized cluster data."""
        mock_module = MagicMock()
        mock_module.predict.return_value = json.dumps(
            {
                "proposed_classes": [],
                "rejected_as_noise": [],
                "analysis_summary": "none",
                "cluster_count": 0,
            }
        )

        proposer = LLMClassProposer(
            primary_model="test-model",
            dspy_module=mock_module,
        )

        proposer.propose_from_clusters(
            discovery_clusters=clusters,
            existing_classes=["physics", "biology"],
            context="Science domain",
        )

        call_kwargs = mock_module.predict.call_args.kwargs
        assert "clusters_json" in call_kwargs

        clusters_data = json.loads(call_kwargs["clusters_json"])
        assert len(clusters_data) == 2
        assert clusters_data[0]["cluster_id"] == 0
        assert clusters_data[0]["sample_count"] == 2
        assert clusters_data[1]["cluster_id"] == 1

        assert call_kwargs["existing_classes"] == "physics, biology"
        assert call_kwargs["domain_context"] == "Science domain"

    def test_dspy_module_without_existing_classes(self, clusters):
        """Existing classes default to 'none' when list is empty."""
        mock_module = MagicMock()
        mock_module.predict.return_value = json.dumps(
            {
                "proposed_classes": [],
                "rejected_as_noise": [],
                "analysis_summary": "none",
                "cluster_count": 0,
            }
        )

        proposer = LLMClassProposer(
            primary_model="test-model",
            dspy_module=mock_module,
        )

        proposer.propose_from_clusters(
            discovery_clusters=clusters,
            existing_classes=[],
            context=None,
        )

        call_kwargs = mock_module.predict.call_args.kwargs
        assert call_kwargs["existing_classes"] == "none"
        assert call_kwargs["domain_context"] == ""

    def test_dspy_module_fallback_on_failure(self, clusters):
        """When DSPy module fails repeatedly, fallback analysis is returned."""
        mock_module = MagicMock()
        mock_module.predict.side_effect = RuntimeError("DSPy call failed")

        proposer = LLMClassProposer(
            primary_model="test-model",
            dspy_module=mock_module,
        )

        result = proposer.propose_from_clusters(
            discovery_clusters=clusters,
            existing_classes=["physics"],
        )

        assert result.model_used == "dspy-fallback"
        assert "Fallback" in result.analysis_summary
        assert result.proposed_classes == []
        assert result.cluster_count == 2

    def test_dspy_module_retries_on_failure(self, clusters):
        """DSPy module is retried up to max_retries + 1 times."""
        mock_module = MagicMock()
        mock_module.predict.side_effect = [
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            json.dumps(
                {
                    "proposed_classes": [
                        {
                            "name": "Recovered Class",
                            "description": "Recovered on retry",
                            "confidence": 0.7,
                            "sample_count": 1,
                            "example_samples": ["test"],
                            "justification": "recovered",
                        }
                    ],
                    "rejected_as_noise": [],
                    "analysis_summary": "recovered",
                    "cluster_count": 1,
                }
            ),
        ]

        proposer = LLMClassProposer(
            primary_model="test-model",
            max_tokens=4096,
            dspy_module=mock_module,
        )

        result = proposer.propose_from_clusters(
            discovery_clusters=clusters,
            existing_classes=["physics"],
            max_retries=3,
        )

        assert result.proposed_classes[0].name == "Recovered Class"
        assert mock_module.predict.call_count == 3

    def test_without_dspy_module_uses_normal_path(self, clusters):
        """Without dspy_module, propose_from_clusters uses normal prompt path."""
        proposer = LLMClassProposer(
            primary_model="test-model",
            dspy_module=None,
        )

        with patch.object(proposer, "_build_cluster_prompt", return_value="prompt"):
            with patch.object(
                proposer,
                "_run_structured_cluster_proposal",
                return_value=MagicMock(
                    proposed_classes=[],
                    rejected_as_noise=[],
                    analysis_summary="normal path",
                    cluster_count=0,
                    model_used="test",
                    validation_errors=[],
                    proposal_metadata={},
                ),
            ) as mock_run:
                proposer.propose_from_clusters(
                    discovery_clusters=clusters,
                    existing_classes=["physics"],
                )
                mock_run.assert_called_once()

    def test_dspy_module_bypasses_hierarchical(self, clusters):
        """DSPy module bypasses hierarchical split regardless of cluster count."""
        mock_module = MagicMock()
        mock_module.predict.return_value = json.dumps(
            {
                "proposed_classes": [],
                "rejected_as_noise": [],
                "analysis_summary": "dspy",
                "cluster_count": 0,
            }
        )

        proposer = LLMClassProposer(
            primary_model="test-model",
            max_clusters_per_summary=1,
            dspy_module=mock_module,
        )

        # Would trigger hierarchical without DSPy (2 clusters > max_clusters_per_summary=1)
        with patch.object(proposer, "_propose_hierarchical") as mock_hierarchical:
            proposer.propose_from_clusters(
                discovery_clusters=clusters,
                existing_classes=["physics"],
                hierarchical=True,
            )
            mock_hierarchical.assert_not_called()
            mock_module.predict.assert_called_once()

    def test_dspy_skipped_for_propose_classes(self, clusters):
        """propose_classes should not use dspy_module (only propose_from_clusters)."""
        mock_module = MagicMock()

        proposer = LLMClassProposer(
            primary_model="test-model",
            dspy_module=mock_module,
        )

        from novelentitymatcher.novelty.schemas import NovelSampleMetadata

        samples = [
            NovelSampleMetadata(
                text="test",
                index=0,
                confidence=0.5,
                predicted_class="physics",
                cluster_id=0,
                signals={},
            )
        ]

        with patch.object(proposer, "_call_litellm", return_value=("{}", "test")):
            with patch.object(proposer, "_parse_response") as mock_parse:
                mock_parse.return_value = MagicMock(
                    proposed_classes=[],
                    rejected_as_noise=[],
                    analysis_summary="skipped",
                    cluster_count=0,
                    model_used="test",
                    validation_errors=[],
                    proposal_metadata={},
                )
                proposer.propose_classes(
                    novel_samples=samples,
                    existing_classes=["physics"],
                )
                mock_module.predict.assert_not_called()
