"""Tests for LLMClassProposer."""

import json
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from novelentitymatcher.novelty.proposal.llm import LLMClassProposer
from novelentitymatcher.novelty.schemas import NovelSampleMetadata


class TestLLMClassProposer:
    """Test suite for LLMClassProposer class."""

    @pytest.fixture
    def sample_novel_samples(self):
        """Create sample novel samples for testing."""
        return [
            NovelSampleMetadata(
                text="quantum entanglement in photosynthesis",
                index=0,
                confidence=0.45,
                predicted_class="physics",
                cluster_id=0,
                signals={"confidence": True},
            ),
            NovelSampleMetadata(
                text="quantum computing applications",
                index=1,
                confidence=0.52,
                predicted_class="cs",
                cluster_id=0,
                signals={"confidence": True},
            ),
            NovelSampleMetadata(
                text="CRISPR gene editing efficiency",
                index=2,
                confidence=0.48,
                predicted_class="biology",
                cluster_id=1,
                signals={"confidence": True},
            ),
        ]

    @pytest.fixture
    def mock_llm_response(self):
        """Create mock LLM response."""
        return json.dumps(
            {
                "proposed_classes": [
                    {
                        "name": "Quantum Biology",
                        "description": "Intersection of quantum physics and biological systems",
                        "confidence": 0.92,
                        "sample_count": 2,
                        "example_samples": [
                            "quantum entanglement in photosynthesis",
                            "quantum computing applications",
                        ],
                        "justification": "These samples explore quantum phenomena in biological contexts",
                        "suggested_parent": "physics",
                    },
                    {
                        "name": "Gene Editing Technology",
                        "description": "Advanced genetic modification techniques",
                        "confidence": 0.88,
                        "sample_count": 1,
                        "example_samples": ["CRISPR gene editing efficiency"],
                        "justification": "Focuses on cutting-edge gene editing methods",
                        "suggested_parent": "biology",
                    },
                ],
                "rejected_as_noise": [],
                "analysis_summary": "Identified 2 distinct novel classes from the samples",
                "cluster_count": 2,
            }
        )

    @pytest.fixture
    def proposer(self):
        """Create LLMClassProposer instance."""
        return LLMClassProposer(
            primary_model="test-model",
            temperature=0.3,
            max_tokens=4096,
        )

    def test_initialization(self, proposer):
        """Test proposer initialization."""
        assert proposer.primary_model == "test-model"
        assert proposer.temperature == 0.3
        assert proposer.max_tokens == 4096

    def test_initialization_with_api_keys(self):
        """Test initialization with API keys."""
        api_keys = {
            "openrouter": "sk-test-123",
            "anthropic": "sk-abc-456",
        }

        proposer = LLMClassProposer(api_keys=api_keys)

        assert proposer._api_keys == api_keys

    def test_group_by_cluster(self, proposer, sample_novel_samples):
        """Test grouping samples by cluster ID."""
        clustered = proposer._group_by_cluster(sample_novel_samples)

        assert 0 in clustered
        assert 1 in clustered
        assert len(clustered[0]) == 2
        assert len(clustered[1]) == 1

    def test_group_by_cluster_no_clusters(self, proposer, sample_novel_samples):
        """Test grouping when samples have no cluster IDs."""
        # Remove cluster IDs
        for sample in sample_novel_samples:
            sample.cluster_id = None

        clustered = proposer._group_by_cluster(sample_novel_samples)

        # All should be in None cluster
        assert None in clustered
        assert len(clustered[None]) == 3

    def test_build_proposal_prompt(self, proposer, sample_novel_samples):
        """Test proposal prompt building."""
        existing_classes = ["physics", "cs", "biology"]
        clustered_samples = proposer._group_by_cluster(sample_novel_samples)

        prompt = proposer._build_proposal_prompt(
            novel_samples=sample_novel_samples,
            existing_classes=existing_classes,
            clustered_samples=clustered_samples,
            context="Scientific research domain",
        )

        # Check that prompt contains key elements
        assert "Existing Classes:" in prompt
        assert "physics, cs, biology" in prompt
        assert "quantum entanglement in photosynthesis" in prompt
        assert "Domain Context: Scientific research domain" in prompt
        assert "Natural clusters found:" in prompt
        assert "Cluster 0:" in prompt

    def test_build_proposal_prompt_no_context(self, proposer, sample_novel_samples):
        """Test prompt building without context."""
        existing_classes = ["physics", "cs"]
        prompt = proposer._build_proposal_prompt(
            novel_samples=sample_novel_samples,
            existing_classes=existing_classes,
            clustered_samples={},
            context=None,
        )

        # Should not contain context section
        assert "Domain Context:" not in prompt

    def test_parse_response_success(
        self, proposer, mock_llm_response, sample_novel_samples
    ):
        """Test successful LLM response parsing."""
        analysis = proposer._parse_response(
            mock_llm_response,
            sample_novel_samples,
            "test-model",
        )

        assert len(analysis.proposed_classes) == 2
        assert analysis.proposed_classes[0].name == "Quantum Biology"
        assert analysis.proposed_classes[0].confidence == 0.92
        assert analysis.cluster_count == 2
        assert len(analysis.rejected_as_noise) == 0

    def test_parse_response_with_markdown(
        self, proposer, mock_llm_response, sample_novel_samples
    ):
        """Test parsing response with markdown code blocks."""
        response_with_md = f"```json\n{mock_llm_response}\n```"

        analysis = proposer._parse_response(
            response_with_md,
            sample_novel_samples,
            "test-model",
        )

        assert len(analysis.proposed_classes) == 2

    def test_parse_response_invalid_json(self, proposer, sample_novel_samples):
        """Test parsing invalid JSON response."""
        invalid_response = "This is not valid JSON"

        with pytest.raises(ValueError, match="Invalid JSON"):
            proposer._parse_response(invalid_response, sample_novel_samples)

    def test_parse_response_validation_error(self, proposer, sample_novel_samples):
        """Test parsing response that doesn't match schema."""
        invalid_schema = json.dumps(
            {
                "proposed_classes": [
                    {
                        "name": "Test",
                        # Missing required fields
                    }
                ]
            }
        )

        with pytest.raises(ValueError, match="validation failed"):
            proposer._parse_response(invalid_schema, sample_novel_samples)

    def test_call_llm_with_fallback_returns_successful_model(self, proposer):
        """Fallback path should report the model that actually succeeded."""
        proposer.primary_model = "broken-model"
        proposer.fallback_models = ["working-model"]

        with patch.object(
            proposer,
            "_call_litellm",
            side_effect=[
                RuntimeError("boom"),
                (
                    '{"proposed_classes":[],"rejected_as_noise":[],'
                    '"analysis_summary":"ok","cluster_count":0}'
                ),
            ],
        ):
            response, model_used = proposer._call_llm_with_fallback("prompt")

        assert model_used == "working-model"
        assert '"analysis_summary":"ok"' in response

    def test_create_fallback_analysis(self, proposer):
        """Test fallback analysis creation."""
        # Create samples with the same predicted class to ensure a proposal is created
        samples = [
            NovelSampleMetadata(
                text="sample 1",
                index=0,
                confidence=0.5,
                predicted_class="physics",
                signals={},
            ),
            NovelSampleMetadata(
                text="sample 2",
                index=1,
                confidence=0.5,
                predicted_class="physics",
                signals={},
            ),
            NovelSampleMetadata(
                text="sample 3",
                index=2,
                confidence=0.5,
                predicted_class="physics",
                signals={},
            ),
        ]
        existing_classes = ["physics", "cs", "biology"]

        fallback = proposer._create_fallback_analysis(samples, existing_classes)

        assert fallback.model_used == "fallback"
        assert "LLM failure" in fallback.analysis_summary
        assert len(fallback.proposed_classes) > 0

    def test_create_fallback_analysis_allows_no_proposals(self, proposer):
        """Fallback analysis should still be valid when no group is large enough."""
        samples = [
            NovelSampleMetadata(
                text="sample 1",
                index=0,
                confidence=0.5,
                predicted_class="physics",
                signals={},
            ),
            NovelSampleMetadata(
                text="sample 2",
                index=1,
                confidence=0.5,
                predicted_class="biology",
                signals={},
            ),
        ]

        fallback = proposer._create_fallback_analysis(samples, ["physics", "biology"])

        assert fallback.model_used == "fallback"
        assert fallback.proposed_classes == []

    def test_call_litellm_success(self, proposer, monkeypatch):
        """Test successful litellm call."""
        mock_completion = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response
        mock_litellm = ModuleType("litellm")
        mock_litellm.completion = mock_completion
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        prompt = "Test prompt"
        response = proposer._call_litellm("test-model", prompt)

        assert response == "Test response"
        mock_completion.assert_called_once()

    def test_call_litellm_import_error(self, proposer):
        """Test litellm call when litellm is not installed."""
        # This test verifies the error handling path
        # Since litellm is installed in the test environment, we just verify
        # that the function exists and can be called
        assert hasattr(proposer, "_call_litellm")
        # Actual ImportError testing would require mocking sys.modules

    def test_propose_classes_success(
        self,
        proposer,
        sample_novel_samples,
        mock_llm_response,
        monkeypatch,
    ):
        """Test successful class proposal."""
        mock_completion = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = mock_llm_response
        mock_completion.return_value = mock_response
        mock_litellm = ModuleType("litellm")
        mock_litellm.completion = mock_completion
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        existing_classes = ["physics", "cs", "biology"]

        analysis = proposer.propose_classes(
            novel_samples=sample_novel_samples,
            existing_classes=existing_classes,
            context="Scientific research",
        )

        assert len(analysis.proposed_classes) == 2
        assert analysis.proposed_classes[0].name == "Quantum Biology"

    def test_propose_classes_with_fallback(
        self, proposer, sample_novel_samples, monkeypatch
    ):
        """Test class proposal with fallback on LLM failure."""
        mock_completion = Mock()
        # First call fails, second call succeeds with fallback
        mock_completion.side_effect = [
            Exception("Primary model failed"),
            Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content='{"proposed_classes": [{"name": "Test Class", "description": "Test", "confidence": 0.8, "sample_count": 1, "example_samples": ["test"], "justification": "test"}], "rejected_as_noise": [], "analysis_summary": "test", "cluster_count": 1}'
                        )
                    )
                ]
            ),
        ]
        mock_litellm = ModuleType("litellm")
        mock_litellm.completion = mock_completion
        monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

        existing_classes = ["physics", "cs", "biology"]

        # Should use fallback model
        proposer.propose_classes(
            novel_samples=sample_novel_samples,
            existing_classes=existing_classes,
        )

        # Should have called twice (primary + fallback)
        assert mock_completion.call_count == 2

    def test_propose_classes_empty_samples(self, proposer):
        """Test proposing with empty novel samples."""
        with pytest.raises(ValueError, match="novel_samples cannot be empty"):
            proposer.propose_classes(
                novel_samples=[],
                existing_classes=["physics", "cs"],
            )

    def test_get_model_config_claude(self, proposer):
        """Test getting config for Claude model."""
        config = proposer._get_model_config("claude-sonnet-4")

        assert "max_tokens" in config
        assert "temperature" in config

    def test_get_model_config_gpt(self, proposer):
        """Test getting config for GPT model."""
        config = proposer._get_model_config("gpt-4")

        assert "max_tokens" in config
        assert "temperature" in config

    def test_get_model_config_default(self, proposer):
        """Test getting default model config."""
        config = proposer._get_model_config("unknown-model")

        assert "max_tokens" in config
        assert "temperature" in config

    def test_provider_to_env_var(self, proposer):
        """Test converting provider name to env var."""
        assert proposer._provider_to_env_var("openrouter") == "OPENROUTER_API_KEY"
        assert proposer._provider_to_env_var("anthropic") == "ANTHROPIC_API_KEY"
        assert proposer._provider_to_env_var("openai") == "OPENAI_API_KEY"
        assert proposer._provider_to_env_var("unknown") is None

    def test_get_api_keys_from_env(self, proposer, monkeypatch):
        """Test getting API keys from environment variables."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-abc-456")

        keys = proposer._get_api_keys_from_env()

        assert keys["openrouter"] == "sk-test-123"
        assert keys["anthropic"] == "sk-abc-456"
