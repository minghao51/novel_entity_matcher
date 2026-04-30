"""
LLM-based class proposal system for novel class discovery.

Uses litellm with structured output to generate meaningful class names
and descriptions for clusters of novel samples.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, Field, ValidationError

try:
    from tenacity import (
        RetryCallState,
        before_sleep_log,
        retry,
        retry_if_exception_type,
        wait_exponential_jitter,
    )
except ImportError:  # pragma: no cover - optional dependency
    RetryCallState = Any  # type: ignore[assignment,misc]

    def retry(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
        def _decorator(func: Any) -> Any:
            return func

        return _decorator

    def wait_exponential_jitter(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None

    def retry_if_exception_type(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None

    def before_sleep_log(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[misc]
        return None


try:
    from aiobreaker import CircuitBreaker, CircuitBreakerError
except ImportError:  # pragma: no cover - optional dependency

    class CircuitBreaker:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        def __call__(self, func: Any) -> Any:
            return func

    class CircuitBreakerError(Exception):  # type: ignore[no-redef]
        pass


from ...exceptions import LLMError, _redact_api_keys
from ...utils.logging_config import get_logger
from ..schemas import (
    ClassProposal,
    DiscoveryCluster,
    NovelClassAnalysis,
    NovelSampleMetadata,
)

logger = get_logger(__name__)


def _stop_after_configured_attempts(retry_state: RetryCallState) -> bool:
    """Use configured max retry attempts from ``LLMConfig``."""
    instance = retry_state.args[0] if retry_state.args else None
    max_retries = getattr(getattr(instance, "config", None), "max_retries", 5)
    return retry_state.attempt_number >= int(max_retries)


# Retryable LLM API exceptions
try:
    from litellm import (
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
        RateLimitError,
        APITimeoutError,
        InternalServerError,
        TimeoutError,
    )
except ImportError:
    RETRYABLE_EXCEPTIONS = (TimeoutError,)


class LLMProposalSchema(BaseModel):
    """Schema enforcing the exact JSON structure expected from LLM proposals."""

    proposed_classes: list[dict] = Field(
        default_factory=list,
        description="List of proposed classes, each with name, description, confidence, sample_count, example_samples, justification",
    )
    rejected_as_noise: list[str] = Field(
        default_factory=list,
        description="List of sample texts or cluster IDs to reject as noise",
    )
    analysis_summary: str = Field(
        ...,
        description="Brief summary of the analysis",
    )
    cluster_count: int = Field(
        ...,
        description="Number of distinct clusters found",
    )

    @classmethod
    def get_schema_json(cls) -> str:
        """Return the JSON Schema representation."""
        return json.dumps(cls.model_json_schema(), indent=2)


class LLMProposalWithSchemaSchema(BaseModel):
    """Schema for proposals that include attribute/field discovery."""

    proposed_classes: list[dict] = Field(
        default_factory=list,
        description="List of proposed classes with discovered attributes",
    )
    rejected_as_noise: list[str] = Field(
        default_factory=list,
        description="List of sample texts or cluster IDs to reject as noise",
    )
    analysis_summary: str = Field(
        ...,
        description="Brief summary of the analysis",
    )
    cluster_count: int = Field(
        ...,
        description="Number of distinct clusters found",
    )

    @classmethod
    def get_schema_json(cls) -> str:
        """Return the JSON Schema representation."""
        return json.dumps(cls.model_json_schema(), indent=2)


# Default LLM providers with fallback chain
DEFAULT_PROVIDERS = [
    "openrouter/anthropic/claude-sonnet-4",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "openrouter/openai/gpt-4o",
]

# Model-specific configuration
MODEL_CONFIGS = {
    "claude": {
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "gpt-4": {
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "default": {
        "max_tokens": 4096,
        "temperature": 0.3,
    },
}


class LLMClassProposer:
    """
    Propose new class names and descriptions using LLMs.

    Uses litellm for multi-provider support with automatic fallback.
    """

    def __init__(
        self,
        primary_model: str | None = None,
        provider: str | None = None,
        fallback_models: list[str] | None = None,
        api_keys: dict[str, str] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_clusters_per_summary: int = 20,
    ):
        """
        Initialize LLM class proposer.

        Args:
            primary_model: Primary model to use (e.g., 'openrouter/anthropic/claude-sonnet-4')
            provider: Preferred provider when auto-selecting a default model
            fallback_models: Fallback models if primary fails
            api_keys: API keys for providers (e.g., {'openrouter': 'sk-...'})
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_clusters_per_summary: Maximum clusters to include per LLM summary call (for hierarchical mode)
        """
        self.primary_model = primary_model or os.getenv(
            "LLM_CLASS_PROPOSER_MODEL",
            self._default_model_for_provider(provider),
        )
        default_fallbacks = [
            model for model in DEFAULT_PROVIDERS if model != self.primary_model
        ]
        self.fallback_models = fallback_models or default_fallbacks
        self._api_keys = api_keys or self._get_api_keys_from_env()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_clusters_per_summary = max_clusters_per_summary

        # Load LLM configuration with environment variable support when available.
        try:
            from .config import get_llm_config

            self.config = get_llm_config()
        except ImportError:  # pragma: no cover - optional dependency
            self.config = SimpleNamespace(  # type: ignore[assignment]
                timeout=30,
                max_retries=5,
                circuit_fail_max=3,
                circuit_reset_seconds=60,
            )

        # Create circuit breaker for LLM API calls
        self.llm_circuit_breaker = CircuitBreaker(
            fail_max=self.config.circuit_fail_max,
            timeout_duration=timedelta(seconds=self.config.circuit_reset_seconds),
        )

        logger.info(
            f"LLMClassProposer initialized: timeout={self.config.timeout}s, "
            f"max_retries={self.config.max_retries}, "
            f"circuit_fail_max={self.config.circuit_fail_max}, "
            f"circuit_reset={self.config.circuit_reset_seconds}s"
        )

    def _default_model_for_provider(self, provider: str | None) -> str:
        """Select a default model, optionally honoring a preferred provider."""
        if not provider:
            return DEFAULT_PROVIDERS[0]

        provider_prefixes = {
            "openrouter": "openrouter/",
            "anthropic": "anthropic/",
            "openai": "openai/",
        }
        prefix = provider_prefixes.get(provider.lower())
        if not prefix:
            return DEFAULT_PROVIDERS[0]

        for model in DEFAULT_PROVIDERS:
            if model.startswith(prefix):
                return model
        return DEFAULT_PROVIDERS[0]

    def _get_api_keys_from_env(self) -> dict[str, str]:
        """Get API keys from environment variables."""
        keys = {}
        env_mappings = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }

        for provider, env_var in env_mappings.items():
            key = os.getenv(env_var)
            if key:
                keys[provider] = key

        return keys

    def _provider_to_env_var(self, provider: str) -> str | None:
        """Convert provider name to environment variable name."""
        mappings = {
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        return mappings.get(provider)

    def _api_key_for_model(self, model: str) -> str | None:
        """Resolve provider-specific API key for a model identifier."""
        provider = (model.split("/", 1)[0] if model else "").lower()
        if provider in self._api_keys:
            return self._api_keys[provider]
        return None

    def propose_classes(
        self,
        novel_samples: list[NovelSampleMetadata],
        existing_classes: list[str],
        context: str | None = None,
    ) -> NovelClassAnalysis:
        """
        Propose new classes based on novel samples.

        Args:
            novel_samples: List of detected novel samples
            existing_classes: List of existing class names
            context: Optional domain context

        Returns:
            NovelClassAnalysis with proposed classes
        """
        if not novel_samples:
            raise ValueError("novel_samples cannot be empty")

        logger.info(
            f"Proposing classes for {len(novel_samples)} novel samples "
            f"using model: {self.primary_model}"
        )

        clustered_samples = self._group_by_cluster(novel_samples)
        prompt = self._build_proposal_prompt(
            novel_samples,
            existing_classes,
            clustered_samples,
            context,
        )
        clusters = self._clusters_from_samples(clustered_samples)
        return self._run_structured_cluster_proposal(
            prompt=prompt,
            discovery_clusters=clusters,
            novel_samples=novel_samples,
        )

    def propose_from_clusters(
        self,
        discovery_clusters: list[DiscoveryCluster],
        existing_classes: list[str],
        context: str | None = None,
        max_retries: int = 2,
        hierarchical: bool = True,
    ) -> NovelClassAnalysis:
        """Generate proposals from cluster-level evidence.

        Args:
            discovery_clusters: List of discovery clusters.
            existing_classes: List of existing class names.
            context: Optional domain context.
            max_retries: Maximum retry attempts.
            hierarchical: If True, use hierarchical summarization for large cluster sets.
        """
        if not discovery_clusters:
            raise ValueError("discovery_clusters cannot be empty")

        if hierarchical and len(discovery_clusters) > self.max_clusters_per_summary:
            return self._propose_hierarchical(
                discovery_clusters, existing_classes, context, max_retries
            )

        prompt = self._build_cluster_prompt(
            discovery_clusters=discovery_clusters,
            existing_classes=existing_classes,
            context=context,
        )
        return self._run_structured_cluster_proposal(
            prompt=prompt,
            discovery_clusters=discovery_clusters,
            max_retries=max_retries,
        )

    def propose_from_clusters_with_schema(
        self,
        discovery_clusters: list[DiscoveryCluster],
        existing_classes: list[str],
        context: str | None = None,
        max_retries: int = 2,
        hierarchical: bool = True,
        max_attributes: int = 10,
    ) -> NovelClassAnalysis:
        """Generate proposals with attribute/field discovery from cluster evidence.

        Like ``propose_from_clusters`` but the LLM prompt requests discovery of
        common attributes and data structures for each proposed class.

        Args:
            discovery_clusters: List of discovery clusters.
            existing_classes: List of existing class names.
            context: Optional domain context.
            max_retries: Maximum retry attempts.
            hierarchical: If True, use hierarchical summarization for large cluster sets.
        """
        if not discovery_clusters:
            raise ValueError("discovery_clusters cannot be empty")

        if hierarchical and len(discovery_clusters) > self.max_clusters_per_summary:
            return self._propose_hierarchical(
                discovery_clusters,
                existing_classes,
                context,
                max_retries,
                include_schema_discovery=True,
                max_attributes=max_attributes,
            )

        prompt = self._build_cluster_prompt_with_schema(
            discovery_clusters=discovery_clusters,
            existing_classes=existing_classes,
            context=context,
            max_attributes=max_attributes,
        )
        analysis = self._run_structured_cluster_proposal(
            prompt=prompt,
            discovery_clusters=discovery_clusters,
            max_retries=max_retries,
            retry_schema_json=LLMProposalWithSchemaSchema.get_schema_json(),
        )

        if analysis.proposed_classes:
            analysis = self._enrich_proposals_with_schema(analysis)

        return analysis

    def _build_cluster_prompt_with_schema(
        self,
        discovery_clusters: list[DiscoveryCluster],
        existing_classes: list[str],
        context: str | None,
        max_attributes: int,
    ) -> str:
        """Build prompt requesting attribute discovery alongside class proposals."""
        cluster_lines = []
        for cluster in discovery_clusters:
            keywords = ", ".join(cluster.keywords or [])
            examples = "; ".join(
                cluster.evidence.representative_examples
                if cluster.evidence
                else cluster.example_texts
            )
            cluster_lines.append(
                f"- Cluster {cluster.cluster_id}: sample_count={cluster.sample_count}; "
                f"keywords=[{keywords}]; examples=[{examples}]"
            )

        context_section = f"\nDomain Context: {context}" if context else ""
        return f"""You are reviewing clusters of likely novel concepts and discovering their data structure.

Existing Classes: {", ".join(existing_classes)}{context_section}

Discovery Clusters:
{chr(10).join(cluster_lines)}

For each proposed class, also discover common attributes/fields that would describe
entities of this type. Think about what properties these samples share.

Return valid JSON with this schema:
{{
  "proposed_classes": [
    {{
      "name": "class name",
      "description": "what the class represents",
      "confidence": 0.0,
      "sample_count": 0,
      "example_samples": ["example"],
      "justification": "why this cluster should become a class",
      "suggested_parent": null,
      "source_cluster_ids": [0],
      "provenance": {{"keywords": ["keyword"]}},
      "discovered_attributes": [
        {{
          "name": "attribute_name",
          "description": "what this attribute represents",
          "value_type": "string",
          "example_values": ["value1", "value2"]
        }}
      ]
    }}
  ],
  "rejected_as_noise": ["cluster ids or sample text"],
  "analysis_summary": "brief summary",
  "cluster_count": {len(discovery_clusters)}
}}

For discovered_attributes:
- value_type must be one of: "string", "number", "boolean", "enum", "date"
- Use "enum" when the attribute takes a fixed set of values (include enum_values)
- Include 3-{max_attributes} attributes per class that best describe the entities
- Only include attributes that are supported by the sample evidence

Prefer one proposal per coherent cluster. Use source_cluster_ids for traceability."""

    def _enrich_proposals_with_schema(
        self, analysis: NovelClassAnalysis
    ) -> NovelClassAnalysis:
        """Parse and attach discovered attributes from proposal data."""
        from ..schemas.models import DiscoveredAttribute

        enriched_classes = []
        for proposal in analysis.proposed_classes:
            raw_attrs = proposal.discovered_attributes or []
            if not raw_attrs:
                provenance = proposal.provenance or {}
                raw_attrs = provenance.get("discovered_attributes", [])
            if raw_attrs:
                parsed = []
                for attr in raw_attrs:
                    try:
                        parsed.append(
                            attr
                            if isinstance(attr, DiscoveredAttribute)
                            else DiscoveredAttribute(**attr)
                        )
                    except Exception:
                        continue
                if parsed:
                    proposal.discovered_attributes = parsed
                    proposal.attribute_schema = {
                        attr.name: {
                            "type": attr.value_type,
                            "description": attr.description,
                        }
                        for attr in parsed
                    }
                    proposal.provenance.setdefault(
                        "discovered_attributes",
                        [attr.model_dump() for attr in parsed],
                    )
            enriched_classes.append(proposal)
        analysis.proposed_classes = enriched_classes
        return analysis

    def _propose_hierarchical(
        self,
        discovery_clusters: list[DiscoveryCluster],
        existing_classes: list[str],
        context: str | None,
        max_retries: int,
        include_schema_discovery: bool = False,
        max_attributes: int = 10,
    ) -> NovelClassAnalysis:
        """Generate proposals using hierarchical summarization for large cluster sets."""
        summaries = self._hierarchical_summarize_clusters(
            discovery_clusters, existing_classes, context
        )
        prompt = self._build_hierarchical_prompt(
            summaries,
            existing_classes,
            context,
            include_schema_discovery=include_schema_discovery,
            max_attributes=max_attributes,
        )

        top_clusters = discovery_clusters[: self.max_clusters_per_summary]
        analysis = self._run_structured_cluster_proposal(
            prompt=prompt,
            discovery_clusters=top_clusters,
            max_retries=max_retries,
            retry_schema_json=(
                LLMProposalWithSchemaSchema.get_schema_json()
                if include_schema_discovery
                else None
            ),
        )
        if include_schema_discovery and analysis.proposed_classes:
            analysis = self._enrich_proposals_with_schema(analysis)
        return analysis

    def _build_hierarchical_prompt(
        self,
        summaries: list[dict[str, Any]],
        existing_classes: list[str],
        context: str | None,
        include_schema_discovery: bool = False,
        max_attributes: int = 10,
    ) -> str:
        """Build prompt from hierarchical cluster summaries."""
        summary_lines = []
        for i, summary in enumerate(summaries):
            cluster_ids = ", ".join(str(c) for c in summary["cluster_ids"])
            summary_lines.append(
                f"- Group {i + 1} (clusters {cluster_ids}, {summary['total_samples']} samples): "
                f"keywords=[{summary['keywords']}]; examples=[{summary['representative_examples']}]"
            )

        context_section = f"\nDomain Context: {context}" if context else ""
        proposal_block = """{
      "name": "class name",
      "description": "what the class represents",
      "confidence": 0.0,
      "sample_count": 0,
      "example_samples": ["example"],
      "justification": "why this cluster group suggests a class",
      "suggested_parent": null,
      "source_cluster_ids": [0],
      "provenance": {"keywords": ["keyword"]}"""
        if include_schema_discovery:
            proposal_block += """,
    "discovered_attributes": [
      {
        "name": "attribute_name",
        "description": "what this attribute represents",
        "value_type": "string",
        "example_values": ["value1", "value2"]
      }
    ]"""
        proposal_block += "\n    }"

        schema_guidance = ""
        if include_schema_discovery:
            schema_guidance = f"""

For discovered_attributes:
- value_type must be one of: "string", "number", "boolean", "enum", "date"
- Use "enum" when the attribute takes a fixed set of values (include enum_values)
- Include 3-{max_attributes} attributes per class that best describe the entities
- Only include attributes that are supported by the summarized evidence"""

        return f"""You are reviewing summarized groups of likely novel concepts.

Existing Classes: {", ".join(existing_classes)}{context_section}

Cluster Groups:
{chr(10).join(summary_lines)}

Return valid JSON with this schema:
{{
  "proposed_classes": [
{proposal_block}
  ],
  "rejected_as_noise": ["group ids or sample text"],
  "analysis_summary": "brief summary",
  "cluster_count": {sum(len(s["cluster_ids"]) for s in summaries)}
}}

Prefer one proposal per coherent group. Use source_cluster_ids for traceability.{schema_guidance}"""

    def _group_by_cluster(
        self, samples: list[NovelSampleMetadata]
    ) -> dict[int | None, list[NovelSampleMetadata]]:
        """Group samples by cluster ID."""
        clusters: dict[int | None, list[NovelSampleMetadata]] = defaultdict(list)
        for sample in samples:
            clusters[sample.cluster_id].append(sample)
        return dict(clusters)

    def _hierarchical_summarize_clusters(
        self,
        discovery_clusters: list[DiscoveryCluster],
        existing_classes: list[str],
        context: str | None,
    ) -> list[dict[str, Any]]:
        """Create hierarchical summaries of clusters to manage token budget."""
        if len(discovery_clusters) <= self.max_clusters_per_summary:
            return [self._summarize_cluster_group(discovery_clusters)]

        summaries = []
        for i in range(0, len(discovery_clusters), self.max_clusters_per_summary):
            chunk = discovery_clusters[i : i + self.max_clusters_per_summary]
            summaries.append(self._summarize_cluster_group(chunk))
        return summaries

    def _summarize_cluster_group(
        self, clusters: list[DiscoveryCluster]
    ) -> dict[str, Any]:
        """Create a summary dict for a group of clusters."""
        return {
            "cluster_ids": [c.cluster_id for c in clusters],
            "total_samples": sum(c.sample_count for c in clusters),
            "keywords": ", ".join(", ".join(c.keywords or []) for c in clusters[:3]),
            "representative_examples": "; ".join(
                ex
                for c in clusters[:2]
                for ex in (
                    c.evidence.representative_examples[:2]
                    if c.evidence
                    else c.example_texts[:2]
                )
            ),
        }

    def _build_proposal_prompt(
        self,
        novel_samples: list[NovelSampleMetadata],
        existing_classes: list[str],
        clustered_samples: dict[int | None, list[NovelSampleMetadata]],
        context: str | None,
        max_input_chars: int = 100_000,
    ) -> str:
        """Build the proposal prompt for the LLM."""
        # Format samples for the prompt
        sample_texts = [
            f"- {sample.text}"
            for sample in novel_samples[:20]  # Limit to 20
        ]

        if len(novel_samples) > 20:
            sample_texts.append(f"... and {len(novel_samples) - 20} more samples")

        samples_section = "\n".join(sample_texts)

        # Format existing classes
        existing_classes_section = ", ".join(existing_classes)

        # Build context section
        context_section = ""
        if context:
            context_section = f"\n\nDomain Context: {context}"

        # Build cluster section if applicable
        cluster_section = ""
        if clustered_samples and len(clustered_samples) > 1:
            cluster_info = []
            for cluster_id, samples in clustered_samples.items():
                cluster_name = (
                    f"Cluster {cluster_id}" if cluster_id is not None else "Unclustered"
                )
                sample_list = ", ".join([s.text[:50] for s in samples[:3]])
                cluster_info.append(f"- {cluster_name}: {sample_list}")
            cluster_section = "\n\nNatural clusters found:\n" + "\n".join(cluster_info)

        prompt = f"""You are analyzing text samples that don't fit well into existing categories.

Existing Classes: {existing_classes_section}{context_section}{cluster_section}

Novel Samples (detected as not fitting existing classes):
{samples_section}

Your task is to:
1. Analyze these samples to identify meaningful new categories
2. Propose concise, descriptive class names
3. Provide justifications for each proposal
4. Identify samples that should be rejected as noise

IMPORTANT RESPONSE FORMAT:
You must respond with a valid JSON object matching this schema:
{{
  "proposed_classes": [
    {{
      "name": "class name (2-4 words)",
      "description": "clear description of what this class represents",
      "confidence": 0.0-1.0,
      "sample_count": number of samples fitting this class,
      "example_samples": ["sample1", "sample2", "sample3"],
      "justification": "why this class makes sense",
      "suggested_parent": null or "parent class name if hierarchical"
    }}
  ],
  "rejected_as_noise": ["sample text to reject"],
  "analysis_summary": "brief summary of your analysis",
  "cluster_count": number of distinct clusters found
}}

Guidelines:
- Class names should be concise (2-4 words), descriptive, and follow naming conventions of existing classes
- Confidence should reflect how clearly the samples form a coherent category
- Only propose classes with at least 3 supporting samples
- Reject samples that appear to be noise, errors, or too diverse
- Return "proposed_classes": [] if no coherent new class should be created
- Consider hierarchical relationships if relevant to the domain

Provide your analysis as a JSON object:"""

        if len(prompt) > max_input_chars:
            prompt = (
                prompt[:max_input_chars]
                + "\n\n[TRUNCATED: prompt exceeded maximum length]"
            )

        return prompt

    def _build_cluster_prompt(
        self,
        discovery_clusters: list[DiscoveryCluster],
        existing_classes: list[str],
        context: str | None,
    ) -> str:
        cluster_lines = []
        for cluster in discovery_clusters:
            keywords = ", ".join(cluster.keywords or [])
            examples = "; ".join(
                cluster.evidence.representative_examples
                if cluster.evidence
                else cluster.example_texts
            )
            cluster_lines.append(
                f"- Cluster {cluster.cluster_id}: sample_count={cluster.sample_count}; "
                f"keywords=[{keywords}]; examples=[{examples}]"
            )

        context_section = f"\nDomain Context: {context}" if context else ""
        return f"""You are reviewing clusters of likely novel concepts.

Existing Classes: {", ".join(existing_classes)}{context_section}

Discovery Clusters:
{chr(10).join(cluster_lines)}

Return valid JSON with this schema:
{{
  "proposed_classes": [
    {{
      "name": "class name",
      "description": "what the class represents",
      "confidence": 0.0,
      "sample_count": 0,
      "example_samples": ["example"],
      "justification": "why this cluster should become a class",
      "suggested_parent": null,
      "source_cluster_ids": [0],
      "provenance": {{"keywords": ["keyword"]}}
    }}
  ],
  "rejected_as_noise": ["cluster ids or sample text"],
  "analysis_summary": "brief summary",
  "cluster_count": {len(discovery_clusters)}
}}

Prefer one proposal per coherent cluster. Use source_cluster_ids for traceability."""

    def _run_structured_cluster_proposal(
        self,
        *,
        prompt: str,
        discovery_clusters: list[DiscoveryCluster],
        novel_samples: list[NovelSampleMetadata] | None = None,
        max_retries: int = 2,
        retry_schema_json: str | None = None,
    ) -> NovelClassAnalysis:
        attempts = 0
        retry_prompt = prompt
        last_error: Exception | None = None

        while attempts <= max_retries:
            try:
                response, model_used = self._call_llm_with_fallback(retry_prompt)
                analysis = self._parse_response(
                    response,
                    novel_samples or [],
                    model_used,
                )
                if not analysis.cluster_count:
                    analysis.cluster_count = len(discovery_clusters)
                analysis.proposal_metadata.setdefault("attempts", attempts + 1)
                return analysis
            except ValidationError as exc:
                last_error = exc
                attempts += 1
                if attempts > max_retries:
                    break
                field_errors = self._format_validation_errors(exc)
                retry_prompt = self._build_retry_prompt(
                    prompt,
                    field_errors,
                    exc,
                    retry_schema_json=retry_schema_json,
                )
            except (ValueError, TypeError, ConnectionError, RuntimeError) as exc:
                last_error = exc
                attempts += 1
                if attempts > max_retries:
                    break
                retry_prompt = (
                    f"{prompt}\n\nThe previous response was invalid: {exc}. "
                    "Return only valid JSON matching the requested schema."
                )

        logger.error(
            f"Failed to generate LLM proposals: {_redact_api_keys(str(last_error))}"
        )
        if novel_samples is not None:
            return self._create_fallback_analysis(novel_samples, [])
        return NovelClassAnalysis(
            proposed_classes=[],
            rejected_as_noise=[],
            analysis_summary="Fallback analysis due to repeated validation failures.",
            cluster_count=len(discovery_clusters),
            model_used="fallback",
            validation_errors=[str(last_error)] if last_error else [],
            proposal_metadata={"attempts": attempts},
        )

    def _format_validation_errors(self, exc: ValidationError) -> list[dict[str, Any]]:
        """Extract structured field-level errors from a Pydantic ValidationError."""
        errors = []
        for error in exc.errors():
            loc = error.get("loc", [])
            field_path = ".".join(str(part) for part in loc) if loc else "root"
            errors.append(
                {
                    "field": field_path,
                    "type": error.get("type", "unknown"),
                    "message": error.get("msg", ""),
                    "input": str(error.get("input", ""))[:100],
                }
            )
        return errors

    def _build_retry_prompt(
        self,
        original_prompt: str,
        field_errors: list[dict[str, Any]],
        exc: ValidationError,
        retry_schema_json: str | None = None,
    ) -> str:
        """Build a structured retry prompt with schema and field-level errors."""
        error_lines = []
        for err in field_errors:
            error_lines.append(
                f"  - Field '{err['field']}': {err['message']} (got: {err['input']})"
            )

        return f"""{original_prompt}

--- VALIDATION ERRORS ---
Your previous response failed validation with the following errors:
{chr(10).join(error_lines)}

--- REQUIRED JSON SCHEMA ---
{retry_schema_json or LLMProposalSchema.get_schema_json()}

Please fix the errors above and return ONLY valid JSON matching the schema."""

    def _clusters_from_samples(
        self,
        clustered_samples: dict[int | None, list[NovelSampleMetadata]],
    ) -> list[DiscoveryCluster]:
        clusters: list[DiscoveryCluster] = []
        for cluster_id, samples in clustered_samples.items():
            resolved_cluster_id = (
                int(cluster_id) if cluster_id is not None else len(clusters)
            )
            clusters.append(
                DiscoveryCluster(
                    cluster_id=resolved_cluster_id,
                    sample_indices=[sample.index for sample in samples],
                    sample_count=len(samples),
                    example_texts=[sample.text for sample in samples[:4]],
                    keywords=[],
                )
            )
        return clusters

    def _call_llm_with_fallback(self, prompt: str) -> tuple[str, str]:
        """Call LLM with automatic fallback on failure."""
        try:
            from litellm import (
                AuthenticationError,
                RateLimitError,
                ServiceUnavailableError,
            )
            from litellm.exceptions import (
                AuthenticationError as LiteLLMAuthError,
            )
            from litellm.exceptions import (
                RateLimitError as LiteLLMRateLimitError,
            )
            from litellm.exceptions import (
                ServiceUnavailableError as LiteLLMServiceUnavailableError,
            )
        except ImportError:
            AuthenticationError = type("AuthenticationError", (Exception,), {})  # type: ignore[misc,assignment]
            RateLimitError = type("RateLimitError", (Exception,), {})  # type: ignore[misc,assignment]
            ServiceUnavailableError = type("ServiceUnavailableError", (Exception,), {})  # type: ignore[misc,assignment]
            LiteLLMAuthError = AuthenticationError  # type: ignore[misc]
            LiteLLMRateLimitError = RateLimitError  # type: ignore[misc]
            LiteLLMServiceUnavailableError = ServiceUnavailableError  # type: ignore[misc]

        LLM_RETRYABLE_ERRORS = (
            RateLimitError,
            ServiceUnavailableError,
            LiteLLMRateLimitError,
            LiteLLMServiceUnavailableError,
            TimeoutError,
            ConnectionError,
        )
        LLM_AUTH_ERRORS = (
            AuthenticationError,
            LiteLLMAuthError,
        )

        models_to_try = [m for m in ([self.primary_model, *self.fallback_models]) if m]

        last_error: Exception | None = None
        for model in models_to_try:
            try:
                logger.info(f"Trying model: {model}")
                response = self._call_litellm(model, prompt)
                return response, model
            except LLM_AUTH_ERRORS as e:
                logger.error(
                    f"Authentication failed for model {model}: {_redact_api_keys(str(e))}"
                )
                raise LLMError(
                    f"LLM authentication failed for model {model}. Check your API key.",
                    last_error=e,
                    attempted_models=models_to_try,
                ) from e
            except LLM_RETRYABLE_ERRORS as e:
                logger.warning(
                    f"Retryable error for model {model}: {_redact_api_keys(str(e))}"
                )
                last_error = e
                continue
            except (ImportError, ValueError, TypeError, RuntimeError) as e:
                logger.warning(
                    f"Non-retryable error for model {model}: {_redact_api_keys(str(e))}"
                )
                last_error = e
                continue
            except Exception as e:  # pragma: no cover - defensive fallback wrapper
                logger.warning(
                    f"Unexpected error for model {model}: {_redact_api_keys(str(e))}"
                )
                last_error = e
                continue

        # All models failed
        error_msg = (
            f"All LLM providers failed. Last error: {_redact_api_keys(str(last_error))}"
        )
        logger.error(error_msg)
        raise LLMError(
            error_msg,
            last_error=last_error,
            attempted_models=models_to_try,
        ) from last_error

    @retry(
        stop=_stop_after_configured_attempts,
        wait=wait_exponential_jitter(initial=2, max=60, jitter=2),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_litellm(self, model: str, prompt: str) -> str:
        """Call litellm completion API with retry, timeout, and circuit breaker."""
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError(
                "litellm is required for LLM class proposal. "
                "Install with: pip install litellm"
            ) from exc

        model_config = self._get_model_config(model)
        temperature = (
            self.temperature
            if self.temperature is not None
            else model_config["temperature"]
        )
        max_tokens = (
            self.max_tokens
            if self.max_tokens is not None
            else model_config["max_tokens"]
        )

        # Wrap LLM call with circuit breaker
        @self.llm_circuit_breaker
        def call_with_circuit_breaker():
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at categorizing text samples and proposing meaningful class names.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": self.config.timeout,  # CRITICAL: Use configured timeout
            }
            api_key = self._api_key_for_model(model)
            if api_key:
                kwargs["api_key"] = api_key

            return completion(
                **kwargs,
            )

        try:
            response = call_with_circuit_breaker()
        except CircuitBreakerError as exc:
            raise RuntimeError("LLM circuit breaker is open") from exc

        return response.choices[0].message.content

    def _get_model_config(self, model: str) -> dict[str, Any]:
        """Get configuration for a specific model."""
        # Determine model type from model name
        if "claude" in model.lower():
            return MODEL_CONFIGS["claude"]
        elif "gpt-4" in model.lower():
            return MODEL_CONFIGS["gpt-4"]
        else:
            return MODEL_CONFIGS["default"]

    def _parse_response(
        self,
        response: str,
        novel_samples: list[NovelSampleMetadata],
        model_used: str | None = None,
    ) -> NovelClassAnalysis:
        """Parse structured LLM response into NovelClassAnalysis."""
        # Extract JSON from response (handle markdown code blocks)
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code block markers
            lines = response.split("\n")
            if lines[0].startswith("```json"):
                response = "\n".join(lines[1:-1])
            elif lines[0].startswith("```"):
                response = "\n".join(lines[1:-1])
            else:
                # Try to find JSON content
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    response = response[start:end]

        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}") from e

        # Validate with Pydantic
        try:
            final_model = model_used or self.primary_model or "unknown"
            analysis = NovelClassAnalysis(
                proposed_classes=[
                    ClassProposal(**proposal)
                    for proposal in data.get("proposed_classes", [])
                ],
                rejected_as_noise=data.get("rejected_as_noise", []),
                analysis_summary=data.get("analysis_summary", ""),
                cluster_count=data.get("cluster_count", 0),
                model_used=final_model,
                validation_errors=[],
                proposal_metadata=data.get("proposal_metadata", {}),
            )
            for proposal in analysis.proposed_classes:
                proposal.provenance.setdefault("model_used", final_model)
            return analysis
        except ValidationError as e:
            raise ValueError(f"Response validation failed: {e}") from e

    def _create_fallback_analysis(
        self, novel_samples: list[NovelSampleMetadata], existing_classes: list[str]
    ) -> NovelClassAnalysis:
        """Create fallback analysis when LLM fails."""
        logger.warning("Creating fallback analysis due to LLM failure")

        # Simple fallback: group by predicted class
        predicted_groups: dict[str, list[NovelSampleMetadata]] = {}
        for sample in novel_samples:
            pred_class = sample.predicted_class
            if pred_class not in predicted_groups:
                predicted_groups[pred_class] = []
            predicted_groups[pred_class].append(sample)

        proposals = []
        for pred_class, samples in predicted_groups.items():
            if len(samples) >= 3:
                proposals.append(
                    ClassProposal(
                        name=f"Novel {pred_class}",
                        description=f"Samples related to {pred_class} that don't fit existing categories",
                        confidence=0.5,
                        sample_count=len(samples),
                        example_samples=[s.text for s in samples[:3]],
                        justification=f"Grouped by predicted class '{pred_class}'",
                        suggested_parent=pred_class,
                    )
                )

        return NovelClassAnalysis(
            proposed_classes=proposals,
            rejected_as_noise=[],
            analysis_summary="Fallback analysis due to LLM failure. Samples grouped by predicted class.",
            cluster_count=len(proposals),
            model_used="fallback",
            validation_errors=[],
            proposal_metadata={},
        )
