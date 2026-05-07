"""
DSPy-based proposal module for optimizing LLM class proposals.

Replaces handcrafted prompts with DSPy Signatures and Modules.
Supports GEPA optimization using ProposalReviewManager training signal.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import dspy
except ImportError:  # pragma: no cover - optional dependency
    dspy = None


if dspy is not None:

    class ClusterProposalSignature(dspy.Signature):
        """Generate novel class proposals from discovery clusters.

        Analyze the provided clusters of novel samples and propose meaningful
        class names and descriptions. Avoid duplicating existing classes.
        Each cluster may represent a distinct novel concept worth adding as a new class.
        """

        clusters_json: str = dspy.InputField(
            desc="JSON array of clusters, each with cluster_id, sample_count, keywords, and example_texts"
        )
        existing_classes: str = dspy.InputField(
            desc="Comma-separated existing class names to avoid duplicating"
        )
        domain_context: str = dspy.InputField(
            desc="Optional domain context describing the data domain", default=""
        )
        analysis_json: str = dspy.OutputField(
            desc='JSON matching schema: {"proposed_classes":[{"name","description","confidence","sample_count","example_samples","justification","source_cluster_ids"}],"rejected_as_noise":[],"analysis_summary":"","cluster_count":0}'
        )

    class DSPyProposalModule(dspy.Module):
        """DSPy module for generating class proposals from discovery clusters.

        Uses ChainOfThought to produce structured JSON proposals.
        Can be optimized via GEPA or other DSPy teleprompters.
        """

        def __init__(self, lm: dspy.LM | None = None):
            super().__init__()
            if lm is not None:
                self.lm = lm
            self.propose = dspy.ChainOfThought(ClusterProposalSignature)

        def forward(
            self,
            clusters_json: str,
            existing_classes: str,
            domain_context: str = "",
        ) -> dspy.Prediction:
            return self.propose(
                clusters_json=clusters_json,
                existing_classes=existing_classes,
                domain_context=domain_context,
            )

        def predict(
            self,
            clusters_json: str,
            existing_classes: str,
            domain_context: str = "",
        ) -> str:
            """Run inference and return the raw analysis JSON string."""
            result = self.forward(
                clusters_json=clusters_json,
                existing_classes=existing_classes,
                domain_context=domain_context,
            )
            return result.analysis_json

    def review_records_to_examples(
        approved_records: list[dict[str, Any]],
        rejected_records: list[dict[str, Any]] | None = None,
    ) -> list[dspy.Example]:
        """Convert ProposalReviewManager records to DSPy training examples.

        Args:
            approved_records: List of approved proposal review records (positive examples).
            rejected_records: List of rejected proposal review records (negative examples).

        Returns:
            List of dspy.Example objects ready for DSPy teleprompt training.
        """
        examples: list[dspy.Example] = []
        for record in approved_records:
            cluster_data = _record_to_cluster_json(record)
            example = dspy.Example(
                clusters_json=cluster_data,
                existing_classes=_extract_existing_classes(record),
                domain_context=_extract_domain_context(record),
                analysis_json=_record_to_analysis_json(record),
            ).with_inputs("clusters_json", "existing_classes", "domain_context")
            examples.append(example)

        for record in rejected_records or []:
            cluster_data = _record_to_cluster_json(record)
            empty_analysis = json.dumps(
                {
                    "proposed_classes": [],
                    "rejected_as_noise": ["all"],
                    "analysis_summary": "No coherent classes found in this cluster set.",
                    "cluster_count": 0,
                }
            )
            example = dspy.Example(
                clusters_json=cluster_data,
                existing_classes=_extract_existing_classes(record),
                domain_context=_extract_domain_context(record),
                analysis_json=empty_analysis,
            ).with_inputs("clusters_json", "existing_classes", "domain_context")
            examples.append(example)

        return examples

    def proposal_metric(
        predicted: dspy.Prediction | str,
        expected: dspy.Example,
        trace: Any = None,
    ) -> float:
        """Metric for GEPA optimization: Jaccard similarity of class names.

        Compares the set of proposed class names against the expected set
        from approved review records. Returns 0.0-1.0.

        Args:
            predicted: DSPy Prediction or raw JSON string.
            expected: DSPy Example with expected analysis_json.
            trace: Optional trace for DSPy internals.

        Returns:
            Score between 0.0 and 1.0.
        """
        raw = (
            predicted.analysis_json
            if isinstance(predicted, dspy.Prediction)
            else getattr(predicted, "analysis_json", str(predicted))
        )
        expected_raw = (
            expected.analysis_json
            if isinstance(expected, dspy.Example)
            else getattr(expected, "analysis_json", str(expected))
        )

        try:
            pred_data = json.loads(raw) if isinstance(raw, str) else raw
            exp_data = (
                json.loads(expected_raw)
                if isinstance(expected_raw, str)
                else expected_raw
            )
        except (json.JSONDecodeError, TypeError):
            return 0.0

        pred_names = {p.get("name", "") for p in pred_data.get("proposed_classes", [])}
        exp_names = {p.get("name", "") for p in exp_data.get("proposed_classes", [])}

        if not exp_names and not pred_names:
            return 1.0
        if not exp_names or not pred_names:
            return 0.0

        intersection = pred_names & exp_names
        union = pred_names | exp_names
        return len(intersection) / len(union)

else:
    # Stub classes when dspy is not installed

    class ClusterProposalSignature:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "dspy is required for DSPy proposal optimization. "
                "Install with: pip install 'novel-entity-matcher[opinion]' or 'pip install dspy'"
            )

    class DSPyProposalModule:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "dspy is required for DSPy proposal optimization. "
                "Install with: pip install 'novel-entity-matcher[opinion]' or 'pip install dspy'"
            )

    def review_records_to_examples(
        approved_records: list[dict[str, Any]],
        rejected_records: list[dict[str, Any]] | None = None,
    ) -> list[Any]:
        raise ImportError(
            "dspy is required for DSPy proposal optimization. "
            "Install with: pip install 'novel-entity-matcher[opinion]' or 'pip install dspy'"
        )

    def proposal_metric(
        predicted: Any | str,
        expected: Any,
        trace: Any = None,
    ) -> float:
        raise ImportError(
            "dspy is required for DSPy proposal optimization. "
            "Install with: pip install 'novel-entity-matcher[opinion]' or 'pip install dspy'"
        )


def _record_to_cluster_json(record: dict[str, Any]) -> str:
    """Convert a review record's provenance cluster IDs into cluster JSON."""
    provenance = record.get("provenance", {})
    cluster_ids = provenance.get("cluster_ids", [])
    proposal = record.get("proposal", {})

    clusters = [
        {
            "cluster_id": cid,
            "sample_count": proposal.get("sample_count", 0),
            "keywords": provenance.get("keywords", []),
            "example_texts": proposal.get("example_samples", [])[:4],
        }
        for cid in cluster_ids
    ]
    if not clusters:
        clusters = [
            {
                "cluster_id": 0,
                "sample_count": proposal.get("sample_count", 0),
                "keywords": [],
                "example_texts": proposal.get("example_samples", [])[:4],
            }
        ]
    return json.dumps(clusters)


def _record_to_analysis_json(record: dict[str, Any]) -> str:
    """Convert a review record's proposal into analysis JSON."""
    proposal = record.get("proposal", {})
    return json.dumps(
        {
            "proposed_classes": [
                {
                    "name": proposal.get("name", ""),
                    "description": proposal.get("description", ""),
                    "confidence": proposal.get("confidence", 0.0),
                    "sample_count": proposal.get("sample_count", 0),
                    "example_samples": proposal.get("example_samples", []),
                    "justification": proposal.get("justification", ""),
                    "source_cluster_ids": proposal.get("source_cluster_ids", []),
                }
            ],
            "rejected_as_noise": [],
            "analysis_summary": "Optimized proposal.",
            "cluster_count": 1,
        }
    )


def _extract_existing_classes(record: dict[str, Any]) -> str:
    """Extract existing classes from a review record's provenance."""
    provenance = record.get("provenance", {})
    diagnostics = provenance.get("diagnostics", {})
    existing = diagnostics.get("existing_classes", [])
    return ", ".join(existing) if existing else "unknown"


def _extract_domain_context(record: dict[str, Any]) -> str:
    """Extract domain context from a review record's provenance."""
    provenance = record.get("provenance", {})
    diagnostics = provenance.get("diagnostics", {})
    return diagnostics.get("domain_context", "")
