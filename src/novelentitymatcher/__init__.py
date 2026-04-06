"""
novel_entity_matcher - Entity matching with novelty detection and discovery.

Public API hierarchy:
- Matcher: Known-entity matching (zero-shot, SetFit, BERT, hybrid modes)
- NovelEntityMatcher: Novelty-aware matching (matcher-first orchestration)
- DiscoveryPipeline: Pipeline-first discovery with staged processing

For new projects, prefer DiscoveryPipeline for discovery workflows.
NovelEntityMatcher remains supported for matcher-first patterns.
"""

from importlib import import_module
import os

# Configure logging early, before other imports
# Check NOVEL_ENTITY_MATCHER_VERBOSE environment variable
_verbose = os.getenv("NOVEL_ENTITY_MATCHER_VERBOSE", "false").lower() == "true"
try:
    from .utils.logging_config import configure_logging

    configure_logging(verbose=_verbose)
except Exception:
    # If logging configuration fails, continue without it
    pass

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("novel-entity-matcher")
except Exception:  # pragma: no cover - fallback for local source usage
    __version__ = "0.1.0"

__all__ = [
    "Matcher",
    "SetFitClassifier",
    "TextNormalizer",
    "CrossEncoderReranker",
    "HierarchicalMatcher",
    "BlockingStrategy",
    "BM25Blocking",
    "TFIDFBlocking",
    "FuzzyBlocking",
    "NoOpBlocking",
    "NovelEntityMatcher",
    "DiscoveryPipeline",
    "NoveltyDetector",
    "LLMClassProposer",
    "SemanticMatcherError",
    "ValidationError",
    "TrainingError",
    "MatchingError",
    "ModeError",
    "PipelineConfig",
    "PipelineStage",
    "StageContext",
    "StageResult",
    "PipelineRunResult",
    "ClusteringBackend",
    "ClusteringBackendRegistry",
]

_EXPORTS = {
    "Matcher": (".core.matcher", "Matcher"),
    "SetFitClassifier": (".core.classifier", "SetFitClassifier"),
    "TextNormalizer": (".core.normalizer", "TextNormalizer"),
    "CrossEncoderReranker": (".core.reranker", "CrossEncoderReranker"),
    "HierarchicalMatcher": (".core.hierarchy", "HierarchicalMatcher"),
    "BlockingStrategy": (".core.blocking", "BlockingStrategy"),
    "BM25Blocking": (".core.blocking", "BM25Blocking"),
    "TFIDFBlocking": (".core.blocking", "TFIDFBlocking"),
    "FuzzyBlocking": (".core.blocking", "FuzzyBlocking"),
    "NoOpBlocking": (".core.blocking", "NoOpBlocking"),
    "NovelEntityMatcher": (".novelty.entity_matcher", "NovelEntityMatcher"),
    "DiscoveryPipeline": (".pipeline.discovery", "DiscoveryPipeline"),
    "NoveltyDetector": (".novelty.core.detector", "NoveltyDetector"),
    "LLMClassProposer": (".novelty.proposal.llm", "LLMClassProposer"),
    "SemanticMatcherError": (".exceptions", "SemanticMatcherError"),
    "ValidationError": (".exceptions", "ValidationError"),
    "TrainingError": (".exceptions", "TrainingError"),
    "MatchingError": (".exceptions", "MatchingError"),
    "ModeError": (".exceptions", "ModeError"),
    "PipelineConfig": (".pipeline.config", "PipelineConfig"),
    "PipelineStage": (".pipeline.contracts", "PipelineStage"),
    "StageContext": (".pipeline.contracts", "StageContext"),
    "StageResult": (".pipeline.contracts", "StageResult"),
    "PipelineRunResult": (".pipeline.contracts", "PipelineRunResult"),
    "ClusteringBackend": (".novelty.clustering.base", "ClusteringBackend"),
    "ClusteringBackendRegistry": (
        ".novelty.clustering.backends",
        "ClusteringBackendRegistry",
    ),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
