from importlib import import_module
import warnings
import os

# Configure logging early, before other imports
# Check SEMANTIC_MATCHER_VERBOSE environment variable
_verbose = os.getenv("SEMANTIC_MATCHER_VERBOSE", "false").lower() == "true"
try:
    from .utils.logging_config import configure_logging
    configure_logging(verbose=_verbose)
except Exception:
    # If logging configuration fails, continue without it
    pass

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("semantic-matcher")
except Exception:  # pragma: no cover - fallback for local source usage
    __version__ = "0.1.0"

__all__ = [
    "Matcher",  # New unified API
    "EntityMatcher",  # Deprecated - use Matcher
    "EmbeddingMatcher",  # Deprecated - use Matcher
    "SetFitClassifier",
    "TextNormalizer",
    "CrossEncoderReranker",
    "HybridMatcher",  # Deprecated - use Matcher(mode='hybrid')
    "HierarchicalMatcher",  # Hierarchical entity matching
    "BlockingStrategy",
    "BM25Blocking",
    "TFIDFBlocking",
    "FuzzyBlocking",
    "NoOpBlocking",
    # Exceptions
    "SemanticMatcherError",
    "ValidationError",
    "TrainingError",
    "MatchingError",
    "ModeError",
]

_EXPORTS = {
    "Matcher": (".core.matcher", "Matcher"),
    "EntityMatcher": (".core.matcher", "EntityMatcher"),
    "EmbeddingMatcher": (".core.matcher", "EmbeddingMatcher"),
    "SetFitClassifier": (".core.classifier", "SetFitClassifier"),
    "TextNormalizer": (".core.normalizer", "TextNormalizer"),
    "CrossEncoderReranker": (".core.reranker", "CrossEncoderReranker"),
    "HybridMatcher": (".core.hybrid", "HybridMatcher"),
    "HierarchicalMatcher": (".core.hierarchy", "HierarchicalMatcher"),
    "BlockingStrategy": (".core.blocking", "BlockingStrategy"),
    "BM25Blocking": (".core.blocking", "BM25Blocking"),
    "TFIDFBlocking": (".core.blocking", "TFIDFBlocking"),
    "FuzzyBlocking": (".core.blocking", "FuzzyBlocking"),
    "NoOpBlocking": (".core.blocking", "NoOpBlocking"),
    # Exceptions
    "SemanticMatcherError": (".exceptions", "SemanticMatcherError"),
    "ValidationError": (".exceptions", "ValidationError"),
    "TrainingError": (".exceptions", "TrainingError"),
    "MatchingError": (".exceptions", "MatchingError"),
    "ModeError": (".exceptions", "ModeError"),
}

# Classes that should show deprecation warnings
_DEPRECATED_CLASSES = {
    "EntityMatcher": "Matcher",
    "EmbeddingMatcher": "Matcher",
    "HybridMatcher": "Matcher",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Show deprecation warning for old matcher classes
    if name in _DEPRECATED_CLASSES:
        replacement = _DEPRECATED_CLASSES[name]
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            f"Use the unified {replacement} class instead. "
            f"See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
