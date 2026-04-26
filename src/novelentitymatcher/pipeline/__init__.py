"""Internal staged discovery pipeline contracts and adapters."""

from importlib import import_module

from .adapters import (
    ClusterEvidenceStage,
    CommunityDetectionStage,
    MatcherMetadataStage,
    OODDetectionStage,
    ProposalStage,
)
from .config import PipelineConfig
from .contracts import PipelineRunResult, PipelineStage, StageContext, StageResult
from .match_result import MatchRecord, MatchResultWithMetadata
from .orchestrator import PipelineOrchestrator

__all__ = [
    "ClusterEvidenceStage",
    "CommunityDetectionStage",
    "DiscoveryPipeline",
    "MatchRecord",
    "MatchResultWithMetadata",
    "MatcherMetadataStage",
    "OODDetectionStage",
    "PipelineConfig",
    "PipelineOrchestrator",
    "PipelineRunResult",
    "PipelineStage",
    "ProposalStage",
    "StageContext",
    "StageResult",
]


def __getattr__(name):
    if name != "DiscoveryPipeline":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".discovery", __name__), name)
    globals()[name] = value
    return value
