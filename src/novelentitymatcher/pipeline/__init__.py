"""Internal staged discovery pipeline contracts and adapters."""

from ..core.match_result import MatchRecord, MatchResultWithMetadata
from .adapters import (
    ClusterEvidenceStage,
    CommunityDetectionStage,
    MatcherMetadataStage,
    OODDetectionStage,
    ProposalStage,
)
from .config import PipelineConfig
from .contracts import (
    PipelineArtifacts,
    PipelineRunResult,
    PipelineStage,
    PipelineStageError,
    StageContext,
    StageResult,
)
from .orchestrator import PipelineOrchestrator

__all__ = [
    "ClusterEvidenceStage",
    "CommunityDetectionStage",
    "DiscoveryPipeline",
    "MatchRecord",
    "MatchResultWithMetadata",
    "MatcherMetadataStage",
    "OODDetectionStage",
    "PipelineArtifacts",
    "PipelineConfig",
    "PipelineOrchestrator",
    "PipelineRunResult",
    "PipelineStage",
    "PipelineStageError",
    "ProposalStage",
    "StageContext",
    "StageResult",
]


def __getattr__(name):
    if name != "DiscoveryPipeline":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from ..novelty.entity_matcher import NovelEntityMatcher

    value = NovelEntityMatcher
    globals()[name] = value
    return value
