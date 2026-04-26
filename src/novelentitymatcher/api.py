"""Single import surface for the novel_entity_matcher public API.

Usage:
    from novelentitymatcher.api import *

    # or selective imports:
    from novelentitymatcher.api import (
        Matcher,
        NovelEntityMatcher,
        DiscoveryPipeline,
        PipelineConfig,
        DetectionConfig,
        NovelSampleMetadata,
        DiscoveryCluster,
        ClassProposal,
    )
"""

from __future__ import annotations

# --- Core matching ---
# --- Novelty-aware orchestration ---
# --- Pipeline ---
# --- Clustering ---
# --- Exceptions ---
from novelentitymatcher import (
    BlockingStrategy,
    BM25Blocking,
    ClusteringBackend,
    ClusteringBackendRegistry,
    CrossEncoderReranker,
    DiscoveryPipeline,
    FuzzyBlocking,
    HierarchicalMatcher,
    LLMClassProposer,
    Matcher,
    MatchingError,
    ModeError,
    NoOpBlocking,
    NovelEntityMatcher,
    NoveltyDetector,
    PipelineConfig,
    PipelineRunResult,
    PipelineStage,
    SemanticMatcherError,
    SetFitClassifier,
    StageContext,
    StageResult,
    TextNormalizer,
    TFIDFBlocking,
    TrainingError,
    ValidationError,
)

# --- Core extras ---
from novelentitymatcher.core.bert_classifier import BERTClassifier
from novelentitymatcher.core.embedding_matcher import EmbeddingMatcher
from novelentitymatcher.core.hierarchy import HierarchicalScoring, HierarchyIndex

# --- Clustering backends ---
from novelentitymatcher.novelty.clustering.backends import (
    HDBSCANBackend,
    SOPTICSBackend,
    UMAPHDBSCANBackend,
)
from novelentitymatcher.novelty.clustering.scalable import ScalableClusterer
from novelentitymatcher.novelty.clustering.validation import ClusterValidator

# --- Configs ---
from novelentitymatcher.novelty.config.base import DetectionConfig
from novelentitymatcher.novelty.config.strategies import (
    ClusteringConfig,
    ConfidenceConfig,
    KNNConfig,
    LOFConfig,
    MahalanobisConfig,
    OneClassConfig,
    PatternConfig,
    PrototypicalConfig,
    SelfKnowledgeConfig,
    SetFitConfig,
    UncertaintyConfig,
)
from novelentitymatcher.novelty.config.weights import WeightConfig
from novelentitymatcher.novelty.core.metadata import MetadataBuilder
from novelentitymatcher.novelty.core.signal_combiner import SignalCombiner

# --- Novelty core ---
from novelentitymatcher.novelty.core.strategies import StrategyRegistry

# --- Novelty entity matcher ---
from novelentitymatcher.novelty.entity_matcher import NovelEntityMatchResult
from novelentitymatcher.novelty.evaluation.evaluator import NoveltyEvaluator

# --- Evaluation ---
from novelentitymatcher.novelty.evaluation.splitters import (
    GradualNoveltySplitter,
    OODSplitter,
)

# --- Proposal ---
from novelentitymatcher.novelty.proposal.retrieval import (
    BGERetriever,
    RetrievalAugmentedProposer,
)

# --- Schemas ---
from novelentitymatcher.novelty.schemas.models import (
    ClassProposal,
    ClusterEvidence,
    DiscoveryCluster,
    NovelClassAnalysis,
    NovelClassDiscoveryReport,
    NovelSampleMetadata,
    NovelSampleReport,
    ProposalReviewRecord,
)
from novelentitymatcher.novelty.schemas.results import (
    DetectionReport,
    EvaluationReport,
    SampleMetrics,
    StrategyMetrics,
)
from novelentitymatcher.novelty.storage.index import ANNBackend, ANNIndex

# --- Storage ---
from novelentitymatcher.novelty.storage.review import (
    PromotionResult,
    ProposalReviewManager,
)

# --- Novelty strategies ---
from novelentitymatcher.novelty.strategies.base import NoveltyStrategy
from novelentitymatcher.novelty.strategies.clustering import ClusteringStrategy
from novelentitymatcher.novelty.strategies.confidence import ConfidenceStrategy
from novelentitymatcher.novelty.strategies.knn_distance import KNNDistanceStrategy
from novelentitymatcher.novelty.strategies.lof import LOFStrategy
from novelentitymatcher.novelty.strategies.mahalanobis import (
    MahalanobisDistanceStrategy,
)
from novelentitymatcher.novelty.strategies.oneclass import OneClassStrategy
from novelentitymatcher.novelty.strategies.pattern import PatternStrategy
from novelentitymatcher.novelty.strategies.prototypical import PrototypicalStrategy
from novelentitymatcher.novelty.strategies.self_knowledge import SelfKnowledgeStrategy
from novelentitymatcher.novelty.strategies.uncertainty import UncertaintyStrategy

# --- Pipeline helpers ---
from novelentitymatcher.pipeline.match_result import (
    MatchRecord,
    MatchResultWithMetadata,
)

__all__ = [
    "ANNBackend",
    "ANNIndex",
    "BERTClassifier",
    "BGERetriever",
    "BM25Blocking",
    "BlockingStrategy",
    "ClassProposal",
    "ClusterEvidence",
    "ClusterValidator",
    # Clustering
    "ClusteringBackend",
    "ClusteringBackendRegistry",
    "ClusteringConfig",
    "ClusteringStrategy",
    "ConfidenceConfig",
    "ConfidenceStrategy",
    "CrossEncoderReranker",
    # Configs
    "DetectionConfig",
    "DetectionReport",
    "DiscoveryCluster",
    # Pipeline
    "DiscoveryPipeline",
    "EmbeddingMatcher",
    "EvaluationReport",
    "FuzzyBlocking",
    "GradualNoveltySplitter",
    "HDBSCANBackend",
    "HierarchicalMatcher",
    "HierarchicalScoring",
    "HierarchyIndex",
    "KNNConfig",
    "KNNDistanceStrategy",
    "LLMClassProposer",
    "LOFConfig",
    "LOFStrategy",
    "MahalanobisConfig",
    "MahalanobisDistanceStrategy",
    "MatchRecord",
    "MatchResultWithMetadata",
    # Matching
    "Matcher",
    "MatchingError",
    "MetadataBuilder",
    "ModeError",
    "NoOpBlocking",
    "NovelClassAnalysis",
    "NovelClassDiscoveryReport",
    "NovelEntityMatchResult",
    # Novelty orchestration
    "NovelEntityMatcher",
    # Schemas
    "NovelSampleMetadata",
    "NovelSampleReport",
    "NoveltyDetector",
    "NoveltyEvaluator",
    # Novelty strategies
    "NoveltyStrategy",
    # Evaluation
    "OODSplitter",
    "OneClassConfig",
    "OneClassStrategy",
    "PatternConfig",
    "PatternStrategy",
    "PipelineConfig",
    "PipelineRunResult",
    "PipelineStage",
    # Storage
    "PromotionResult",
    "ProposalReviewManager",
    "ProposalReviewRecord",
    "PrototypicalConfig",
    "PrototypicalStrategy",
    # Proposal
    "RetrievalAugmentedProposer",
    "SOPTICSBackend",
    "SampleMetrics",
    "ScalableClusterer",
    "SelfKnowledgeConfig",
    "SelfKnowledgeStrategy",
    # Exceptions
    "SemanticMatcherError",
    "SetFitClassifier",
    "SetFitConfig",
    "SignalCombiner",
    "StageContext",
    "StageResult",
    "StrategyMetrics",
    # Novelty core
    "StrategyRegistry",
    "TFIDFBlocking",
    "TextNormalizer",
    "TrainingError",
    "UMAPHDBSCANBackend",
    "UncertaintyConfig",
    "UncertaintyStrategy",
    "ValidationError",
    "WeightConfig",
]
