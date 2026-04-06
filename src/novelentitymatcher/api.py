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
from novelentitymatcher import (
    Matcher,
    SetFitClassifier,
    TextNormalizer,
    CrossEncoderReranker,
    HierarchicalMatcher,
    BlockingStrategy,
    BM25Blocking,
    TFIDFBlocking,
    FuzzyBlocking,
    NoOpBlocking,
)

# --- Novelty-aware orchestration ---
from novelentitymatcher import (
    NovelEntityMatcher,
    NoveltyDetector,
    LLMClassProposer,
)

# --- Pipeline ---
from novelentitymatcher import (
    DiscoveryPipeline,
    PipelineConfig,
    PipelineStage,
    StageContext,
    StageResult,
    PipelineRunResult,
)

# --- Clustering ---
from novelentitymatcher import (
    ClusteringBackend,
    ClusteringBackendRegistry,
)

# --- Exceptions ---
from novelentitymatcher import (
    SemanticMatcherError,
    ValidationError,
    TrainingError,
    MatchingError,
    ModeError,
)

# --- Schemas ---
from novelentitymatcher.novelty.schemas.models import (
    NovelSampleMetadata,
    NovelSampleReport,
    ClusterEvidence,
    DiscoveryCluster,
    ClassProposal,
    NovelClassAnalysis,
    ProposalReviewRecord,
    NovelClassDiscoveryReport,
)

from novelentitymatcher.novelty.schemas.results import (
    StrategyMetrics,
    SampleMetrics,
    DetectionReport,
    EvaluationReport,
)

# --- Configs ---
from novelentitymatcher.novelty.config.base import DetectionConfig
from novelentitymatcher.novelty.config.strategies import (
    ConfidenceConfig,
    KNNConfig,
    UncertaintyConfig,
    ClusteringConfig,
    SelfKnowledgeConfig,
    PatternConfig,
    OneClassConfig,
    PrototypicalConfig,
    MahalanobisConfig,
    LOFConfig,
    SetFitConfig,
)
from novelentitymatcher.novelty.config.weights import WeightConfig

# --- Novelty strategies ---
from novelentitymatcher.novelty.strategies.base import NoveltyStrategy
from novelentitymatcher.novelty.strategies.confidence import ConfidenceStrategy
from novelentitymatcher.novelty.strategies.knn_distance import KNNDistanceStrategy
from novelentitymatcher.novelty.strategies.uncertainty import UncertaintyStrategy
from novelentitymatcher.novelty.strategies.clustering import ClusteringStrategy
from novelentitymatcher.novelty.strategies.self_knowledge import SelfKnowledgeStrategy
from novelentitymatcher.novelty.strategies.pattern import PatternStrategy
from novelentitymatcher.novelty.strategies.oneclass import OneClassStrategy
from novelentitymatcher.novelty.strategies.prototypical import PrototypicalStrategy
from novelentitymatcher.novelty.strategies.lof import LOFStrategy
from novelentitymatcher.novelty.strategies.mahalanobis import (
    MahalanobisDistanceStrategy,
)

# --- Novelty entity matcher ---
from novelentitymatcher.novelty.entity_matcher import NovelEntityMatchResult

# --- Novelty core ---
from novelentitymatcher.novelty.core.strategies import StrategyRegistry
from novelentitymatcher.novelty.core.signal_combiner import SignalCombiner
from novelentitymatcher.novelty.core.metadata import MetadataBuilder

# --- Clustering backends ---
from novelentitymatcher.novelty.clustering.backends import (
    HDBSCANBackend,
    SOPTICSBackend,
    UMAPHDBSCANBackend,
)
from novelentitymatcher.novelty.clustering.scalable import ScalableClusterer
from novelentitymatcher.novelty.clustering.validation import ClusterValidator

# --- Pipeline helpers ---
from novelentitymatcher.pipeline.match_result import (
    MatchRecord,
    MatchResultWithMetadata,
)

# --- Core extras ---
from novelentitymatcher.core.bert_classifier import BERTClassifier
from novelentitymatcher.core.embedding_matcher import EmbeddingMatcher
from novelentitymatcher.core.hierarchy import HierarchyIndex, HierarchicalScoring

# --- Evaluation ---
from novelentitymatcher.novelty.evaluation.splitters import (
    OODSplitter,
    GradualNoveltySplitter,
)
from novelentitymatcher.novelty.evaluation.evaluator import NoveltyEvaluator

# --- Storage ---
from novelentitymatcher.novelty.storage.review import (
    PromotionResult,
    ProposalReviewManager,
)
from novelentitymatcher.novelty.storage.index import ANNBackend, ANNIndex

# --- Proposal ---
from novelentitymatcher.novelty.proposal.retrieval import (
    RetrievalAugmentedProposer,
    BGERetriever,
)

__all__ = [
    # Matching
    "Matcher",
    "SetFitClassifier",
    "BERTClassifier",
    "TextNormalizer",
    "CrossEncoderReranker",
    "HierarchicalMatcher",
    "HierarchyIndex",
    "HierarchicalScoring",
    "EmbeddingMatcher",
    "BlockingStrategy",
    "BM25Blocking",
    "TFIDFBlocking",
    "FuzzyBlocking",
    "NoOpBlocking",
    # Novelty orchestration
    "NovelEntityMatcher",
    "NovelEntityMatchResult",
    "NoveltyDetector",
    "LLMClassProposer",
    # Pipeline
    "DiscoveryPipeline",
    "PipelineConfig",
    "PipelineStage",
    "StageContext",
    "StageResult",
    "PipelineRunResult",
    "MatchRecord",
    "MatchResultWithMetadata",
    # Clustering
    "ClusteringBackend",
    "ClusteringBackendRegistry",
    "HDBSCANBackend",
    "SOPTICSBackend",
    "UMAPHDBSCANBackend",
    "ScalableClusterer",
    "ClusterValidator",
    # Exceptions
    "SemanticMatcherError",
    "ValidationError",
    "TrainingError",
    "MatchingError",
    "ModeError",
    # Schemas
    "NovelSampleMetadata",
    "NovelSampleReport",
    "ClusterEvidence",
    "DiscoveryCluster",
    "ClassProposal",
    "NovelClassAnalysis",
    "ProposalReviewRecord",
    "NovelClassDiscoveryReport",
    "StrategyMetrics",
    "SampleMetrics",
    "DetectionReport",
    "EvaluationReport",
    # Configs
    "DetectionConfig",
    "ConfidenceConfig",
    "KNNConfig",
    "UncertaintyConfig",
    "ClusteringConfig",
    "SelfKnowledgeConfig",
    "PatternConfig",
    "OneClassConfig",
    "PrototypicalConfig",
    "MahalanobisConfig",
    "LOFConfig",
    "SetFitConfig",
    "WeightConfig",
    # Novelty strategies
    "NoveltyStrategy",
    "ConfidenceStrategy",
    "KNNDistanceStrategy",
    "UncertaintyStrategy",
    "ClusteringStrategy",
    "SelfKnowledgeStrategy",
    "PatternStrategy",
    "OneClassStrategy",
    "PrototypicalStrategy",
    "LOFStrategy",
    "MahalanobisDistanceStrategy",
    # Novelty core
    "StrategyRegistry",
    "SignalCombiner",
    "MetadataBuilder",
    # Evaluation
    "OODSplitter",
    "GradualNoveltySplitter",
    "NoveltyEvaluator",
    # Storage
    "PromotionResult",
    "ProposalReviewManager",
    "ANNBackend",
    "ANNIndex",
    # Proposal
    "RetrievalAugmentedProposer",
    "BGERetriever",
]
