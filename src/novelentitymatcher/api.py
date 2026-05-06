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
from novelentitymatcher.core.bert_classifier import BERTClassifier
from novelentitymatcher.core.embedding_matcher import EmbeddingMatcher
from novelentitymatcher.core.hierarchy import HierarchicalScoring, HierarchyIndex
from novelentitymatcher.core.vector_store import InMemoryVectorStore, VectorStore
from novelentitymatcher.novelty.clustering.backends import (
    HDBSCANBackend,
    SOPTICSBackend,
    UMAPHDBSCANBackend,
)
from novelentitymatcher.novelty.clustering.scalable import ScalableClusterer
from novelentitymatcher.novelty.clustering.validation import ClusterValidator
from novelentitymatcher.novelty.config.base import DetectionConfig
from novelentitymatcher.novelty.config.strategies import (
    ClusteringConfig,
    ConfidenceConfig,
    EnergyConfig,
    KNNConfig,
    LOFConfig,
    MahalanobisConfig,
    MixtureGaussianConfig,
    OneClassConfig,
    PatternConfig,
    PrototypicalConfig,
    ReActConfig,
    SelfKnowledgeConfig,
    SetFitCentroidConfig,
    SetFitConfig,
    UncertaintyConfig,
)
from novelentitymatcher.novelty.config.weights import WeightConfig
from novelentitymatcher.novelty.core.metadata import MetadataBuilder
from novelentitymatcher.novelty.core.score_calibrator import OODScoreCalibrator
from novelentitymatcher.novelty.core.signal_combiner import SignalCombiner
from novelentitymatcher.novelty.core.strategies import StrategyRegistry
from novelentitymatcher.novelty.entity_matcher import NovelEntityMatchResult
from novelentitymatcher.novelty.evaluation.evaluator import NoveltyEvaluator
from novelentitymatcher.novelty.evaluation.splitters import (
    GradualNoveltySplitter,
    OODSplitter,
)
from novelentitymatcher.novelty.proposal.retrieval import (
    BGERetriever,
    RetrievalAugmentedProposer,
)
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
from novelentitymatcher.novelty.storage.review import (
    PromotionResult,
    ProposalReviewManager,
)
from novelentitymatcher.novelty.strategies.base import NoveltyStrategy
from novelentitymatcher.novelty.strategies.clustering import ClusteringStrategy
from novelentitymatcher.novelty.strategies.confidence import ConfidenceStrategy
from novelentitymatcher.novelty.strategies.energy import EnergyOODStrategy
from novelentitymatcher.novelty.strategies.knn_distance import KNNDistanceStrategy
from novelentitymatcher.novelty.strategies.lof import LOFStrategy
from novelentitymatcher.novelty.strategies.mahalanobis import (
    MahalanobisDistanceStrategy,
)
from novelentitymatcher.novelty.strategies.mixture_gaussian import (
    MixtureGaussianStrategy,
)
from novelentitymatcher.novelty.strategies.oneclass import OneClassStrategy
from novelentitymatcher.novelty.strategies.pattern import PatternStrategy
from novelentitymatcher.novelty.strategies.prototypical import PrototypicalStrategy
from novelentitymatcher.novelty.strategies.react_hybrid import ReActEnergyStrategy
from novelentitymatcher.novelty.strategies.self_knowledge import SelfKnowledgeStrategy
from novelentitymatcher.novelty.strategies.setfit_centroid import SetFitCentroidStrategy
from novelentitymatcher.novelty.strategies.uncertainty import UncertaintyStrategy
from novelentitymatcher.pipeline.match_result import (
    MatchRecord,
    MatchResultWithMetadata,
)
from novelentitymatcher.utils.embedding_cache import LRUEmbeddingCache

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
    "ClusteringBackend",
    "ClusteringBackendRegistry",
    "ClusteringConfig",
    "ClusteringStrategy",
    "ConfidenceConfig",
    "ConfidenceStrategy",
    "CrossEncoderReranker",
    "DetectionConfig",
    "DetectionReport",
    "DiscoveryCluster",
    "DiscoveryPipeline",
    "EmbeddingMatcher",
    "EnergyConfig",
    "EnergyOODStrategy",
    "EvaluationReport",
    "FuzzyBlocking",
    "GradualNoveltySplitter",
    "HDBSCANBackend",
    "HierarchicalMatcher",
    "HierarchicalScoring",
    "HierarchyIndex",
    "InMemoryVectorStore",
    "KNNConfig",
    "KNNDistanceStrategy",
    "LLMClassProposer",
    "LOFConfig",
    "LOFStrategy",
    "LRUEmbeddingCache",
    "MahalanobisConfig",
    "MahalanobisDistanceStrategy",
    "MatchRecord",
    "MatchResultWithMetadata",
    "Matcher",
    "MatchingError",
    "MetadataBuilder",
    "MixtureGaussianConfig",
    "MixtureGaussianStrategy",
    "ModeError",
    "NoOpBlocking",
    "NovelClassAnalysis",
    "NovelClassDiscoveryReport",
    "NovelEntityMatchResult",
    "NovelEntityMatcher",
    "NovelSampleMetadata",
    "NovelSampleReport",
    "NoveltyDetector",
    "NoveltyEvaluator",
    "NoveltyStrategy",
    "OODScoreCalibrator",
    "OODSplitter",
    "OneClassConfig",
    "OneClassStrategy",
    "PatternConfig",
    "PatternStrategy",
    "PipelineConfig",
    "PipelineRunResult",
    "PipelineStage",
    "PromotionResult",
    "ProposalReviewManager",
    "ProposalReviewRecord",
    "PrototypicalConfig",
    "PrototypicalStrategy",
    "ReActConfig",
    "ReActEnergyStrategy",
    "RetrievalAugmentedProposer",
    "SOPTICSBackend",
    "SampleMetrics",
    "ScalableClusterer",
    "SelfKnowledgeConfig",
    "SelfKnowledgeStrategy",
    "SemanticMatcherError",
    "SetFitCentroidConfig",
    "SetFitCentroidStrategy",
    "SetFitClassifier",
    "SetFitConfig",
    "SignalCombiner",
    "StageContext",
    "StageResult",
    "StrategyMetrics",
    "StrategyRegistry",
    "TFIDFBlocking",
    "TextNormalizer",
    "TrainingError",
    "UMAPHDBSCANBackend",
    "UncertaintyConfig",
    "UncertaintyStrategy",
    "ValidationError",
    "VectorStore",
    "WeightConfig",
]
