"""Single import surface for the novel_entity_matcher public API.

Usage:
    from novelentitymatcher.api import Matcher, NovelEntityMatcher, DiscoveryPipeline
    # or
    from novelentitymatcher.api import *

This module re-exports everything from the package's public surface.
"""

# ruff: noqa: F405
from . import *  # noqa: F403

# Additional imports beyond what __init__ provides via lazy loading
from .core.bert_classifier import BERTClassifier
from .core.embedding_matcher import EmbeddingMatcher
from .core.hierarchy import HierarchicalScoring, HierarchyIndex
from .core.vector_store import InMemoryVectorStore, VectorStore
from .novelty.active_learning.annotation import AnnotationCollector, AnnotationResult
from .novelty.active_learning.sampler import UncertaintySampler
from .novelty.clustering.backends import (
    HDBSCANBackend,
    SOPTICSBackend,
    UMAPHDBSCANBackend,
)
from .novelty.clustering.scalable import ScalableClusterer
from .novelty.clustering.validation import ClusterValidator
from .novelty.config.base import DetectionConfig
from .novelty.config.strategies import (
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
from .novelty.config.weights import WeightConfig
from .novelty.core.metadata import MetadataBuilder
from .novelty.core.score_calibrator import OODScoreCalibrator
from .novelty.core.signal_combiner import SignalCombiner
from .novelty.core.strategies import StrategyRegistry
from .novelty.entity_matcher import NovelEntityMatchResult
from .novelty.evaluation.evaluator import NoveltyEvaluator
from .novelty.evaluation.splitters import GradualNoveltySplitter, OODSplitter
from .novelty.proposal.retrieval import BGERetriever, RetrievalAugmentedProposer
from .novelty.schemas.models import (
    ClassProposal,
    ClusterEvidence,
    DiscoveryCluster,
    NovelClassAnalysis,
    NovelClassDiscoveryReport,
    NovelSampleMetadata,
    NovelSampleReport,
    ProposalReviewRecord,
)
from .novelty.schemas.results import (
    DetectionReport,
    EvaluationReport,
    SampleMetrics,
    StrategyMetrics,
)
from .novelty.storage.index import ANNBackend, ANNIndex
from .novelty.storage.review import PromotionResult, ProposalReviewManager
from .novelty.strategies.base import NoveltyStrategy
from .novelty.strategies.clustering import ClusteringStrategy
from .novelty.strategies.confidence import ConfidenceStrategy
from .novelty.strategies.energy import EnergyOODStrategy
from .novelty.strategies.knn_distance import KNNDistanceStrategy
from .novelty.strategies.lof import LOFStrategy
from .novelty.strategies.mahalanobis import MahalanobisDistanceStrategy
from .novelty.strategies.mixture_gaussian import MixtureGaussianStrategy
from .novelty.strategies.oneclass import OneClassStrategy
from .novelty.strategies.pattern import PatternStrategy
from .novelty.strategies.prototypical import PrototypicalStrategy
from .novelty.strategies.react_hybrid import ReActEnergyStrategy
from .novelty.strategies.self_knowledge import SelfKnowledgeStrategy
from .novelty.strategies.setfit_centroid import SetFitCentroidStrategy
from .novelty.strategies.uncertainty import UncertaintyStrategy
from .pipeline.match_result import MatchRecord, MatchResultWithMetadata
from .utils.embeddings import LRUEmbeddingCache

__all__ = [
    "ANNBackend",
    "ANNIndex",
    "AnnotationCollector",
    "AnnotationResult",
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
    "UncertaintySampler",
    "UncertaintyStrategy",
    "ValidationError",
    "VectorStore",
    "WeightConfig",
]
