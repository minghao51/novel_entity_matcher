"""
Novel class detection and proposal system.

This module provides functionality for detecting novel classes in text data
and proposing meaningful new category names using LLMs.

The restructured module provides:
- Core detection via NoveltyDetector with strategy pattern
- Pluggable strategies for different detection algorithms
- Unified evaluation system for benchmarking and research
- Clean separation of concerns across submodules
"""

# Core detection
from .core.detector import NoveltyDetector
from .core.strategies import StrategyRegistry
from .strategies.base import NoveltyStrategy

# Configuration
from .config.base import DetectionConfig
from .config.strategies import (
    ConfidenceConfig,
    KNNConfig,
    ClusteringConfig,
    SelfKnowledgeConfig,
    PatternConfig,
    OneClassConfig,
    PrototypicalConfig,
    SetFitConfig,
    MahalanobisConfig,
    LOFConfig,
    SetFitCentroidConfig,
)
from .config.weights import WeightConfig

# Evaluation
from .evaluation.evaluator import NoveltyEvaluator
from .evaluation.metrics import (
    compute_auroc,
    compute_auprc,
    compute_detection_rates,
    compute_precision_recall_f1,
)
from .evaluation.splitters import OODSplitter, GradualNoveltySplitter

# Results and reports
from .schemas import (
    ClusterEvidence,
    DiscoveryCluster,
    ClassProposal,
    NovelClassAnalysis,
    NovelClassDiscoveryReport,
    NovelSampleMetadata,
    NovelSampleReport,
    ProposalReviewRecord,
)
from .schemas.results import EvaluationReport

# Storage and indexing
from .storage.persistence import (
    export_summary,
    save_proposals,
    load_proposals,
    list_proposals,
)
from .storage.index import ANNBackend, ANNIndex
from .storage.review import ProposalReviewManager

# Clustering
from .clustering.scalable import ScalableClusterer
from .clustering.validation import ClusterValidator

# Proposers
from .proposal.llm import LLMClassProposer
from .proposal.retrieval import RetrievalAugmentedProposer
from .proposal.schema_enforcement import SchemaEnforcer

# Extraction
from .extraction import ClusterEvidenceExtractor

from .strategies import _register_all

_register_all()

__all__ = [
    # Core
    "NoveltyDetector",
    "StrategyRegistry",
    "NoveltyStrategy",
    # Configuration
    "DetectionConfig",
    "ConfidenceConfig",
    "KNNConfig",
    "ClusteringConfig",
    "SelfKnowledgeConfig",
    "PatternConfig",
    "OneClassConfig",
    "PrototypicalConfig",
    "SetFitConfig",
    "SetFitCentroidConfig",
    "MahalanobisConfig",
    "LOFConfig",
    "WeightConfig",
    # Evaluation
    "NoveltyEvaluator",
    "compute_auroc",
    "compute_auprc",
    "compute_detection_rates",
    "compute_precision_recall_f1",
    "OODSplitter",
    "GradualNoveltySplitter",
    # Results
    "NovelSampleMetadata",
    "NovelSampleReport",
    "ClusterEvidence",
    "DiscoveryCluster",
    "ClassProposal",
    "NovelClassAnalysis",
    "NovelClassDiscoveryReport",
    "ProposalReviewRecord",
    "EvaluationReport",
    # Storage
    "save_proposals",
    "load_proposals",
    "list_proposals",
    "export_summary",
    "ProposalReviewManager",
    "ANNBackend",
    "ANNIndex",
    # Clustering
    "ScalableClusterer",
    "ClusterValidator",
    # Proposers
    "LLMClassProposer",
    "RetrievalAugmentedProposer",
    "SchemaEnforcer",
    # Extraction
    "ClusterEvidenceExtractor",
]
