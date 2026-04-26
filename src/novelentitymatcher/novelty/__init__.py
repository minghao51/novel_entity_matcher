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
# Clustering
from .clustering.scalable import ScalableClusterer
from .clustering.validation import ClusterValidator

# Configuration
from .config.base import DetectionConfig
from .config.strategies import (
    ClusteringConfig,
    ConfidenceConfig,
    KNNConfig,
    LOFConfig,
    MahalanobisConfig,
    OneClassConfig,
    PatternConfig,
    PrototypicalConfig,
    SelfKnowledgeConfig,
    SetFitCentroidConfig,
    SetFitConfig,
)
from .config.weights import WeightConfig
from .core.detector import NoveltyDetector
from .core.strategies import StrategyRegistry

# Evaluation
from .evaluation.evaluator import NoveltyEvaluator
from .evaluation.metrics import (
    compute_auprc,
    compute_auroc,
    compute_detection_rates,
    compute_precision_recall_f1,
)
from .evaluation.splitters import GradualNoveltySplitter, OODSplitter

# Extraction
from .extraction import ClusterEvidenceExtractor

# Proposers
from .proposal.llm import LLMClassProposer
from .proposal.retrieval import RetrievalAugmentedProposer
from .proposal.schema_enforcement import SchemaEnforcer

# Results and reports
from .schemas import (
    ClassProposal,
    ClusterEvidence,
    DiscoveryCluster,
    NovelClassAnalysis,
    NovelClassDiscoveryReport,
    NovelSampleMetadata,
    NovelSampleReport,
    ProposalReviewRecord,
)
from .schemas.results import EvaluationReport
from .storage.index import ANNBackend, ANNIndex

# Storage and indexing
from .storage.persistence import (
    export_summary,
    list_proposals,
    load_proposals,
    save_proposals,
)
from .storage.review import ProposalReviewManager
from .strategies import _register_all
from .strategies.base import NoveltyStrategy

_register_all()

__all__ = [
    "ANNBackend",
    "ANNIndex",
    "ClassProposal",
    "ClusterEvidence",
    # Extraction
    "ClusterEvidenceExtractor",
    "ClusterValidator",
    "ClusteringConfig",
    "ConfidenceConfig",
    # Configuration
    "DetectionConfig",
    "DiscoveryCluster",
    "EvaluationReport",
    "GradualNoveltySplitter",
    "KNNConfig",
    # Proposers
    "LLMClassProposer",
    "LOFConfig",
    "MahalanobisConfig",
    "NovelClassAnalysis",
    "NovelClassDiscoveryReport",
    # Results
    "NovelSampleMetadata",
    "NovelSampleReport",
    # Core
    "NoveltyDetector",
    # Evaluation
    "NoveltyEvaluator",
    "NoveltyStrategy",
    "OODSplitter",
    "OneClassConfig",
    "PatternConfig",
    "ProposalReviewManager",
    "ProposalReviewRecord",
    "PrototypicalConfig",
    "RetrievalAugmentedProposer",
    # Clustering
    "ScalableClusterer",
    "SchemaEnforcer",
    "SelfKnowledgeConfig",
    "SetFitCentroidConfig",
    "SetFitConfig",
    "StrategyRegistry",
    "WeightConfig",
    "compute_auprc",
    "compute_auroc",
    "compute_detection_rates",
    "compute_precision_recall_f1",
    "export_summary",
    "list_proposals",
    "load_proposals",
    # Storage
    "save_proposals",
]
