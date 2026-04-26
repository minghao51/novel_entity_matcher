"""HuggingFace benchmark module for novel_entity_matcher.

This module provides tools for benchmarking novel_entity_matcher on external
HuggingFace datasets for entity resolution, text classification, and
novelty detection tasks.

Usage:
    from novelentitymatcher.benchmarks import BenchmarkRunner, DATASET_REGISTRY

    runner = BenchmarkRunner()
    runner.load_all()

    er_results = runner.run_entity_resolution()
    clf_results = runner.run_classification()
    novelty_results = runner.run_novelty()

    all_results = runner.run_all()
"""

from ..novelty.entity_matcher import (
    NovelEntityMatcher,
    NovelEntityMatchResult,
    NoveltyMatchResult,
    create_novel_entity_matcher,
)
from .base import BaseEvaluator, EvaluationResult
from .classification import (
    ClassificationEvaluator,
    ClassificationSample,
    evaluate_by_class_count,
    sweep_num_classes,
)
from .entity_resolution import (
    EntityResolutionEvaluator,
    MatchPair,
    find_optimal_threshold,
    sweep_threshold,
)
from .loader import DatasetLoader
from .novelty import (
    NoveltyEvaluator,
    NoveltySample,
    find_optimal_knn_params,
    sweep_knn_params,
)
from .registry import (
    DATASET_REGISTRY,
    CacheConfig,
    DatasetConfig,
    get_dataset_config,
    get_datasets_by_task,
    get_default_datasets,
)
from .runner import BenchmarkRunner
from .shared import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OOD_RATIO,
    DEFAULT_RANDOM_SEED,
    SplitData,
    benchmark_inference,
    benchmark_training,
    compute_ood_metrics,
    generate_synthetic_data,
    prepare_binary_labels,
    timer,
)

__all__ = [
    "DATASET_REGISTRY",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_OOD_RATIO",
    "DEFAULT_RANDOM_SEED",
    "BaseEvaluator",
    "BenchmarkRunner",
    "CacheConfig",
    "ClassificationEvaluator",
    "ClassificationSample",
    "DatasetConfig",
    "DatasetLoader",
    "EntityResolutionEvaluator",
    "EvaluationResult",
    "MatchPair",
    "NovelEntityMatchResult",
    "NovelEntityMatcher",
    "NoveltyEvaluator",
    "NoveltyMatchResult",
    "NoveltySample",
    "SplitData",
    "benchmark_inference",
    "benchmark_training",
    "compute_ood_metrics",
    "create_novel_entity_matcher",
    "evaluate_by_class_count",
    "find_optimal_knn_params",
    "find_optimal_threshold",
    "generate_synthetic_data",
    "get_dataset_config",
    "get_datasets_by_task",
    "get_default_datasets",
    "prepare_binary_labels",
    "sweep_knn_params",
    "sweep_num_classes",
    "sweep_threshold",
    "timer",
]
