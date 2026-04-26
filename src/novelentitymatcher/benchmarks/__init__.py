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

from .registry import (
    DATASET_REGISTRY,
    DatasetConfig,
    CacheConfig,
    get_dataset_config,
    get_datasets_by_task,
    get_default_datasets,
)

from .loader import DatasetLoader

from .runner import BenchmarkRunner

from .entity_resolution import (
    EntityResolutionEvaluator,
    sweep_threshold,
    find_optimal_threshold,
    MatchPair,
)

from .classification import (
    ClassificationEvaluator,
    sweep_num_classes,
    evaluate_by_class_count,
    ClassificationSample,
)

from .novelty import (
    NoveltyEvaluator,
    sweep_knn_params,
    find_optimal_knn_params,
    NoveltySample,
)

from ..novelty.entity_matcher import (
    NovelEntityMatcher,
    NovelEntityMatchResult,
    NoveltyMatchResult,
    create_novel_entity_matcher,
)

from .base import BaseEvaluator, EvaluationResult

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
    "DatasetConfig",
    "CacheConfig",
    "get_dataset_config",
    "get_datasets_by_task",
    "get_default_datasets",
    "DatasetLoader",
    "BenchmarkRunner",
    "BaseEvaluator",
    "EvaluationResult",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_OOD_RATIO",
    "DEFAULT_RANDOM_SEED",
    "SplitData",
    "timer",
    "compute_ood_metrics",
    "prepare_binary_labels",
    "generate_synthetic_data",
    "benchmark_training",
    "benchmark_inference",
    "EntityResolutionEvaluator",
    "sweep_threshold",
    "find_optimal_threshold",
    "MatchPair",
    "ClassificationEvaluator",
    "sweep_num_classes",
    "evaluate_by_class_count",
    "ClassificationSample",
    "NoveltyEvaluator",
    "sweep_knn_params",
    "find_optimal_knn_params",
    "NoveltySample",
    "NovelEntityMatcher",
    "NovelEntityMatchResult",
    "NoveltyMatchResult",
    "create_novel_entity_matcher",
]
