"""CLI for HuggingFace benchmarks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .runner import BenchmarkRunner
from .registry import DATASET_REGISTRY, get_datasets_by_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_run_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("run", help="Run benchmarks")
    parser.add_argument(
        "--task",
        choices=[
            "all",
            "entity_resolution",
            "er",
            "classification",
            "novelty",
            "processed",
        ],
        default="all",
        help="Which benchmark task to run (er=entity_resolution)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to run (default: all available)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["potion-32m"],
        help="Embedding models to benchmark",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["zero-shot"],
        help="Matcher modes to test",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Thresholds to sweep for entity resolution",
    )
    parser.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=[0.2, 0.3, 0.4, 0.5],
        help="Confidence thresholds for novelty detection",
    )
    parser.add_argument(
        "--class-counts",
        nargs="+",
        type=int,
        default=[4, 10, 28],
        help="Class counts to test for classification scaling",
    )
    parser.add_argument(
        "--ood-ratio",
        type=float,
        default=0.2,
        help="Ratio of classes to hold out as OOD for novelty detection",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=500,
        help="Max training samples per dataset for trained modes (default: 500)",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Max test samples per dataset (default: all)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Body L2 regularization (default: auto by sample count)",
    )
    parser.add_argument(
        "--head-c",
        type=float,
        default=None,
        help="Head inverse-L2 strength (default: auto by sample count)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Contrastive training iterations (default: 1)",
    )
    parser.add_argument(
        "--no-regularize",
        action="store_true",
        help="Disable auto-regularization (use old defaults)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path for results JSON",
    )
    return parser


def add_load_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("load", help="Load/download datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to load (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    return parser


def add_list_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("list", help="List available datasets")
    parser.add_argument(
        "--task",
        choices=["all", "entity_resolution", "classification", "novelty"],
        default="all",
        help="Filter by task type",
    )
    return parser


def add_clear_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("clear", help="Clear cached datasets")
    parser.add_argument(
        "--dataset",
        help="Specific dataset to clear (default: all)",
    )
    return parser


def add_sweep_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("sweep", help="Run parameter sweep")
    parser.add_argument(
        "--task",
        choices=["er", "clf", "novelty"],
        required=True,
        help="Task type for sweep",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset to use for sweep",
    )
    parser.add_argument(
        "--param",
        choices=[
            "threshold", "k", "distance",
            "knn_k", "knn_metric",
            "lof_neighbors", "lof_metric",
            "svm_nu", "svm_kernel",
            "mahalanobis_conformal", "mahalanobis_threshold",
            "cluster_min_size", "cluster_persistence",
            "centroid_percentile",
            "self_knowledge_dim", "self_knowledge_epochs",
        ],
        required=True,
        help="Parameter to sweep",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        help="Values to sweep",
    )
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--output", default=None)
    return parser


def add_bench_classifier_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("bench-classifier", help="Benchmark classifiers")
    parser.add_argument(
        "--mode",
        choices=["compare", "sweep-models", "sweep-setfit", "scale-test", "sweep-modes"],
        default="compare",
        help="'compare' BERT vs SetFit, 'sweep-models' BERT sweep, "
             "'sweep-setfit' SetFit model sweep, 'scale-test' sample scaling, "
             "'sweep-modes' mode comparison",
    )
    parser.add_argument("--num-entities", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["distilbert", "tinybert", "roberta-base"])
    parser.add_argument("--output", type=Path)
    return parser


def add_bench_weights_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("bench-weights", help="Bayesian optimization of ensemble weights")
    parser.add_argument("--dataset", default="ag_news")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--output", default=None, help="Output JSON path")
    return parser


def add_bench_ann_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("bench-ann", help="Benchmark ANN backends (hnswlib vs faiss vs exact)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--output", default=None)
    return parser


def add_bench_reranker_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("bench-reranker", help="Benchmark reranker models")
    parser.add_argument("--models", nargs="+", default=["bge-m3", "bge-large", "ms-marco"])
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--output", default=None)
    return parser


def add_bench_novelty_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("bench-novelty", help="Benchmark novelty detection strategies")
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "full"],
        default="standard",
        help="quick (basic), standard (+ ensembles), full (+ signal combiner)",
    )
    parser.add_argument("--datasets", nargs="+", default=["ag_news", "go_emotions"])
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Single model")
    parser.add_argument("--models", nargs="+", default=None, help="Multiple models to sweep")
    parser.add_argument("--output", default=None, help="Output CSV path")
    return parser


def add_bench_async_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("bench-async", help="Benchmark sync vs async matcher APIs")
    parser.add_argument("--multiplier", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--section", default="languages/languages")
    parser.add_argument("--model", default="default")
    parser.add_argument("--modes", nargs="+", default=["zero-shot"])
    parser.add_argument("--max-entities", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=25)
    parser.add_argument("--output", type=Path)
    return parser


def add_render_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("render", help="Render benchmark JSON as markdown")
    parser.add_argument("input", type=Path, help="Benchmark JSON file")
    return parser


def add_plot_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("plot", help="Generate charts from benchmark JSON")
    parser.add_argument("--embedding-results", required=True)
    parser.add_argument("--training-results", required=True)
    parser.add_argument("--bert-results", required=True)
    parser.add_argument("--output-dir", default="docs/images/benchmarks")
    return parser


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="novelentitymatcher-bench",
        description="HuggingFace benchmark runner for novel_entity_matcher",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_run_parser(subparsers)
    add_load_parser(subparsers)
    add_list_parser(subparsers)
    add_clear_parser(subparsers)
    add_sweep_parser(subparsers)
    add_bench_classifier_parser(subparsers)
    add_bench_novelty_parser(subparsers)
    add_bench_async_parser(subparsers)
    add_bench_weights_parser(subparsers)
    add_bench_ann_parser(subparsers)
    add_bench_reranker_parser(subparsers)
    add_render_parser(subparsers)
    add_plot_parser(subparsers)

    return parser


def list_datasets(task: str = "all") -> None:
    if task == "all":
        for name, config in DATASET_REGISTRY.items():
            logger.info(f"{name}: {config.hf_path} ({config.task_type})")
    else:
        task_map = {
            "entity_resolution": "entity_matching",
            "classification": "classification",
            "novelty": "novelty",
        }
        task_type = task_map.get(task, task)
        datasets = get_datasets_by_task(task_type)  # type: ignore[arg-type]
        for name, config in datasets.items():
            logger.info(f"{name}: {config.hf_path}")


def load_datasets(
    datasets: list[str] | None = None,
    force: bool = False,
) -> None:
    runner = BenchmarkRunner()
    results = runner.load_all(datasets=datasets, force_redownload=force)
    for name, data in results.items():
        if "error" in data:
            logger.error(f"Failed to load {name}: {data['error']}")
        else:
            logger.info(
                f"Loaded {name}: {data.get('metadata', {}).get('num_rows', 'unknown')} rows"
            )


def run_benchmarks(
    task: str = "all",
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    modes: list[str] | None = None,
    thresholds: list[float] | None = None,
    class_counts: list[int] | None = None,
    ood_ratio: float = 0.2,
    output: Path | None = None,
    confidence_thresholds: list[float] | None = None,
    max_train_samples: int | None = 500,
    max_test_samples: int | None = None,
    weight_decay: float | None = None,
    head_c: float | None = None,
    num_iterations: int | None = None,
    no_regularize: bool = False,
) -> None:
    runner = BenchmarkRunner()
    results_to_save = None

    if task == "all":
        results = runner.run_all(
            embedding_models=models,
            modes=modes,
            thresholds=thresholds,
            class_counts=class_counts,
            ood_ratio=ood_ratio,
        )
        results_to_save = results
        logger.info("\n=== Entity Resolution Results ===")
        for model_results in results["entity_resolution"]:
            for r in model_results:
                logger.info(f"  {r.get('dataset')}: F1={r.get('f1', 0):.4f}")

        logger.info("\n=== Classification Results ===")
        for model_results in results["classification"]:
            for r in model_results:
                logger.info(
                    f"  {r.get('dataset')}: Accuracy={r.get('accuracy', 0):.4f}"
                )

        logger.info("\n=== Novelty Detection Results ===")
        for model_results in results["novelty"]:
            for r in model_results:
                logger.info(f"  {r.get('dataset')}: AUROC={r.get('auroc', 0):.4f}")

    elif task in ("entity_resolution", "er"):
        df = runner.run_entity_resolution_benchmark(
            datasets=datasets,
            model=models[0] if models else "all-MiniLM-L6-v2",
            thresholds=thresholds,
        )
        results_to_save = {
            "metadata": {"task": task},
            "entity_resolution": df.to_dict(orient="records"),
        }
        logger.info(df.to_string(index=False))

    elif task == "classification":
        clf_results = []
        for mode in modes or ["zero-shot"]:
            fit_kwargs = {}
            if weight_decay is not None:
                fit_kwargs["weight_decay"] = weight_decay
            if head_c is not None:
                fit_kwargs["head_c"] = head_c
            if num_iterations is not None:
                fit_kwargs["num_iterations"] = num_iterations
            df = runner.run_classification(
                datasets=datasets,
                model=models[0] if models else "potion-32m",
                mode=mode,
                class_counts=class_counts,
                max_train_samples=max_train_samples,
                max_test_samples=max_test_samples,
                per_split=True,
                regularize=not no_regularize,
                **fit_kwargs,
            )
            clf_results.extend(df.to_dict(orient="records"))
            logger.info(f"\n--- mode={mode} ---")
            logger.info(df.to_string(index=False))
        results_to_save = {
            "metadata": {"task": task},
            "classification": clf_results,
        }

    elif task == "novelty":
        novelty_results = []
        df = runner.run_novelty(
            datasets=datasets,
            model=models[0] if models else "potion-32m",
            ood_ratio=ood_ratio,
        )
        novelty_results.extend(df.to_dict(orient="records"))
        logger.info("\n--- Novelty Detection ---")
        logger.info(df.to_string(index=False))
        results_to_save = {
            "metadata": {"task": task},
            "novelty": novelty_results,
        }

    elif task == "processed":
        df = runner.run_novelty_on_processed(
            datasets=datasets,
            model=models[0] if models else "potion-32m",
            confidence_thresholds=confidence_thresholds
            or thresholds
            or [0.2, 0.3, 0.4, 0.5],
        )
        results_to_save = {
            "metadata": {"task": task},
            "processed": df.to_dict(orient="records"),
        }
        logger.info(df.to_string(index=False))

    if output:
        runner.save_results(
            results_to_save or {"metadata": {"task": task}}, str(output)
        )
        logger.info(f"\nResults saved to {output}")


def clear_cache(dataset: str | None = None) -> None:
    runner = BenchmarkRunner()
    runner.loader.clear_cache(dataset)
    logger.info("Cache cleared" if dataset else "All caches cleared")


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.command == "run":
            run_benchmarks(
                task=args.task,
                datasets=args.datasets,
                models=args.models,
                modes=args.modes,
                thresholds=args.thresholds,
                class_counts=args.class_counts,
                ood_ratio=args.ood_ratio,
                output=args.output,
                confidence_thresholds=getattr(args, "confidence_thresholds", None),
                max_train_samples=getattr(args, "max_train_samples", 500),
                max_test_samples=getattr(args, "max_test_samples", None),
                weight_decay=getattr(args, "weight_decay", None),
                head_c=getattr(args, "head_c", None),
                num_iterations=getattr(args, "num_iterations", None),
                no_regularize=getattr(args, "no_regularize", False),
            )

        elif args.command == "load":
            load_datasets(
                datasets=args.datasets,
                force=args.force,
            )

        elif args.command == "list":
            list_datasets(task=args.task)

        elif args.command == "clear":
            clear_cache(dataset=args.dataset)

        elif args.command == "sweep":
            from .novelty_bench import NoveltyBenchmark, load_and_split_data
            from .shared import prepare_binary_labels, compute_ood_metrics

            ds_name = args.dataset
            split = load_and_split_data(
                ds_name, args.max_train, args.max_val, args.max_test,
            )
            benchmark = NoveltyBenchmark(args.model)

            param = args.param
            values = args.values or []

            if param == "knn_k":
                values = values or [1, 3, 5, 7, 10, 15, 20, 30, 50]
                results = benchmark.benchmark_knn(split, k_values=[int(v) for v in values])
            elif param == "lof_neighbors":
                values = values or [5, 10, 15, 20, 30, 50]
                results = benchmark.benchmark_lof(split, n_neighbors_list=[int(v) for v in values])
            elif param == "svm_nu":
                values = values or [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
                results = benchmark.benchmark_oneclass_svm(split, nu_values=values)
            elif param == "mahalanobis_conformal":
                from ..novelty.config.strategies import MahalanobisConfig
                from ..novelty.strategies.mahalanobis import MahalanobisDistanceStrategy

                train_emb = benchmark.encode_texts(split.train_texts)
                val_emb = benchmark.encode_texts(split.val_texts)
                test_emb = benchmark.encode_texts(split.test_texts)
                val_true = prepare_binary_labels(split.val_labels, "__OOD__")
                test_true = prepare_binary_labels(split.test_labels, "__OOD__")

                results = []
                for method in ["split", "mondrian"]:
                    for alpha in [0.05, 0.1, 0.15, 0.2]:
                        try:
                            config = MahalanobisConfig(
                                calibration_mode="conformal",
                                calibration_method=method,  # type: ignore[arg-type]
                                calibration_alpha=alpha,
                            )
                            s = MahalanobisDistanceStrategy()
                            s.initialize(train_emb, split.train_labels, config)
                            _, val_m = s.detect(
                                split.val_texts, val_emb,
                                ["unknown"] * len(split.val_texts),
                                np.ones(len(split.val_texts)) * 0.5,
                            )
                            _, test_m = s.detect(
                                split.test_texts, test_emb,
                                ["unknown"] * len(split.test_texts),
                                np.ones(len(split.test_texts)) * 0.5,
                            )
                            results.append(benchmark._make_result(
                                "mahalanobis_conformal",
                                {"method": method, "alpha": alpha},
                                compute_ood_metrics(
                                    val_true,
                                    np.array([val_m[i].get("p_value", val_m[i].get("mahalanobis_novelty_score", 0.0)) for i in range(len(split.val_texts))]),
                                ),
                                compute_ood_metrics(
                                    test_true,
                                    np.array([test_m[i].get("p_value", test_m[i].get("mahalanobis_novelty_score", 0.0)) for i in range(len(split.test_texts))]),
                                ),
                            ))
                        except (ValueError, RuntimeError) as e:
                            logger.warning(f"mahalanobis_conformal {method}/{alpha} failed: {e}")
            elif param == "centroid_percentile":
                values = values or [90, 92, 95, 97, 99]
                results = benchmark.benchmark_setfit_centroid(split, percentile_values=values)
            elif param == "self_knowledge_dim":
                values = values or [32, 64, 128, 256, 512]
                results = benchmark.benchmark_self_knowledge(split, hidden_dims=[int(v) for v in values])
            elif param == "self_knowledge_epochs":
                values = values or [25, 50, 100, 150, 200]
                results = benchmark.benchmark_self_knowledge(split, epoch_values=[int(v) for v in values])
            else:
                logger.info(f"Sweep for param '{param}' not yet implemented")
                return 1

            if results:
                import pandas as pd
                records = [
                    {
                        "strategy": r.strategy,
                        "params": str(r.params),
                        "val_auroc": r.val_auroc,
                        "test_auroc": r.test_auroc,
                        "test_dr_1fp": r.test_dr_1fp,
                    }
                    for r in results
                ]
                df = pd.DataFrame(records)
                print(df.to_string(index=False))

                if args.output:
                    df.to_csv(args.output, index=False)
                    logger.info(f"Results saved to {args.output}")

        elif args.command == "bench-classifier":
            from .classifier_bench import main as classifier_main

            return classifier_main([
                "--mode", args.mode,
                "--num-entities", str(args.num_entities),
                "--num-samples", str(args.num_samples),
                "--num-epochs", str(args.num_epochs),
                *(["--models"] + args.models if args.models else []),
                *(["--output", str(args.output)] if args.output else []),
            ])

        elif args.command == "bench-novelty":
            from .novelty_bench import main as novelty_main

            argv_list = [
                "--depth", args.depth,
                "--datasets", *args.datasets,
                "--max-train", str(args.max_train),
                "--max-val", str(args.max_val),
                "--max-test", str(args.max_test),
                "--model", args.model,
            ]
            if args.models:
                argv_list.extend(["--models", *args.models])
            if args.output:
                argv_list.extend(["--output", args.output])
            return novelty_main(argv_list)

        elif args.command == "bench-async":
            from .async_bench import main as async_main

            argv_list = [
                "--multiplier", str(args.multiplier),
                "--concurrency", str(args.concurrency),
                "--section", args.section,
                "--model", args.model,
                "--modes", *args.modes,
                "--max-entities", str(args.max_entities),
                "--max-queries", str(args.max_queries),
            ]
            if args.output:
                argv_list.extend(["--output", str(args.output)])
            return async_main(argv_list)

        elif args.command == "bench-weights":
            from .weight_optimizer import main as weights_main

            argv_list = [
                "--dataset", args.dataset,
                "--model", args.model,
                "--trials", str(args.trials),
                "--max-train", str(args.max_train),
                "--max-val", str(args.max_val),
                "--max-test", str(args.max_test),
            ]
            if args.output:
                argv_list.extend(["--output", args.output])
            return weights_main(argv_list)

        elif args.command == "bench-ann":
            from .infra_bench import main_ann

            argv_list = [
                "--sizes", *[str(s) for s in args.sizes],
                "--dim", str(args.dim),
                "--k", str(args.k),
                "--queries", str(args.queries),
            ]
            if args.output:
                argv_list.extend(["--output", args.output])
            return main_ann(argv_list)

        elif args.command == "bench-reranker":
            from .infra_bench import main_reranker

            argv_list = [
                "--models", *args.models,
                "--queries", str(args.queries),
                "--candidates", str(args.candidates),
            ]
            if args.output:
                argv_list.extend(["--output", args.output])
            return main_reranker(argv_list)

        elif args.command == "render":
            from .visualization import render_main

            return render_main([str(args.input)])

        elif args.command == "plot":
            from .visualization import plot_main

            return plot_main([
                "--embedding-results", args.embedding_results,
                "--training-results", args.training_results,
                "--bert-results", args.bert_results,
                "--output-dir", args.output_dir,
            ])

        return 0

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logger.warning("Benchmarks command failed: %s", e)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
