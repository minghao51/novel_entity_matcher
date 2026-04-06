"""CLI for HuggingFace benchmarks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
        choices=["threshold", "k", "distance"],
        required=True,
        help="Parameter to sweep",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        help="Values to sweep",
    )
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
                logger.info(f"  {r.get('dataset')}: Accuracy={r.get('accuracy', 0):.4f}")

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
            logger.info(f"Sweep command: {args.task} {args.dataset} {args.param}")
            logger.info("Note: Detailed sweep functionality coming soon")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
