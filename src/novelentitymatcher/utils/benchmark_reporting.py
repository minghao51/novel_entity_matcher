"""Reporting and CLI helpers for semantic matcher benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


def save_benchmark_report(results: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(results.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
    elif path.suffix.lower() == ".csv":
        results.to_csv(path, index=False)
    else:
        raise ValueError("output_path must end with .json or .csv")

    return path


def format_benchmark_summary(results: pd.DataFrame) -> str:
    if results.empty:
        return "No benchmark results collected."

    lines = ["BENCHMARK RESULTS"]
    for track in results["track"].dropna().unique():
        track_subset = results[results["track"] == track]
        lines.append("")
        lines.append(f"[{track}]")
        for section in track_subset["section"].dropna().unique():
            section_subset = track_subset[track_subset["section"] == section]
            lines.append(f"<section: {section}>")
            if track == "ood_novelty":
                columns = [
                    col
                    for col in [
                        "model",
                        "selected_threshold",
                        "num_threshold_candidates",
                        "num_known_classes",
                        "num_heldout_classes",
                        "validation_novel_f1",
                        "validation_known_accuracy",
                        "validation_false_positive_novel_rate",
                        "novel_precision",
                        "novel_recall",
                        "novel_f1",
                        "known_accuracy",
                        "false_positive_novel_rate",
                        "overall_accuracy",
                        "artifact_path",
                    ]
                    if col in section_subset.columns
                ]
            elif (
                "mode" in section_subset.columns
                and section_subset["mode"].notna().any()
            ):
                columns = [
                    col
                    for col in [
                        "mode",
                        "model",
                        "status",
                        "throughput_qps",
                        "base_accuracy",
                        "train_accuracy",
                        "val_accuracy",
                        "test_accuracy",
                        "skip_reason",
                    ]
                    if col in section_subset.columns
                ]
            else:
                columns = [
                    col
                    for col in [
                        "model",
                        "backend",
                        "status",
                        "throughput_qps",
                        "base_accuracy",
                        "train_accuracy",
                        "val_accuracy",
                        "test_accuracy",
                        "speedup_vs_minilm",
                        "skip_reason",
                    ]
                    if col in section_subset.columns
                ]
            lines.append(section_subset[columns].to_string(index=False))
            lines.append("")
    return "\n".join(line for line in lines if line is not None).rstrip()


def print_benchmark_report(results: pd.DataFrame):
    logger.info(format_benchmark_summary(results))


def parse_benchmark_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark semantic matcher models")
    parser.add_argument(
        "--track",
        choices=("embeddings", "trained", "all"),
        default="all",
        help="Benchmark embeddings, trained modes, or both",
    )
    parser.add_argument(
        "--output",
        help="Optional output path (.json or .csv) for the combined results",
    )
    parser.add_argument(
        "--embedding-models",
        nargs="*",
        default=None,
        help="Optional subset of embedding aliases to benchmark",
    )
    parser.add_argument(
        "--training-models",
        nargs="*",
        default=None,
        help="Optional subset of training-compatible aliases to benchmark",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="Optional subset of processed-data sections such as languages/languages",
    )
    parser.add_argument(
        "--max-entities-per-section",
        type=int,
        default=200,
        help="Maximum entities loaded from each processed dataset section",
    )
    parser.add_argument(
        "--max-queries-per-section",
        type=int,
        default=50,
        help="Maximum benchmark queries generated per processed dataset section",
    )
    return parser.parse_args(argv)
