"""Visualization utilities for benchmark results.

Consolidates:
- render_benchmark_report.py (JSON -> markdown tables)
- visualize_benchmarks.py (JSON -> PNG charts)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _render_model_benchmark(rows: list[dict]) -> str:
    sections = []
    by_section: dict[str, list[dict]] = {}
    for row in rows:
        by_section.setdefault(row.get("section", "unknown"), []).append(row)

    for section, section_rows in by_section.items():
        sections.append(f"## `{section}`")
        summary_columns = [
            "track",
            "mode",
            "model",
            "status",
            "throughput_qps",
            "avg_latency",
            "accuracy_split",
            "base_accuracy",
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
        ]
        summary_columns = [
            col for col in summary_columns if any(col in row for row in section_rows)
        ]
        sections.append(_markdown_table(section_rows, summary_columns))

        perturbation_columns = [
            "model",
            "mode",
            "typo_accuracy",
            "remove_parenthetical_accuracy",
            "ampersand_expanded_accuracy",
            "first_clause_accuracy",
            "normalized_verbatim_accuracy",
        ]
        perturbation_columns = [
            col
            for col in perturbation_columns
            if any(col in row for row in section_rows)
        ]
        if len(perturbation_columns) > 2:
            sections.append("")
            sections.append("Perturbation breakdown:")
            sections.append(_markdown_table(section_rows, perturbation_columns))

    return "\n\n".join(sections)


def _render_speed_benchmark(rows: list[dict]) -> str:
    sections = []
    by_section: dict[str, list[dict]] = {}
    for row in rows:
        by_section.setdefault(row.get("section", "unknown"), []).append(row)

    for section, section_rows in by_section.items():
        sections.append(f"## `{section}`")
        columns = [
            "mode",
            "route",
            "construct_seconds",
            "fit_seconds",
            "cold_query_seconds",
            "match_seconds",
            "end_to_end_seconds",
            "qps",
            "avg_ms_per_query",
            "end_to_end_ms_per_query",
        ]
        columns = [col for col in columns if any(col in row for row in section_rows)]
        sections.append(_markdown_table(section_rows, columns))

    return "\n\n".join(sections)


def render_json_to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "No benchmark rows found."
    if "route" in rows[0]:
        return _render_speed_benchmark(rows)
    return _render_model_benchmark(rows)


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def extract_embedding_data(results: Any) -> Any:
    import pandas as pd

    rows = []
    entries = results if isinstance(results, list) else results.get("results", [])
    model_map = {
        "potion-base-8M": "potion-8m",
        "potion-base-32M": "potion-32m",
        "stable-static-embedding-fast-retrieval-mrl-en": "mrl-en",
        "all-MiniLM-L6-v2": "minilm",
        "bge-base-en-v1.5": "bge-base",
        "all-mpnet-base-v2": "mpnet",
    }
    for entry in entries:
        if entry.get("status") == "ok":
            section = entry.get("section", "unknown").split("/")[-1]
            model = model_map.get(entry.get("model", ""), entry.get("model", ""))
            rows.append(
                {
                    "section": section,
                    "model": model,
                    "throughput_qps": entry.get("throughput_qps", 0),
                    "avg_latency_ms": entry.get("avg_latency", 0) * 1000,
                    "p95_latency_ms": entry.get("p95_latency", 0) * 1000,
                    "accuracy": entry.get("accuracy", 0),
                }
            )
    return pd.DataFrame(rows)


def extract_training_data(results: Any) -> Any:
    import pandas as pd

    rows = []
    entries = results if isinstance(results, list) else results.get("results", [])
    model_map = {
        "all-MiniLM-L6-v2": "minilm",
        "bge-base-en-v1.5": "bge-base",
        "all-mpnet-base-v2": "mpnet",
    }
    for entry in entries:
        if entry.get("status") == "ok":
            section = entry.get("section", "unknown").split("/")[-1]
            model = model_map.get(entry.get("model", ""), entry.get("model", ""))
            rows.append(
                {
                    "section": section,
                    "mode": entry.get("mode", "unknown"),
                    "model": model,
                    "throughput_qps": entry.get("throughput_qps", 0),
                    "avg_latency_ms": entry.get("avg_latency", 0) * 1000,
                    "p95_latency_ms": entry.get("p95_latency", 0) * 1000,
                    "accuracy": entry.get("accuracy", 0),
                    "training_time_s": entry.get("training_time", 0),
                }
            )
    return pd.DataFrame(rows)


def extract_bert_data(results: Any) -> Any:
    import pandas as pd

    rows = []
    if isinstance(results, dict):
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and metrics.get("status") == "ok":
                short_name = model_name
                for alias, full in [
                    ("distilbert", "distilbert"),
                    ("TinyBERT", "tinybert"),
                    ("roberta", "roberta-base"),
                    ("deberta", "deberta-v3"),
                    ("bert-multilingual", "bert-multilingual"),
                ]:
                    if alias in model_name:
                        short_name = full
                        break
                rows.append(
                    {
                        "model": short_name,
                        "training_time_s": metrics.get("training_time", 0),
                        "memory_peak_mb": metrics.get("memory_peak_mb", 0),
                        "inference_time_s": metrics.get("inference_time", 0),
                        "throughput_samples_per_sec": metrics.get(
                            "throughput_samples_per_sec", 0
                        ),
                        "accuracy": metrics.get("accuracy", 0),
                    }
                )
    return pd.DataFrame(rows)


def generate_charts(
    embedding_results: Any,
    training_results: Any,
    bert_results: Any,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 10

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_df = extract_embedding_data(embedding_results)
    training_df = extract_training_data(training_results)
    bert_df = extract_bert_data(bert_results)

    if not embedding_df.empty:
        metric = "throughput_qps"
        _, ax = plt.subplots(figsize=(14, 6))
        pivot_df = embedding_df.pivot(index="section", columns="model", values=metric)
        pivot_df = pivot_df[pivot_df.mean().sort_values(ascending=False).index]
        pivot_df.plot(kind="bar", ax=ax, rot=45, width=0.8)
        ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
        ax.set_ylabel("Throughput (QPS)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Embedding Model Throughput Comparison", fontsize=14, fontweight="bold"
        )
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "embeddings_performance.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        _, axes = plt.subplots(1, 2, figsize=(16, 6))
        embedding_df.groupby("model")["avg_latency_ms"].mean().sort_values().plot(
            kind="barh", ax=axes[0], color="steelblue"
        )
        axes[0].set_xlabel("Average Latency (ms)", fontsize=12, fontweight="bold")
        axes[0].set_title("Average Inference Latency", fontsize=13, fontweight="bold")
        axes[0].grid(axis="x", alpha=0.3)

        embedding_df.groupby("model")["p95_latency_ms"].mean().sort_values().plot(
            kind="barh", ax=axes[1], color="coral"
        )
        axes[1].set_xlabel("P95 Latency (ms)", fontsize=12, fontweight="bold")
        axes[1].set_title("P95 Inference Latency", fontsize=13, fontweight="bold")
        axes[1].grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "embeddings_latency.png", dpi=150, bbox_inches="tight")
        plt.close()

    if not embedding_df.empty and not training_df.empty and not bert_df.empty:
        _fig, ax = plt.subplots(figsize=(16, 6))
        data = []
        for model in embedding_df["model"].unique():
            subset = embedding_df[embedding_df["model"] == model]
            data.append(
                {
                    "model": model,
                    "route": "zero-shot",
                    "accuracy": subset["accuracy"].mean() * 100,
                }
            )
        for mode in training_df["mode"].unique():
            for model in training_df[training_df["mode"] == mode]["model"].unique():
                subset = training_df[
                    (training_df["mode"] == mode) & (training_df["model"] == model)
                ]
                data.append(
                    {
                        "model": model,
                        "route": mode,
                        "accuracy": subset["accuracy"].mean() * 100,
                    }
                )
        data.extend(
            {
                "model": row.model,
                "route": "bert",
                "accuracy": row.accuracy * 100,
            }
            for row in bert_df.itertuples(index=False)
        )

        import pandas as pd

        plot_df = (
            pd.DataFrame(data)
            .groupby(["model", "route"], as_index=False)["accuracy"]
            .mean()
        )
        pivot = plot_df.pivot(index="model", columns="route", values="accuracy")
        sort_col = "zero-shot" if "zero-shot" in pivot.columns else pivot.columns[0]
        pivot = pivot.sort_values(by=sort_col, ascending=False)
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Accuracy Comparison Across Routes and Models",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(title="Route", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "accuracy_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    print(f"Charts saved to {output_dir}")


def render_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render benchmark JSON as markdown")
    parser.add_argument("input", type=Path, help="Benchmark JSON artifact")
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("Expected a JSON array of benchmark rows")

    print(render_json_to_markdown(payload))
    return 0


def plot_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark charts")
    parser.add_argument("--embedding-results", required=True)
    parser.add_argument("--training-results", required=True)
    parser.add_argument("--bert-results", required=True)
    parser.add_argument("--output-dir", default="docs/images/benchmarks")
    args = parser.parse_args(argv)

    generate_charts(
        load_json(args.embedding_results),
        load_json(args.training_results),
        load_json(args.bert_results),
        Path(args.output_dir),
    )
    return 0
