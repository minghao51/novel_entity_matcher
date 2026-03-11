#!/usr/bin/env python
"""Generate performance visualizations from benchmark results.

This script creates 6 comprehensive charts comparing embedding models,
training routes, and BERT classifiers based on benchmark results.

Usage:
    python scripts/visualize_benchmarks.py \
        --embedding-results artifacts/benchmarks/benchmarks_embedding_results.json \
        --training-results artifacts/benchmarks/benchmarks_training_results.json \
        --bert-results artifacts/benchmarks/benchmarks_bert_results.json \
        --output-dir docs/images/benchmarks/
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_embedding_data(results: Any) -> pd.DataFrame:
    """Extract embedding benchmark data into a DataFrame."""
    rows = []
    # Handle both list and dict formats
    entries = results if isinstance(results, list) else results.get("results", [])

    for entry in entries:
        if entry.get("status") == "ok":
            # Extract section name from path
            section = entry.get("section", "unknown")
            section = section.split("/")[-1] if "/" in section else section

            # Extract model name (shorten it)
            model = entry.get("model", "unknown")
            if "/" in model:
                model = model.split("/")[-1]
            # Map common model names to short names
            model_map = {
                "potion-base-8M": "potion-8m",
                "potion-base-32M": "potion-32m",
                "stable-static-embedding-fast-retrieval-mrl-en": "mrl-en",
                "all-MiniLM-L6-v2": "minilm",
                "bge-base-en-v1.5": "bge-base",
                "all-mpnet-base-v2": "mpnet",
            }
            model = model_map.get(model, model)

            rows.append({
                "section": section,
                "model": model,
                "throughput_qps": entry.get("throughput_qps", 0),
                "avg_latency_ms": entry.get("avg_latency", 0) * 1000,
                "p95_latency_ms": entry.get("p95_latency", 0) * 1000,
                "accuracy": entry.get("accuracy", 0),
            })
    return pd.DataFrame(rows)


def extract_training_data(results: Any) -> pd.DataFrame:
    """Extract training benchmark data into a DataFrame."""
    rows = []
    # Handle both list and dict formats
    entries = results if isinstance(results, list) else results.get("results", [])

    for entry in entries:
        if entry.get("status") == "ok":
            # Extract section name from path
            section = entry.get("section", "unknown")
            section = section.split("/")[-1] if "/" in section else section

            # Extract model name (shorten it)
            model = entry.get("model", "unknown")
            if "/" in model:
                model = model.split("/")[-1]
            # Map common model names to short names
            model_map = {
                "all-MiniLM-L6-v2": "minilm",
                "bge-base-en-v1.5": "bge-base",
                "all-mpnet-base-v2": "mpnet",
            }
            model = model_map.get(model, model)

            rows.append({
                "section": section,
                "mode": entry.get("mode", "unknown"),
                "model": model,
                "throughput_qps": entry.get("throughput_qps", 0),
                "avg_latency_ms": entry.get("avg_latency", 0) * 1000,
                "p95_latency_ms": entry.get("p95_latency", 0) * 1000,
                "accuracy": entry.get("accuracy", 0),
                "training_time_s": entry.get("training_time", 0),
            })
    return pd.DataFrame(rows)


def extract_bert_data(results: Any) -> pd.DataFrame:
    """Extract BERT benchmark data into a DataFrame."""
    rows = []
    # Handle dict format
    if isinstance(results, dict):
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and metrics.get("status") == "ok":
                # Shorten model names
                short_name = model_name
                if "distilbert" in model_name:
                    short_name = "distilbert"
                elif "TinyBERT" in model_name:
                    short_name = "tinybert"
                elif "roberta" in model_name:
                    short_name = "roberta-base"
                elif "deberta" in model_name:
                    short_name = "deberta-v3"
                elif "bert-multilingual" in model_name:
                    short_name = "bert-multilingual"

                rows.append({
                    "model": short_name,
                    "training_time_s": metrics.get("training_time", 0),
                    "memory_peak_mb": metrics.get("memory_peak_mb", 0),
                    "inference_time_s": metrics.get("inference_time", 0),
                    "throughput_samples_per_sec": metrics.get("throughput_samples_per_sec", 0),
                    "accuracy": metrics.get("accuracy", 0),
                })
    return pd.DataFrame(rows)


def plot_embedding_performance(
    df: pd.DataFrame,
    output_path: Path,
    metric: str = "throughput_qps",
    title: str = "Embedding Model Throughput Comparison",
):
    """Plot embedding model performance by dataset."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Pivot data for grouped bar chart
    pivot_df = df.pivot(index="section", columns="model", values=metric)

    # Sort by average performance
    avg_order = pivot_df.mean().sort_values(ascending=False).index
    pivot_df = pivot_df[avg_order]

    # Plot
    pivot_df.plot(kind="bar", ax=ax, rot=45, width=0.8)
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Throughput (QPS)" if metric == "throughput_qps" else "Latency (ms)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_comparison(df: pd.DataFrame, output_path: Path):
    """Plot latency comparison across models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Average latency
    avg_data = df.groupby("model")["avg_latency_ms"].mean().sort_values()
    avg_data.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_xlabel("Average Latency (ms)", fontsize=12, fontweight="bold")
    axes[0].set_title("Average Inference Latency", fontsize=13, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)

    # P95 latency
    p95_data = df.groupby("model")["p95_latency_ms"].mean().sort_values()
    p95_data.plot(kind="barh", ax=axes[1], color="coral")
    axes[1].set_xlabel("P95 Latency (ms)", fontsize=12, fontweight="bold")
    axes[1].set_title("P95 Inference Latency", fontsize=13, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_comparison(
    embedding_df: pd.DataFrame,
    training_df: pd.DataFrame,
    bert_df: pd.DataFrame,
    output_path: Path,
):
    """Plot accuracy comparison across all routes and models."""
    fig, ax = plt.subplots(figsize=(16, 6))

    # Combine data
    data = []

    # Embedding (zero-shot) - average across sections
    for model in embedding_df["model"].unique():
        subset = embedding_df[embedding_df["model"] == model]
        avg_acc = subset["accuracy"].mean() * 100
        data.append({
            "model": model,
            "route": "zero-shot",
            "accuracy": avg_acc,
        })

    # Training routes - average across sections and models
    for mode in training_df["mode"].unique():
        subset = training_df[training_df["mode"] == mode]
        # For each model, get average accuracy
        for model in subset["model"].unique():
            model_subset = subset[subset["model"] == model]
            avg_acc = model_subset["accuracy"].mean() * 100
            data.append({
                "model": model,
                "route": mode,
                "accuracy": avg_acc,
            })

    # BERT - single accuracy value per model
    for _, row in bert_df.iterrows():
        data.append({
            "model": row["model"],
            "route": "bert",
            "accuracy": row["accuracy"] * 100,
        })

    plot_df = pd.DataFrame(data)

    # For duplicate model-route combinations, take the mean
    plot_df = plot_df.groupby(["model", "route"], as_index=False)["accuracy"].mean()

    # Plot grouped bar chart
    pivot_df = plot_df.pivot(index="model", columns="route", values="accuracy")
    # Sort by zero-shot if available, otherwise by first column
    if "zero-shot" in pivot_df.columns:
        pivot_df = pivot_df.sort_values(by="zero-shot", ascending=False)
    else:
        pivot_df = pivot_df.sort_values(by=pivot_df.columns[0], ascending=False)

    pivot_df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Accuracy Comparison Across All Routes and Models",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Route", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(85, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_vs_accuracy(
    training_df: pd.DataFrame,
    bert_df: pd.DataFrame,
    output_path: Path,
):
    """Plot training time vs accuracy scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # SetFit routes
    setfit_df = training_df[
        training_df["mode"].isin(["head-only", "full"])
    ].copy()
    setfit_df["classifier_type"] = "SetFit"

    # BERT data (need to reshape)
    bert_rows = []
    for _, row in bert_df.iterrows():
        bert_rows.append({
            "model": row["model"].split("/")[-1],
            "training_time_s": row["training_time_s"],
            "accuracy": row["accuracy"] * 100,
            "classifier_type": "BERT",
        })
    bert_plot_df = pd.DataFrame(bert_rows)

    # Combine
    setfit_grouped = setfit_df.groupby("model").agg({
        "training_time_s": "mean",
        "accuracy": lambda x: (x * 100).mean(),
    }).reset_index()
    setfit_grouped["classifier_type"] = "SetFit"

    combined_df = pd.concat([setfit_grouped, bert_plot_df], ignore_index=True)

    # Plot scatter
    for ctype, color in [("SetFit", "steelblue"), ("BERT", "coral")]:
        subset = combined_df[combined_df["classifier_type"] == ctype]
        ax.scatter(
            subset["training_time_s"],
            subset["accuracy"],
            label=ctype,
            color=color,
            s=100,
            alpha=0.7,
            edgecolors="black",
        )

        # Add labels for top performers
        top_performers = subset.nlargest(3, "accuracy")
        for _, row in top_performers.iterrows():
            ax.annotate(
                row["model"],
                (row["training_time_s"], row["accuracy"]),
                fontsize=8,
                alpha=0.8,
            )

    ax.set_xlabel("Training Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Training Time vs Accuracy Tradeoff",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_static_vs_dynamic(
    embedding_df: pd.DataFrame,
    output_path: Path,
):
    """Plot static vs dynamic embedding performance comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Classify models
    static_models = ["potion-8m", "potion-32m", "mrl-en"]

    df = embedding_df.copy()
    df["embedding_type"] = df["model"].apply(
        lambda x: "Static" if any(m in x for m in static_models) else "Dynamic"
    )

    # Throughput comparison
    throughput_data = (
        df.groupby(["embedding_type", "model"])["throughput_qps"]
        .mean()
        .reset_index()
    )

    for etype, color in [("Static", "steelblue"), ("Dynamic", "coral")]:
        subset = throughput_data[throughput_data["embedding_type"] == etype]
        axes[0].barh(
            subset["model"],
            subset["throughput_qps"],
            label=etype,
            color=color,
            alpha=0.7,
        )

    axes[0].set_xlabel("Throughput (QPS)", fontsize=12, fontweight="bold")
    axes[0].set_title("Throughput: Static vs Dynamic", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.3)

    # Accuracy comparison
    accuracy_data = (
        df.groupby(["embedding_type", "model"])["accuracy"]
        .mean()
        .reset_index()
    )

    for etype, color in [("Static", "steelblue"), ("Dynamic", "coral")]:
        subset = accuracy_data[accuracy_data["embedding_type"] == etype]
        axes[1].barh(
            subset["model"],
            subset["accuracy"] * 100,
            label=etype,
            color=color,
            alpha=0.7,
        )

    axes[1].set_xlabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_title("Accuracy: Static vs Dynamic", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].set_xlim(90, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_selection_guide(
    embedding_df: pd.DataFrame,
    training_df: pd.DataFrame,
    bert_df: pd.DataFrame,
    output_path: Path,
):
    """Create a model selection decision tree visualization."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")

    # Calculate average metrics
    static_avg = embedding_df[
        embedding_df["model"].isin(["potion-8m", "potion-32m", "mrl-en"])
    ]["throughput_qps"].mean()

    dynamic_avg = embedding_df[
        ~embedding_df["model"].isin(["potion-8m", "potion-32m", "mrl-en"])
    ]["throughput_qps"].mean()

    setfit_acc = (training_df["accuracy"] * 100).mean()
    bert_acc = (bert_df["accuracy"] * 100).mean()

    # Create decision tree text
    tree_text = f"""
SEMANTIC MATCHER - MODEL SELECTION GUIDE

┌─────────────────────────────────────────────────────────────────┐
│                        START                                    │
│            Do you have labeled training data?                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │ YES                           │ NO
            ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ How many examples per     │   │   ZERO-SHOT ROUTE         │
│ entity do you have?       │   │                           │
└───────────┬───────────────┘   │ Static Embeddings:        │
            │                   │ • potion-8m ({static_avg:.0f} QPS)   │
    ┌───────┴────────┐          │ • potion-32m             │
    │                │          │ • mrl-en                 │
    ▼                ▼          │                           │
┌──────────┐    ┌──────────┐   │ Dynamic Embeddings:       │
│ 1-2      │    │ 3+       │   │ • minilm ({dynamic_avg:.0f} QPS)       │
│ examples │    │ examples │   │ • bge-base               │
└────┬─────┘    └────┬─────┘   │ • mpnet                  │
     │               │          └───────────────────────────┘
     ▼               ▼
┌──────────┐    ┌──────────┐
│HEAD-ONLY │    │   FULL   │
│  Route   │    │  Route   │
│          │    │          │
│SetFit +  │    │SetFit +  │
│minilm    │    │minilm    │
│          │    │          │
│Fast      │    │Better    │
│training  │    │accuracy  │
└─────┬────┘    └─────┬────┘
      │               │
      │      ┌────────┴────────┐
      │      │ Need maximum    │
      │      │ accuracy for    │
      │      │ nuanced text?   │
      │      └────────┬────────┘
      │               │
      │      ┌────────┴────────┐
      │      │ YES             │ NO
      │      ▼                 ▼
      │ ┌──────────┐    ┌──────────┐
      │ │   BERT   │    │  Use     │
      │ │  Route   │    │ FULL     │
      │ │          │    │ route    │
      │ │•distilbert│    │          │
      │ │•tinybert  │    └──────────┘
      │ │•roberta   │
      │ │          │
      │ │Best      │
      │ │accuracy  │
      │ │(~{bert_acc:.1f}%)  │
      │ └──────────┘
      │
RECOMMENDATIONS SUMMARY:

• Quick prototype → zero-shot with potion-8m
• Small labeled dataset (1-2 examples) → head-only
• Standard production use → full
• Accuracy-critical with rich data → bert
• Large-scale retrieval (>10K entities) → hybrid route

Key Metrics:
• Static embeddings: {static_avg:.0f} QPS (fastest)
• Dynamic embeddings: {dynamic_avg:.0f} QPS
• SetFit accuracy: {setfit_acc:.1f}%
• BERT accuracy: {bert_acc:.1f}% (highest)
"""

    ax.text(
        0.05,
        0.95,
        tree_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.title(
        "Model Selection Decision Tree",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark visualization charts"
    )
    parser.add_argument(
        "--embedding-results",
        required=True,
        help="Path to embedding benchmarks JSON file",
    )
    parser.add_argument(
        "--training-results",
        required=True,
        help="Path to training benchmarks JSON file",
    )
    parser.add_argument(
        "--bert-results",
        required=True,
        help="Path to BERT benchmarks JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/images/benchmarks",
        help="Output directory for charts",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading benchmark results...")
    embedding_results = load_json(args.embedding_results)
    training_results = load_json(args.training_results)
    bert_results = load_json(args.bert_results)

    # Extract data
    print("Extracting data...")
    embedding_df = extract_embedding_data(embedding_results)
    training_df = extract_training_data(training_results)
    bert_df = extract_bert_data(bert_results)

    print(f"\nEmbedding results: {len(embedding_df)} entries")
    print(f"Training results: {len(training_df)} entries")
    print(f"BERT results: {len(bert_df)} entries")

    # Generate charts
    print("\nGenerating charts...")

    # 1. Embedding Performance (Throughput)
    plot_embedding_performance(
        embedding_df,
        output_dir / "embeddings_performance.png",
        metric="throughput_qps",
        title="Embedding Model Throughput Comparison",
    )

    # 2. Latency Comparison
    plot_latency_comparison(
        embedding_df,
        output_dir / "embeddings_latency.png",
    )

    # 3. Accuracy Comparison
    plot_accuracy_comparison(
        embedding_df,
        training_df,
        bert_df,
        output_dir / "accuracy_comparison.png",
    )

    # 4. Training vs Accuracy
    plot_training_vs_accuracy(
        training_df,
        bert_df,
        output_dir / "training_vs_accuracy.png",
    )

    # 5. Static vs Dynamic
    plot_static_vs_dynamic(
        embedding_df,
        output_dir / "static_vs_dynamic.png",
    )

    # 6. Model Selection Guide
    plot_model_selection_guide(
        embedding_df,
        training_df,
        bert_df,
        output_dir / "model_selection_guide.png",
    )

    print(f"\nAll charts saved to: {output_dir}")


if __name__ == "__main__":
    main()
