# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    mo.md(
        """
        # Benchmarks & Production Guide

        Interactive reference for selecting the right classification mode, novelty
        strategies, and embedding models for your use case. All data from real benchmarks
        run on the Novel Entity Matcher codebase.
        """
    )
    return mo, pd, plt


@app.cell
def _(mo):
    mo.md("""
    ## 1. Classification Mode Comparison

    The Matcher auto-selects the best mode based on your training data size. For production
    workloads, **full SetFit** offers the best accuracy/time tradeoff.
    """)
    return


@app.cell
def _(mo, pd):
    _modes = pd.DataFrame([
        {"Mode": "zero-shot", "Accuracy": "73.3%", "Throughput": "~50K qps", "Train Time": "~3s", "Best For": "Prototyping, simple matching"},
        {"Mode": "head-only", "Accuracy": "54.7%", "Throughput": "~100 qps", "Train Time": "~5s", "Best For": "Quick accuracy boost"},
        {"Mode": "full SetFit", "Accuracy": "91.2%", "Throughput": "~100 qps", "Train Time": "~64s", "Best For": "Production, complex variations"},
        {"Mode": "BERT", "Accuracy": "88-98%", "Throughput": "~30 qps", "Train Time": "~5min", "Best For": "High-stakes, 100+ samples/entity"},
        {"Mode": "hybrid", "Accuracy": "90-95%", "Throughput": "~50 qps", "Train Time": "None (index)", "Best For": "10K+ entities, reranking"},
    ])
    mo.ui.table(_modes, label="Classification Mode Comparison (ag_news, 500 samples)")
    return


@app.cell
def _(pd, plt):
    import numpy as np

    _data = pd.DataFrame({
        "Mode": ["Zero-shot", "Head-only", "Full SetFit", "BERT", "Hybrid"],
        "Accuracy": [73.3, 54.7, 91.2, 93.0, 92.5],
        "Throughput (log)": [np.log10(50000), np.log10(100), np.log10(100), np.log10(30), np.log10(50)],
    })

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _colors = ["#95a5a6", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    _sizes = [200, 120, 180, 100, 140]
    _scatter = _ax.scatter(
        _data["Throughput (log)"], _data["Accuracy"],
        s=_sizes, c=_colors, alpha=0.8, edgecolors="black", linewidth=0.5,
    )
    for i, mode in enumerate(_data["Mode"]):
        _ax.annotate(mode, (_data["Throughput (log)"].iloc[i], _data["Accuracy"].iloc[i]),
                     textcoords="offset points", xytext=(10, 5), fontsize=10)
    _ax.set_xlabel("Throughput (log10 qps)")
    _ax.set_ylabel("Test Accuracy (%)")
    _ax.set_title("Accuracy vs Throughput by Mode")
    _ax.grid(alpha=0.3)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Novelty Detection Strategies

    All strategies ranked by AUROC on ag_news (20% OOD). Green = SetFit-based,
    blue = ensemble, gray = traditional.
    """)
    return


@app.cell
def _(mo, pd):
    strategies_df = pd.DataFrame([
        {"Strategy": "setfit_centroid", "Type": "SetFit-based", "AUROC": 0.907, "DR@1%": 0.147, "Weight": 0.45, "Best For": "Production default"},
        {"Strategy": "ensemble_adaptive", "Type": "Ensemble", "AUROC": 0.906, "DR@1%": 0.187, "Weight": "—", "Best For": "Best ensemble"},
        {"Strategy": "ensemble_weighted", "Type": "Ensemble", "AUROC": 0.905, "DR@1%": 0.124, "Weight": "—", "Best For": "Strong ensemble"},
        {"Strategy": "knn_distance (k=30)", "Type": "Traditional", "AUROC": 0.883, "DR@1%": 0.103, "Weight": 0.45, "Best For": "Scalable, production"},
        {"Strategy": "lof", "Type": "Traditional", "AUROC": 0.871, "DR@1%": 0.069, "Weight": 0.30, "Best For": "Varying density"},
        {"Strategy": "oneclass_svm", "Type": "Traditional", "AUROC": 0.834, "DR@1%": 0.143, "Weight": 0.10, "Best For": "Boundary detection"},
        {"Strategy": "mahalanobis", "Type": "Traditional", "AUROC": 0.691, "DR@1%": 0.029, "Weight": 0.35, "Best For": "Gaussian structure"},
        {"Strategy": "pattern", "Type": "Traditional", "AUROC": 0.630, "DR@1%": 0.002, "Weight": 0.20, "Best For": "Entity name matching"},
        {"Strategy": "self_knowledge", "Type": "ML-based", "AUROC": 0.563, "DR@1%": 0.011, "Weight": 0.15, "Best For": "Experimental"},
        {"Strategy": "confidence", "Type": "ML-based", "AUROC": 0.500, "DR@1%": 0.002, "Weight": 0.35, "Best For": "Baseline, always included"},
    ])

    min_auroc = mo.ui.slider(0.45, 0.95, value=0.50, label="Min AUROC filter", step=0.05)
    min_auroc
    return min_auroc, strategies_df


@app.cell
def _(min_auroc, mo, strategies_df):
    _filtered = strategies_df[strategies_df["AUROC"] >= min_auroc.value]
    mo.ui.table(_filtered, label=f"Strategies with AUROC >= {min_auroc.value:.2f}")
    return


@app.cell
def _(plt, strategies_df):
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _top = strategies_df.sort_values("AUROC", ascending=True)
    _colors = [
        "#2ecc71" if "setfit" in s.lower()
        else "#3498db" if "ensemble" in s.lower()
        else "#95a5a6"
        for s in _top["Strategy"]
    ]
    _ax.barh(range(len(_top)), _top["AUROC"], color=_colors)
    _ax.set_yticks(range(len(_top)))
    _ax.set_yticklabels(_top["Strategy"], fontsize=9)
    _ax.set_xlabel("Test AUROC")
    _ax.set_title("Novelty Strategies Ranked by AUROC")
    _ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    _ax.axvline(x=0.8, color="green", linestyle=":", alpha=0.5, label="Good (>0.8)")
    _ax.set_xlim(0, 1.0)
    _ax.legend(fontsize=9)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. KNN Parameter Sweep

    Optimal **k=20–30** provides the best detection rate. Higher k smooths the distance
    estimate without sacrificing AUROC.
    """)
    return


@app.cell
def _(pd, plt):
    _knn = pd.DataFrame([
        {"k": 3, "AUROC": 0.862, "DR@1%": 0.075},
        {"k": 5, "AUROC": 0.873, "DR@1%": 0.078},
        {"k": 10, "AUROC": 0.877, "DR@1%": 0.080},
        {"k": 20, "AUROC": 0.881, "DR@1%": 0.090},
        {"k": 25, "AUROC": 0.882, "DR@1%": 0.097},
        {"k": 30, "AUROC": 0.883, "DR@1%": 0.103},
    ])

    _fig, _ax1 = plt.subplots(figsize=(7, 4))
    _ax1.plot(_knn["k"], _knn["AUROC"], "o-", color="#3498db", linewidth=2, label="AUROC")
    _ax1.set_xlabel("k (neighbors)")
    _ax1.set_ylabel("AUROC", color="#3498db")
    _ax1.tick_params(axis="y", labelcolor="#3498db")
    _ax1.set_ylim(0.85, 0.90)

    _ax2 = _ax1.twinx()
    _ax2.bar(_knn["k"] - 0.4, _knn["DR@1%"], width=0.8, color="#e74c3c", alpha=0.6, label="DR@1%")
    _ax2.set_ylabel("DR@1%", color="#e74c3c")
    _ax2.tick_params(axis="y", labelcolor="#e74c3c")

    _lines1, _labels1 = _ax1.get_legend_handles_labels()
    _lines2, _labels2 = _ax2.get_legend_handles_labels()
    _ax1.legend(_lines1 + _lines2, _labels1 + _labels2, loc="lower right")
    _ax1.grid(alpha=0.3)
    _ax1.set_title("KNN Parameter Sweep")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. BERT vs SetFit

    SetFit dominates on few-shot tasks — higher accuracy, **17x less memory**, 1.68x faster inference.
    """)
    return


@app.cell
def _(mo, pd, plt):
    _compare = pd.DataFrame({
        "Metric": ["Training Time (s)", "Peak Memory (MB)", "Inference (/s)", "Accuracy (%)"],
        "SetFit": [41.26, 32.06, 1671, 100.0],
        "BERT": [13.20, 545.11, 998, 88.0],
    })
    mo.ui.table(_compare, label="SetFit vs BERT (10 entities, 50 samples/entity)")

    _metrics = ["Memory (MB)", "Throughput (/s)", "Accuracy (%)"]
    _sf = [32.06, 1671, 100.0]
    _bert = [545.11, 998, 88.0]
    _x = range(len(_metrics))
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _w = 0.35
    _ax.bar([p - _w / 2 for p in _x], _sf, _w, label="SetFit", color="#2ecc71")
    _ax.bar([p + _w / 2 for p in _x], _bert, _w, label="BERT", color="#e74c3c")
    _ax.set_xticks(list(_x))
    _ax.set_xticklabels(_metrics)
    _ax.legend()
    _ax.set_title("SetFit vs BERT Resource Comparison")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Speed Benchmarks

    Zero-shot achieves **~50K qps** via static embeddings. Trained modes run at **~100 q/s**
    due to model inference overhead.
    """)
    return


@app.cell
def _(mo, pd, plt):
    _speed = pd.DataFrame({
        "Route": ["sync.match.bulk", "async.match_batch", "sync.match.single"],
        "zero-shot": [49670, 34530, 27807],
        "head-only": [104, 84, 44],
        "full": [104, 90, 48],
    })
    mo.ui.table(_speed, label="Throughput by Mode and Route (products_mcc)")

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _x = range(len(_speed))
    _w = 0.25
    _ax.bar([p - _w for p in _x], _speed["zero-shot"], _w, label="Zero-shot", color="#3498db")
    _ax.bar([p for p in _x], _speed["head-only"], _w, label="Head-only", color="#e67e22")
    _ax.bar([p + _w for p in _x], _speed["full"], _w, label="Full", color="#2ecc71")
    _ax.set_xticks(list(_x))
    _ax.set_xticklabels(_speed["Route"])
    _ax.set_ylabel("Queries per second")
    _ax.set_yscale("log")
    _ax.legend()
    _ax.set_title("Throughput by Mode (log scale)")
    _ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Production Recommendations

    | Decision | Recommendation |
    |----------|---------------|
    | **Default mode** | `full` SetFit (91% acc, ~64s train) |
    | **Fast prototyping** | `zero-shot` (73% acc, 50K qps) |
    | **10K+ entities** | `hybrid` (blocking + reranking) |
    | **Novelty strategies** | `confidence + knn_distance + setfit_centroid` |
    | **KNN k** | 20–30 |
    | **Signal combining** | `weighted` (default) |
    | **Embedding model** | `potion-32m` (static, 10x faster than dynamic) |
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Run Locally

    ```bash
    uv sync --extra docs
    uv run marimo edit notebooks/04_benchmarks_production_guide.py
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()
