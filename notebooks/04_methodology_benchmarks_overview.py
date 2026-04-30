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
        # Methodology & Benchmarks Overview

        Novel Entity Matcher uses a **two-phase pipeline**:
        1. **Classification** — assigns a query to a known entity class
        2. **Novelty Detection** — determines if the query actually belongs to an unknown class
        """
    )
    return mo, pd, plt


@app.cell
def _(mo):
    mo.md("""
    ## 1. Classification Methods

    The Matcher auto-selects the best mode based on your data:
    - **Zero-shot** — no training needed, fast but lower accuracy
    - **Head-only** — quick training (~5s), weak accuracy
    - **Full SetFit** — best all-rounder (~64s, 91.2% acc)
    - **BERT** — highest ceiling (88-98%), needs 100+ samples/entity
    - **Hybrid** — blocking + retrieval + reranking, for 10k+ entities
    """)
    return


@app.cell
def _(mo, pd):
    _modes = pd.DataFrame(
        [
            {
                "Mode": "zero-shot",
                "Train Acc": "—",
                "Test Acc": "73.3%",
                "Train Time": "~3s",
                "Best For": "Prototyping, simple matching",
                "Pros": "Fast (~50K qps), no training needed",
                "Cons": "Lower accuracy, no novelty detection",
            },
            {
                "Mode": "head-only",
                "Train Acc": "53.1%",
                "Test Acc": "54.7%",
                "Train Time": "~5s",
                "Best For": "Quick accuracy boost",
                "Pros": "Fast training",
                "Cons": "Severe overfitting with small data",
            },
            {
                "Mode": "full SetFit",
                "Train Acc": "89.8%",
                "Test Acc": "91.2%",
                "Train Time": "~64s",
                "Best For": "Production, complex variations",
                "Pros": "Best accuracy/time tradeoff, setfit_centroid novelty",
                "Cons": "Requires training data",
            },
            {
                "Mode": "BERT",
                "Train Acc": "—",
                "Test Acc": "88-98%",
                "Train Time": "~5min",
                "Best For": "High-stakes, 100+ samples/entity",
                "Pros": "Highest accuracy ceiling",
                "Cons": "Needs lots of data, 17x more memory",
            },
            {
                "Mode": "hybrid",
                "Train Acc": "—",
                "Test Acc": "90-95%",
                "Train Time": "None (index)",
                "Best For": "10k+ entities, reranking",
                "Pros": "Handles large candidate sets",
                "Cons": "Slower, complex setup",
            },
        ]
    )
    mo.ui.table(_modes, label="Classification Mode Comparison (500 samples, ag_news)")
    return


@app.cell
def _(pd, plt):
    _data = pd.DataFrame(
        {
            "Mode": ["Zero-shot", "Head-only", "Full SetFit", "BERT", "Hybrid"],
            "Accuracy": [73.3, 54.7, 91.2, 93.0, 92.5],
        }
    )

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _colors = ["#95a5a6", "#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    _bars = _ax.bar(_data["Mode"], _data["Accuracy"], color=_colors, width=0.6)
    for _bar, _val in zip(_bars, _data["Accuracy"]):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height() + 1,
            f"{_val:.1f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    _ax.set_ylabel("Test Accuracy (%)")
    _ax.set_ylim(0, 105)
    _ax.axhline(
        y=73.3, color="gray", linestyle="--", alpha=0.5, label="Zero-shot baseline"
    )
    _ax.legend(fontsize=9)
    _ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Novelty Detection Strategies

    All 14 strategies ranked by performance on ag_news (20% OOD):
    """)
    return


@app.cell
def _(mo, pd):
    strategies_df = pd.DataFrame(
        [
            {
                "Strategy": "setfit_centroid",
                "Type": "SetFit-based",
                "Val AUROC": 0.915,
                "Test AUROC": 0.907,
                "DR@1%": 0.147,
                "Weight": 0.45,
                "Best For": "Production default, free text",
            },
            {
                "Strategy": "ensemble_adaptive",
                "Type": "Ensemble",
                "Val AUROC": 0.911,
                "Test AUROC": 0.906,
                "DR@1%": 0.187,
                "Weight": "—",
                "Best For": "Best ensemble",
            },
            {
                "Strategy": "ensemble_weighted",
                "Type": "Ensemble",
                "Val AUROC": 0.911,
                "Test AUROC": 0.905,
                "DR@1%": 0.124,
                "Weight": "—",
                "Best For": "Strong ensemble",
            },
            {
                "Strategy": "knn_distance (k=30)",
                "Type": "Traditional",
                "Val AUROC": 0.893,
                "Test AUROC": 0.883,
                "DR@1%": 0.103,
                "Weight": 0.45,
                "Best For": "Scalable, production",
            },
            {
                "Strategy": "lof",
                "Type": "Traditional",
                "Val AUROC": 0.872,
                "Test AUROC": 0.871,
                "DR@1%": 0.069,
                "Weight": 0.30,
                "Best For": "Varying density",
            },
            {
                "Strategy": "oneclass_svm",
                "Type": "Traditional",
                "Val AUROC": 0.836,
                "Test AUROC": 0.834,
                "DR@1%": 0.143,
                "Weight": 0.10,
                "Best For": "Boundary detection",
            },
            {
                "Strategy": "pattern",
                "Type": "Traditional",
                "Val AUROC": 0.682,
                "Test AUROC": 0.630,
                "DR@1%": 0.002,
                "Weight": 0.20,
                "Best For": "Entity name matching only",
            },
            {
                "Strategy": "mahalanobis",
                "Type": "Traditional",
                "Val AUROC": 0.696,
                "Test AUROC": 0.691,
                "DR@1%": 0.029,
                "Weight": 0.35,
                "Best For": "Gaussian structure",
            },
            {
                "Strategy": "self_knowledge",
                "Type": "ML-based",
                "Val AUROC": 0.588,
                "Test AUROC": 0.563,
                "DR@1%": 0.011,
                "Weight": 0.15,
                "Best For": "Experimental",
            },
            {
                "Strategy": "isolation_forest",
                "Type": "Traditional",
                "Val AUROC": 0.577,
                "Test AUROC": 0.572,
                "DR@1%": 0.013,
                "Weight": "—",
                "Best For": "Baseline",
            },
            {
                "Strategy": "prototypical",
                "Type": "ML-based",
                "Val AUROC": 0.507,
                "Test AUROC": 0.507,
                "DR@1%": 0.006,
                "Weight": 0.10,
                "Best For": "Few-shot (near random)",
            },
            {
                "Strategy": "confidence",
                "Type": "ML-based",
                "Val AUROC": 0.500,
                "Test AUROC": 0.500,
                "DR@1%": 0.002,
                "Weight": 0.35,
                "Best For": "Baseline, always included",
            },
            {
                "Strategy": "setfit (contrastive)",
                "Type": "SetFit-based",
                "Val AUROC": 0.483,
                "Test AUROC": 0.452,
                "DR@1%": 0.017,
                "Weight": 0.02,
                "Best For": "Below random — deprecated",
            },
            {
                "Strategy": "mahalanobis_conformal",
                "Type": "Conformal",
                "Val AUROC": 0.545,
                "Test AUROC": 0.520,
                "DR@1%": 0.034,
                "Weight": "—",
                "Best For": "Dataset-dependent",
            },
        ]
    )

    min_auroc_slider = mo.ui.slider(
        0.45, 0.95, value=0.50, label="Min AUROC filter", step=0.05
    )
    min_auroc_slider
    return min_auroc_slider, strategies_df


@app.cell
def _(min_auroc_slider, mo, strategies_df):
    _filtered = strategies_df[strategies_df["Test AUROC"] >= min_auroc_slider.value]
    mo.ui.table(
        _filtered, label=f"Strategies with Test AUROC ≥ {min_auroc_slider.value:.2f}"
    )
    return


@app.cell
def _(plt, strategies_df):
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _top = strategies_df.sort_values("Test AUROC", ascending=True)
    _colors = [
        "#2ecc71"
        if "setfit" in s.lower()
        else "#3498db"
        if "ensemble" in s.lower()
        else "#95a5a6"
        for s in _top["Strategy"]
    ]
    _ax.barh(range(len(_top)), _top["Test AUROC"], color=_colors)
    _ax.set_yticks(range(len(_top)))
    _ax.set_yticklabels(_top["Strategy"], fontsize=9)
    _ax.set_xlabel("Test AUROC")
    _ax.set_title("Novelty Strategies Ranked by AUROC")
    _ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    _ax.axvline(x=0.8, color="green", linestyle=":", alpha=0.5, label="Good")
    _ax.set_xlim(0, 1.0)
    _ax.legend(fontsize=9)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. KNN Parameter Sweep

    Optimal k=20-30 provides the best detection rate.
    """)
    return


@app.cell
def _(pd, plt):
    _knn_data = pd.DataFrame(
        [
            {"k": 3, "Val AUROC": 0.860, "Test AUROC": 0.862, "DR@1%": 0.075},
            {"k": 5, "Val AUROC": 0.870, "Test AUROC": 0.873, "DR@1%": 0.078},
            {"k": 10, "Val AUROC": 0.880, "Test AUROC": 0.877, "DR@1%": 0.080},
            {"k": 20, "Val AUROC": 0.889, "Test AUROC": 0.881, "DR@1%": 0.090},
            {"k": 25, "Val AUROC": 0.891, "Test AUROC": 0.882, "DR@1%": 0.097},
            {"k": 30, "Val AUROC": 0.893, "Test AUROC": 0.883, "DR@1%": 0.103},
        ]
    )

    _fig, _ax1 = plt.subplots(figsize=(7, 4))
    _ax1.plot(
        _knn_data["k"],
        _knn_data["Test AUROC"],
        "o-",
        color="#3498db",
        linewidth=2,
        label="Test AUROC",
    )
    _ax1.set_xlabel("k (neighbors)")
    _ax1.set_ylabel("AUROC", color="#3498db")
    _ax1.tick_params(axis="y", labelcolor="#3498db")
    _ax1.set_ylim(0.85, 0.90)

    _ax2 = _ax1.twinx()
    _ax2.bar(
        _knn_data["k"] - 0.4,
        _knn_data["DR@1%"],
        width=0.8,
        color="#e74c3c",
        alpha=0.6,
        label="DR@1%",
    )
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
    ## 4. BERT vs SetFit Comparison

    SetFit dominates on few-shot tasks — higher accuracy, 17x less memory, 1.68x faster inference.
    """)
    return


@app.cell
def _(mo, pd, plt):
    _compare = pd.DataFrame(
        {
            "Metric": [
                "Training Time (s)",
                "Peak Memory (MB)",
                "Inference Throughput (/s)",
                "Accuracy",
            ],
            "SetFit": [41.26, 32.06, 1671, 100.0],
            "BERT (distilbert)": [13.20, 545.11, 998, 88.0],
        }
    )

    mo.ui.table(_compare, label="SetFit vs BERT (10 entities, 50 samples/entity)")

    _metrics = ["Memory (MB)", "Throughput (/s)", "Accuracy (%)"]
    _setfit_vals = [32.06, 1671, 100.0]
    _bert_vals = [545.11, 998, 88.0]
    _x = range(len(_metrics))
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _w = 0.35
    _ax.bar([p - _w / 2 for p in _x], _setfit_vals, _w, label="SetFit", color="#2ecc71")
    _ax.bar([p + _w / 2 for p in _x], _bert_vals, _w, label="BERT", color="#e74c3c")
    _ax.set_xticks(list(_x))
    _ax.set_xticklabels(_metrics)
    _ax.legend()
    _ax.set_title("SetFit vs BERT")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Speed Benchmarks

    Zero-shot achieves **~50K qps** while trained modes run at **~30 q/s**.
    """)
    return


@app.cell
def _(mo, pd, plt):
    _speed = pd.DataFrame(
        {
            "Route": [
                "sync.match.bulk",
                "async.match_batch_async",
                "sync.match.single",
            ],
            "zero-shot (qps)": [49670, 34530, 27807],
            "head-only (qps)": [104, 84, 44],
            "full (qps)": [104, 90, 48],
        }
    )
    mo.ui.table(_speed, label="Throughput by Mode and Route (products_mcc)")

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _x = range(len(_speed))
    _w = 0.25
    _ax.bar(
        [p - _w for p in _x],
        _speed["zero-shot (qps)"],
        _w,
        label="Zero-shot",
        color="#3498db",
    )
    _ax.bar(
        [p for p in _x],
        _speed["head-only (qps)"],
        _w,
        label="Head-only",
        color="#e67e22",
    )
    _ax.bar(
        [p + _w for p in _x], _speed["full (qps)"], _w, label="Full", color="#2ecc71"
    )
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
    ## 6. Signal Combination Methods

    | Method | Formula | Use Case |
    |--------|---------|----------|
    | **weighted** | `score = Σ(wᵢ × sᵢ) / Σ(wᵢ)` | **Default** — best overall |
    | **union** | `novel = any(flags)` | High recall |
    | **intersection** | `novel = all(flags)` | High precision |
    | **voting** | `novel = count(flags) > n/2` | Balanced |
    | **meta_learner** | Logistic regression | Learned combination |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Recommended Production Config

    ```python
    strategies=["confidence", "knn_distance", "setfit_centroid"]
    combine_method="weighted"
    knn_distance=KNNConfig(k=20)
    weights=WeightConfig(setfit_centroid=0.45, knn=0.45, confidence=0.35)
    ```
    """)
    return


@app.cell
def _(mo, strategies_df):
    _production = strategies_df[
        strategies_df["Strategy"].isin(
            ["setfit_centroid", "knn_distance (k=30)", "confidence"]
        )
    ].copy()

    mo.ui.table(_production, label="Production Default Strategies")
    return


if __name__ == "__main__":
    app.run()
