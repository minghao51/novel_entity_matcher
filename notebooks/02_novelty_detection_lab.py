# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "novel-entity-matcher[novelty]",
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

    from novelentitymatcher import Matcher, NovelEntityMatcher
    from novelentitymatcher.novelty import DetectionConfig
    from novelentitymatcher.novelty.config.strategies import (
        ConfidenceConfig,
        KNNConfig,
        LOFConfig,
        MahalanobisConfig,
    )

    mo.md(
        """
    # Novelty Detection Lab

    Explore how novelty detection flags inputs that **don't belong** to any known
    entity class. Compare strategies side-by-side, tune thresholds, and see score
    distributions for known vs novel inputs.

    **Key concept:** A good matching system doesn't just classify — it also knows
    when it doesn't know.
    """
    )
    return (
        ConfidenceConfig,
        DetectionConfig,
        KNNConfig,
        LOFConfig,
        MahalanobisConfig,
        Matcher,
        NovelEntityMatcher,
        mo,
        pd,
        plt,
    )


@app.cell
def _(Matcher, mo):
    entities = [
        {"id": "physics", "name": "Physics"},
        {"id": "cs", "name": "Computer Science"},
        {"id": "biology", "name": "Biology"},
        {"id": "chemistry", "name": "Chemistry"},
        {"id": "math", "name": "Mathematics"},
    ]

    training_texts = [
        "quantum mechanics", "wave function", "particle physics", "thermodynamics",
        "machine learning", "neural networks", "algorithm design", "data structures",
        "gene expression", "protein synthesis", "cell division", "DNA replication",
        "organic synthesis", "molecular bonding", "periodic table", "reaction kinetics",
        "linear algebra", "calculus", "topology", "number theory",
    ]

    training_labels = [
        "physics", "physics", "physics", "physics",
        "cs", "cs", "cs", "cs",
        "biology", "biology", "biology", "biology",
        "chemistry", "chemistry", "chemistry", "chemistry",
        "math", "math", "math", "math",
    ]

    @mo.persistent_cache
    def _fit():
        m = Matcher(entities=entities, model="minilm", threshold=0.6)
        m.fit(texts=training_texts, labels=training_labels)
        return m

    matcher = _fit()
    mo.md(f"Trained matcher on **{len(training_texts)}** samples across **{len(entities)}** classes.")
    return matcher, training_texts, training_labels, entities


@app.cell
def _(mo):
    mo.md("## Test Queries\n\nQueries marked **known** belong to trained classes. Queries marked **novel** are from unrelated domains.")
    return


@app.cell
def _(mo):
    query_area = mo.ui.text_area(
        value=(
            "quantum superposition\n"
            "deep learning models\n"
            "CRISPR gene editing\n"
            "computational chemistry\n"
            "differential equations\n"
            "discount coupon policy\n"
            "weather forecast today\n"
            "gibberish zxqv placeholder\n"
            "sports scores nfl\n"
            "recipe chocolate cake"
        ),
        label="Enter queries (one per line). Last 4 are from unrelated domains:",
        full_width=True,
    )
    query_area
    return (query_area,)


@app.cell
def _(mo):
    mo.md("## Strategy Explorer")
    return


@app.cell
def _(mo):
    strategy_dropdown = mo.ui.dropdown(
        options={
            "Confidence Threshold": "confidence",
            "KNN Distance (k=20)": "knn_distance",
            "Mahalanobis Distance": "mahalanobis",
            "Local Outlier Factor": "lof",
        },
        value="KNN Distance (k=20)",
        label="Select a novelty strategy:",
    )
    threshold_slider = mo.ui.slider(
        0.1, 0.9, value=0.5, step=0.05, label="Score threshold"
    )
    mo.vstack([strategy_dropdown, threshold_slider])
    return strategy_dropdown, threshold_slider


@app.cell
def _(
    ConfidenceConfig,
    DetectionConfig,
    KNNConfig,
    LOFConfig,
    MahalanobisConfig,
    NovelEntityMatcher,
    matcher,
    mo,
    pd,
    query_area,
    strategy_dropdown,
    threshold_slider,
):
    def _build_detection_config(strategy: str, threshold: float):
        configs = {
            "confidence": DetectionConfig(
                strategies=["confidence"],
                confidence=ConfidenceConfig(threshold=threshold),
            ),
            "knn_distance": DetectionConfig(
                strategies=["confidence", "knn_distance"],
                confidence=ConfidenceConfig(threshold=threshold),
                knn_distance=KNNConfig(k=20, distance_threshold=threshold),
                combine_method="weighted",
            ),
            "mahalanobis": DetectionConfig(
                strategies=["confidence", "mahalanobis"],
                confidence=ConfidenceConfig(threshold=threshold),
                mahalanobis=MahalanobisConfig(),
                combine_method="weighted",
            ),
            "lof": DetectionConfig(
                strategies=["confidence", "lof"],
                confidence=ConfidenceConfig(threshold=threshold),
                lof=LOFConfig(),
                combine_method="weighted",
            ),
        }
        return configs[strategy]

    _queries = [q.strip() for q in query_area.value.strip().split("\n") if q.strip()]
    _strategy = strategy_dropdown.value

    if not _queries or not _strategy:
        detection_table = mo.md("Add queries and select a strategy above.")
    else:
        _nm = NovelEntityMatcher(
            matcher=matcher,
            detection_config=_build_detection_config(_strategy, threshold_slider.value),
        )

        _rows = []
        for _q in _queries:
            _r = _nm.match(_q)
            _rows.append({
                "query": _q,
                "is_novel": _r.is_novel,
                "predicted": _r.predicted_id if not _r.is_novel else "—",
                "confidence": f"{_r.score:.2%}",
                "novel_score": f"{_r.novel_score:.3f}" if _r.novel_score else "—",
                "signals": ", ".join(_r.signals.keys()) if _r.signals else "—",
            })

        detection_table = mo.ui.table(_rows, label=f"Detection Results — {_strategy}")

    detection_table
    return


@app.cell
def _(matcher, mo, plt, query_area):
    _queries = [q.strip() for q in query_area.value.strip().split("\n") if q.strip()]

    if not _queries:
        score_chart = mo.md("No queries to plot.")
    else:
        _scores = []
        for _q in _queries:
            _r = matcher.match(_q)
            _entry = _r if isinstance(_r, dict) else _r
            _score = _entry.get("score", 0) if isinstance(_entry, dict) else 0
            _scores.append(_score)

        _fig, _ax = plt.subplots(figsize=(8, 4))
        _colors = ["#e74c3c" if s < 0.5 else "#2ecc71" for s in _scores]
        _ax.barh(range(len(_queries)), _scores, color=_colors)
        _ax.set_yticks(range(len(_queries)))
        _ax.set_yticklabels(_queries, fontsize=9)
        _ax.set_xlabel("Match Confidence")
        _ax.set_title("Confidence Scores (red = likely novel)")
        _ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="threshold")
        _ax.legend()
        _ax.set_xlim(0, 1)
        plt.tight_layout()
        score_chart = _fig

    score_chart
    return


@app.cell
def _(mo):
    mo.md("## Multi-Strategy Comparison\n\nAll strategies evaluated on the same queries. Higher AUROC = better at separating known from novel.")
    return


@app.cell
def _(
    ConfidenceConfig,
    DetectionConfig,
    KNNConfig,
    LOFConfig,
    MahalanobisConfig,
    NovelEntityMatcher,
    matcher,
    mo,
    plt,
    query_area,
):
    _queries = [q.strip() for q in query_area.value.strip().split("\n") if q.strip()]

    _known_queries = _queries[:6]
    _novel_queries = _queries[6:]
    _all = _known_queries + _novel_queries
    _true_labels = [0] * len(_known_queries) + [1] * len(_novel_queries)

    if len(_all) < 4 or not _novel_queries:
        strategy_chart = mo.md("Need at least 4 queries with some novel ones for comparison.")
    else:

        @mo.persistent_cache
        def _compare_strategies(query_batch: tuple[str, ...]):
            strategy_configs = {
                "confidence": DetectionConfig(
                    strategies=["confidence"],
                    confidence=ConfidenceConfig(threshold=0.5),
                ),
                "knn_distance": DetectionConfig(
                    strategies=["confidence", "knn_distance"],
                    confidence=ConfidenceConfig(threshold=0.5),
                    knn_distance=KNNConfig(k=20, distance_threshold=0.5),
                    combine_method="weighted",
                ),
                "mahalanobis": DetectionConfig(
                    strategies=["confidence", "mahalanobis"],
                    confidence=ConfidenceConfig(threshold=0.5),
                    mahalanobis=MahalanobisConfig(),
                    combine_method="weighted",
                ),
                "lof": DetectionConfig(
                    strategies=["confidence", "lof"],
                    confidence=ConfidenceConfig(threshold=0.5),
                    lof=LOFConfig(),
                    combine_method="weighted",
                ),
            }

            results = {}
            for name, cfg in strategy_configs.items():
                nm = NovelEntityMatcher(matcher=matcher, detection_config=cfg)
                scores = []
                for q in query_batch:
                    r = nm.match(q)
                    scores.append(r.novel_score if r.novel_score else (1.0 - r.score))

                correct_flags = sum(
                    1 for s, t in zip(scores, _true_labels)
                    if (s > 0.5) == (t == 1)
                )
                results[name] = {
                    "accuracy": correct_flags / len(query_batch),
                    "mean_known": sum(s for s, t in zip(scores, _true_labels) if t == 0) / max(len(_known_queries), 1),
                    "mean_novel": sum(s for s, t in zip(scores, _true_labels) if t == 1) / max(len(_novel_queries), 1),
                }
            return results

        _comp = _compare_strategies(tuple(_all))

        _fig, _ax = plt.subplots(figsize=(8, 4))
        _names = list(_comp.keys())
        _known_means = [_comp[n]["mean_known"] for n in _names]
        _novel_means = [_comp[n]["mean_novel"] for n in _names]

        _x = range(len(_names))
        _w = 0.35
        _ax.bar([p - _w / 2 for p in _x], _known_means, _w, label="Known queries", color="#2ecc71")
        _ax.bar([p + _w / 2 for p in _x], _novel_means, _w, label="Novel queries", color="#e74c3c")
        _ax.set_xticks(list(_x))
        _ax.set_xticklabels(_names, fontsize=10)
        _ax.set_ylabel("Mean Novel Score")
        _ax.set_title("Strategy Comparison: Known vs Novel Score Separation")
        _ax.legend()
        _ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        strategy_chart = _fig

    strategy_chart
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Run Locally

    ```bash
    uv sync --extra docs
    uv run marimo edit notebooks/02_novelty_detection_lab.py
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()
