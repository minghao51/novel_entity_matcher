# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "novel-entity-matcher",
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

    from novelentitymatcher import Matcher

    mo.md(
        """
    # Entity Matching Explorer

    Learn how the **Matcher** resolves messy text — typos, aliases, foreign names —
    to canonical entity IDs using embedding similarity and optional few-shot training.

    **Part 1** — Zero-shot matching: no training needed, pure cosine similarity.
    **Part 2** — Training impact: see how few-shot learning boosts accuracy.
    """
    )
    return Matcher, mo, pd, plt


@app.cell
def _(Matcher, mo):
    entities = [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Bundesrepublik"]},
        {"id": "FR", "name": "France", "aliases": ["Frankreich", "La France"]},
        {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
        {"id": "JP", "name": "Japan", "aliases": ["Nihon", "Nippon"]},
        {"id": "CN", "name": "China", "aliases": ["Zhongguo"]},
        {"id": "BR", "name": "Brazil", "aliases": ["Brasil"]},
        {"id": "IN", "name": "India", "aliases": ["Bharat"]},
        {"id": "GB", "name": "United Kingdom", "aliases": ["UK", "Britain", "England"]},
        {"id": "AU", "name": "Australia", "aliases": ["Oz"]},
        {"id": "CA", "name": "Canada", "aliases": ["Canuck Land"]},
    ]

    @mo.persistent_cache
    def _fit_zero_shot():
        m = Matcher(entities=entities, mode="zero-shot")
        m.fit()
        return m

    zero_shot = _fit_zero_shot()
    mo.md(f"Loaded **{len(entities)}** country entities in zero-shot mode.")
    return entities, zero_shot


@app.cell
def _(mo):
    mo.md("## Part 1: Zero-Shot Matching\n\nType a country name below and see cosine similarity to every entity.")
    return


@app.cell
def _(mo):
    query_input = mo.ui.text(
        value="Deutchland",
        label="Enter a country name (try typos, aliases, foreign names):",
        full_width=True,
    )
    top_k_slider = mo.ui.slider(1, 10, value=5, label="Top-K results")
    mo.hstack([query_input, top_k_slider], justify="start")
    return query_input, top_k_slider


@app.cell
def _(mo, pd, query_input, top_k_slider, zero_shot):
    _q = query_input.value
    _k = top_k_slider.value

    if not _q.strip():
        match_table = mo.md("Type a query above to search!")
    else:
        _results = zero_shot.match(_q, top_k=_k)
        _batch = _results if isinstance(_results, list) else [_results]
        _df = pd.DataFrame(_batch)
        match_table = mo.ui.table(_df, label=f"Matches for '{_q}' (top {_k})")

    match_table
    return


@app.cell
def _(mo, plt, query_input, zero_shot):
    _q = query_input.value

    if not _q.strip():
        match_chart = mo.md("")
    else:
        _results = zero_shot.match(_q, top_k=10)
        _batch = _results if isinstance(_results, list) else [_results]
        _labels = [r.get("id", "?") for r in _batch]
        _scores = [r.get("score", 0) for r in _batch]
        _colors = ["#2ecc71" if s >= 0.7 else "#e74c3c" for s in _scores]

        _fig, _ax = plt.subplots(figsize=(7, 4))
        _ax.barh(range(len(_labels)), _scores, color=_colors)
        _ax.set_yticks(range(len(_labels)))
        _ax.set_yticklabels(_labels)
        _ax.set_xlabel("Cosine Similarity")
        _ax.set_title(f"Similarity Scores for '{_q}'")
        _ax.axvline(x=0.7, color="gray", linestyle="--", alpha=0.7, label="threshold (0.7)")
        _ax.legend()
        _ax.set_xlim(0, 1)
        _ax.invert_yaxis()
        plt.tight_layout()
        match_chart = _fig

    match_chart
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Part 2: Training Impact

    See how accuracy changes as you add labeled training samples.
    The Matcher auto-selects the best mode based on data size.

    Adjust the slider and watch the sweep chart update.
    """
    )
    return


@app.cell
def _(Matcher, entities, mo):
    training_pairs = [
        ("Deutschland", "DE"), ("Allemagne", "DE"),
        ("Frankreich", "FR"), ("France", "FR"),
        ("United States of America", "US"), ("Vereinigte Staaten", "US"),
        ("Nihon", "JP"), ("Japan", "JP"),
        ("Zhongguo", "CN"), ("Chine", "CN"),
        ("Brasilien", "BR"), ("Brasil", "BR"),
        ("Bharat", "IN"), ("India", "IN"),
        ("Großbritannien", "GB"), ("United Kingdom", "GB"),
        ("Australien", "AU"), ("Australia", "AU"),
        ("Kanada", "CA"), ("Canada", "CA"),
    ]

    test_queries = [
        ("Deutchland", "DE"), ("America", "US"), ("Frankreich", "FR"),
        ("Nihon", "JP"), ("UK", "GB"), ("Brasil", "BR"),
        ("Bharat", "IN"), ("Canuck Land", "CA"), ("Zhongguo", "CN"), ("Oz", "AU"),
    ]

    sample_slider = mo.ui.slider(
        0, len(training_pairs), value=0,
        label=f"Training samples (0 = zero-shot, max = {len(training_pairs)})",
        step=2,
    )
    sample_slider
    return test_queries, training_pairs, sample_slider


@app.cell
def _(Matcher, entities, mo, sample_slider, test_queries, training_pairs):
    _n = sample_slider.value

    @mo.persistent_cache
    def _evaluate_trained(n: int):
        texts = [t for t, _ in training_pairs[:n]]
        labels = [l for _, l in training_pairs[:n]]
        mode = "zero-shot" if n == 0 else None
        m = Matcher(entities=entities, mode=mode)
        if n > 0:
            m.fit(texts=texts, labels=labels)
        else:
            m.fit()
        correct = 0
        for q, expected in test_queries:
            r = m.match(q)
            entry = r if isinstance(r, dict) else r
            if isinstance(entry, dict) and entry.get("id") == expected:
                correct += 1
        return correct / len(test_queries)

    _acc = _evaluate_trained(_n)
    _mode_label = "zero-shot" if _n == 0 else "trained"
    accuracy_output = mo.md(
        f"**{_n} samples** ({_mode_label}) → accuracy = **{_acc:.0%}** on {len(test_queries)} test queries"
    )
    accuracy_output
    return


@app.cell
def _(Matcher, entities, mo, plt, test_queries, training_pairs):
    @mo.persistent_cache
    def _sweep_accuracy():
        results = {}
        for n in range(0, len(training_pairs) + 1, 2):
            texts = [t for t, _ in training_pairs[:n]]
            labels = [l for _, l in training_pairs[:n]]
            mode = "zero-shot" if n == 0 else None
            m = Matcher(entities=entities, mode=mode)
            if n > 0:
                m.fit(texts=texts, labels=labels)
            else:
                m.fit()
            correct = 0
            for q, expected in test_queries:
                r = m.match(q)
                entry = r if isinstance(r, dict) else r
                if isinstance(entry, dict) and entry.get("id") == expected:
                    correct += 1
            results[n] = correct / len(test_queries)
        return results

    sweep = _sweep_accuracy()

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _xs = list(sweep.keys())
    _ys = list(sweep.values())
    _ax.plot(_xs, _ys, "o-", color="#3498db", linewidth=2, markersize=6)
    _ax.fill_between(_xs, _ys, alpha=0.1, color="#3498db")
    _ax.set_xlabel("Training Samples")
    _ax.set_ylabel("Accuracy")
    _ax.set_title("Accuracy vs Training Size")
    _ax.set_ylim(0, 1.05)
    _ax.axhline(y=sweep[0], color="#e74c3c", linestyle="--", alpha=0.5,
                label=f"Zero-shot baseline ({sweep[0]:.0%})")
    _ax.grid(alpha=0.3)
    _ax.legend()
    _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Run Locally

    ```bash
    uv sync --extra docs
    uv run marimo edit notebooks/01_entity_matching_explorer.py
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()
