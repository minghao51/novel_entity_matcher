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

    from novelentitymatcher import Matcher, NovelEntityMatcher
    from novelentitymatcher.novelty import DetectionConfig
    from novelentitymatcher.novelty.config.strategies import (
        ConfidenceConfig,
        KNNConfig,
    )

    mo.md(
        """
    # Novel Entity Detection Dashboard

    Feed in text queries and see which ones are flagged as **novel** (not matching
    any known entity). Uses confidence + KNN distance strategies.
    """
    )
    return (
        ConfidenceConfig,
        DetectionConfig,
        KNNConfig,
        Matcher,
        NovelEntityMatcher,
        mo,
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
        "quantum mechanics",
        "wave function",
        "particle physics",
        "thermodynamics",
        "machine learning",
        "neural networks",
        "algorithm design",
        "data structures",
        "gene expression",
        "protein synthesis",
        "cell division",
        "DNA replication",
        "organic synthesis",
        "molecular bonding",
        "periodic table",
        "reaction kinetics",
        "linear algebra",
        "calculus",
        "topology",
        "number theory",
    ]

    training_labels = [
        "physics",
        "physics",
        "physics",
        "physics",
        "cs",
        "cs",
        "cs",
        "cs",
        "biology",
        "biology",
        "biology",
        "biology",
        "chemistry",
        "chemistry",
        "chemistry",
        "chemistry",
        "math",
        "math",
        "math",
        "math",
    ]

    @mo.persistent_cache
    def _fit_matcher():
        m = Matcher(entities=entities, model="minilm", threshold=0.6)
        m.fit(texts=training_texts, labels=training_labels)
        return m

    matcher = _fit_matcher()

    mo.md(
        f"**Trained matcher** with {len(training_texts)} samples across {len(entities)} classes."
    )
    return (matcher,)


@app.cell
def _(mo):
    query_area = mo.ui.text_area(
        value="quantum superposition\ndeep learning models\nCRISPR gene editing\ncomputational chemistry methods\nautonomous robotics\ndiscount coupon policy\nweather forecast today\ngibberish zxqv placeholder",
        label="Enter queries (one per line):",
        full_width=True,
    )
    query_area
    return (query_area,)


@app.cell
def _(
    ConfidenceConfig,
    DetectionConfig,
    KNNConfig,
    NovelEntityMatcher,
    matcher,
    mo,
    query_area,
):
    _queries = [q.strip() for q in query_area.value.strip().split("\n") if q.strip()]

    if not _queries:
        detection_output = mo.md("Add some queries above!")
    else:
        _nm = NovelEntityMatcher(
            matcher=matcher,
            detection_config=DetectionConfig(
                strategies=["confidence", "knn_distance"],
                confidence=ConfidenceConfig(threshold=0.45),
                knn_distance=KNNConfig(distance_threshold=0.45),
                combine_method="weighted",
            ),
        )

        _rows = []
        for _q in _queries:
            _r = _nm.match(_q)
            _rows.append(
                {
                    "query": _q,
                    "is_novel": _r.is_novel,
                    "predicted_id": _r.predicted_id if not _r.is_novel else "—",
                    "score": f"{_r.score:.2%}",
                    "signals": ", ".join(_r.signals.keys()) if _r.signals else "—",
                }
            )

        detection_output = mo.ui.table(_rows, label="Detection Results")

    detection_output
    return


@app.cell
def _(matcher, mo, plt, query_area):
    _queries = [q.strip() for q in query_area.value.strip().split("\n") if q.strip()]

    if not _queries:
        chart_output = mo.md("No queries to plot.")
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
        plt.tight_layout()
        chart_output = _fig

    chart_output
    return


if __name__ == "__main__":
    app.run()
