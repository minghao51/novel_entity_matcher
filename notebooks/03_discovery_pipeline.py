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
    )

    mo.md(
        """
    # Discovery Pipeline

    Walk through the full **5-stage discovery pipeline** that powers novel class detection:

    1. **Match** — classify queries against known entities
    2. **Detect** — flag inputs that don't belong to any known class
    3. **Cluster** — group novel samples by similarity
    4. **Evidence** — extract keywords and representative examples per cluster
    5. **Propose** — suggest new class names (LLM-powered, optional)

    This is the most powerful feature of Novel Entity Matcher — it doesn't just reject
    unknowns, it **discovers what they are**.
    """
    )
    return ConfidenceConfig, DetectionConfig, KNNConfig, Matcher, NovelEntityMatcher, mo, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
    ## Setup: Product Categories

    We train the matcher on 5 known product categories, then feed it a mix of known
    products and emerging categories it has never seen.
    """
    )
    return


@app.cell
def _(Matcher, mo):
    entities = [
        {"id": "electronics", "name": "Electronics"},
        {"id": "clothing", "name": "Clothing"},
        {"id": "food", "name": "Food & Beverage"},
        {"id": "books", "name": "Books & Media"},
        {"id": "sports", "name": "Sports Equipment"},
    ]

    training_texts = [
        "smartphone case", "laptop charger", "wireless headphones", "tablet screen protector", "USB-C cable",
        "winter jacket", "running shoes", "cotton t-shirt", "wool sweater", "leather belt",
        "organic coffee beans", "dark chocolate bar", "sparkling water", "olive oil", "protein bars",
        "science fiction novel", "cookbook recipes", "children's picture book", "history biography", "self-help guide",
        "yoga mat", "tennis racket", "swimming goggles", "dumbbell set", "hiking backpack",
    ]

    training_labels = [
        "electronics", "electronics", "electronics", "electronics", "electronics",
        "clothing", "clothing", "clothing", "clothing", "clothing",
        "food", "food", "food", "food", "food",
        "books", "books", "books", "books", "books",
        "sports", "sports", "sports", "sports", "sports",
    ]

    @mo.persistent_cache
    def _fit():
        m = Matcher(entities=entities, model="minilm", threshold=0.6)
        m.fit(texts=training_texts, labels=training_labels)
        return m

    matcher = _fit()
    mo.md(f"Trained on **{len(training_texts)}** samples across **{len(entities)}** product categories.")
    return matcher


@app.cell
def _(ConfidenceConfig, DetectionConfig, KNNConfig):
    def _build_detection_config():
        return DetectionConfig(
            strategies=["confidence", "knn_distance"],
            confidence=ConfidenceConfig(threshold=0.45),
            knn_distance=KNNConfig(k=20, distance_threshold=0.45),
            combine_method="weighted",
        )

    return (_build_detection_config,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Queries: Known + Novel Mix

        The first 5 queries belong to known categories. The rest are from **emerging
        categories** the matcher has never seen: cryptocurrency hardware, smart home,
        electric vehicles, streaming services, IoT devices.
        """
    )
    return


@app.cell
def _(mo):
    query_area = mo.ui.text_area(
        value=(
            "bluetooth speaker\n"
            "denim jeans\n"
            "green tea\n"
            "mystery thriller\n"
            "basketball hoop\n"
            "bitcoin hardware wallet\n"
            "smart thermostat\n"
            "electric scooter\n"
            "music streaming subscription\n"
            "robot vacuum cleaner\n"
            "3D printer filament\n"
            "drone camera\n"
            "solar panel kit\n"
            "meditation app\n"
            "board game expansion"
        ),
        label="Queries (first 5 = known, rest = novel):",
        full_width=True,
    )
    query_area
    return (query_area,)


@app.cell
def _(mo):
    mo.md("## Stage 1–2: Match + Detect")
    return


@app.cell
def _(
    NovelEntityMatcher,
    _build_detection_config,
    matcher,
    mo,
    pd,
    query_area,
):
    _queries = [q.strip() for q in query_area.value.strip().split("\n") if q.strip()]

    if not _queries:
        match_detect_table = mo.md("Add queries above.")
    else:
        _nm = NovelEntityMatcher(
            matcher=matcher,
            detection_config=_build_detection_config(),
        )

        _rows = []
        for _q in _queries:
            _r = _nm.match(_q)
            _rows.append({
                "query": _q,
                "is_novel": _r.is_novel,
                "matched": _r.predicted_id if not _r.is_novel else "—",
                "confidence": f"{_r.score:.2%}",
                "novel_score": f"{_r.novel_score:.3f}" if _r.novel_score else "—",
            })

        match_detect_table = mo.ui.table(_rows, label="Match + Detection Results")

    match_detect_table
    return


@app.cell
def _(mo):
    mo.md("## Stage 3–5: Full Discovery Pipeline")
    return


@app.cell
def _(
    NovelEntityMatcher,
    _build_detection_config,
    matcher,
    mo,
    pd,
    query_area,
):
    _queries = [q.strip() for q in query_area.value.strip().split("\n") if q.strip()]

    if not _queries:
        discovery_output = mo.md("Add queries above to run discovery.")
    else:
        _nm = NovelEntityMatcher(
            matcher=matcher,
            detection_config=_build_detection_config(),
        )

        @mo.persistent_cache
        def _discover(query_batch: tuple[str, ...]):
            return _nm.discover_novel_classes(
                list(query_batch), run_llm_proposal=False, return_metadata=True
            )

        report = _discover(tuple(_queries))
        _novel = report.novel_sample_report.novel_samples

        _rows = []
        for s in _novel:
            _rows.append({
                "text": s.text,
                "predicted_class": s.predicted_class,
                "confidence": f"{s.confidence:.2%}",
                "novelty_score": f"{s.novelty_score:.3f}" if s.novelty_score else "—",
                "cluster_id": s.cluster_id if s.cluster_id is not None else "unclustered",
            })

        discovery_output = mo.vstack([
            mo.md(f"### Novel Samples Detected: **{len(_novel)}** / {len(_queries)} queries"),
            mo.ui.table(_rows, label="Novel Sample Details"),
        ])

    discovery_output
    return


@app.cell
def _(mo):
    mo.md("### Clustering Results")
    return


@app.cell
def _(
    NovelEntityMatcher,
    _build_detection_config,
    matcher,
    mo,
    plt,
    query_area,
):
    _queries = [q.strip() for q in query_area.value.strip().split("\n") if q.strip()]

    if not _queries:
        cluster_chart = mo.md("Add queries above.")
    else:
        _nm = NovelEntityMatcher(
            matcher=matcher,
            detection_config=_build_detection_config(),
        )

        @mo.persistent_cache
        def _discover2(query_batch: tuple[str, ...]):
            return _nm.discover_novel_classes(
                list(query_batch), run_llm_proposal=False, return_metadata=True
            )

        _report = _discover2(tuple(_queries))
        _clusters = _report.discovery_clusters

        if not _clusters:
            cluster_chart = mo.md("No clusters found (novel samples may be too diverse or too few).")
        else:
            _fig, _ax = plt.subplots(figsize=(8, 4))
            _ids = [f"Cluster {c.cluster_id}" for c in _clusters]
            _sizes = [c.sample_count for c in _clusters]
            _colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
            _ax.barh(range(len(_ids)), _sizes, color=_colors[: len(_ids)])
            _ax.set_yticks(range(len(_ids)))
            _ax.set_yticklabels(_ids)
            _ax.set_xlabel("Sample Count")
            _ax.set_title("Novel Sample Clusters")
            for i, (s, c) in enumerate(zip(_sizes, _ids)):
                _ax.text(s + 0.1, i, str(s), va="center", fontweight="bold")
            plt.tight_layout()
            cluster_chart = _fig

    cluster_chart
    return


@app.cell
def _(mo):
    mo.md("### Cluster Evidence")
    return


@app.cell
def _(
    NovelEntityMatcher,
    _build_detection_config,
    matcher,
    mo,
    query_area,
):
    _queries = [q.strip() for q in query_area.value.strip().split("\n") if q.strip()]

    if not _queries:
        evidence_output = mo.md("Add queries above.")
    else:
        _nm = NovelEntityMatcher(
            matcher=matcher,
            detection_config=_build_detection_config(),
        )

        @mo.persistent_cache
        def _discover3(query_batch: tuple[str, ...]):
            return _nm.discover_novel_classes(
                list(query_batch), run_llm_proposal=False, return_metadata=True
            )

        _report = _discover3(tuple(_queries))
        _clusters = _report.discovery_clusters

        if not _clusters:
            evidence_output = mo.md("No clusters to show evidence for.")
        else:
            _rows = []
            for c in _clusters:
                evidence = c.evidence
                _rows.append({
                    "cluster": f"Cluster {c.cluster_id}",
                    "size": c.sample_count,
                    "keywords": ", ".join((evidence.keywords if evidence else c.keywords)[:8]),
                    "examples": "; ".join((evidence.representative_examples if evidence else c.example_texts)[:3]),
                })
            evidence_output = mo.ui.table(_rows, label="Cluster Evidence")

    evidence_output
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Run Locally

    ```bash
    uv sync --extra docs
    uv run marimo edit notebooks/03_discovery_pipeline.py
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()
