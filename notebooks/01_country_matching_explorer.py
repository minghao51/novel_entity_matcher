# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "novel-entity-matcher",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from novelentitymatcher import Matcher

    mo.md(
        """
    # Zero-Shot Country Matching Explorer

    Type a country name (with typos, aliases, or alternate languages) and see how
    the Matcher resolves it to a canonical entity — no training required.
    """
    )
    return Matcher, mo, pd


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
    def _fit_matcher():
        m = Matcher(entities=entities, mode="zero-shot")
        m.fit()
        return m

    matcher = _fit_matcher()
    return (matcher,)


@app.cell
def _(mo):
    query_input = mo.ui.text(
        value="Deutchland",
        label="Enter a country name (try typos!):",
        full_width=True,
    )
    top_k_slider = mo.ui.slider(1, 5, value=3, label="Top-K results")
    mo.hstack([query_input, top_k_slider], justify="start")
    return query_input, top_k_slider


@app.cell
def _(matcher, mo, pd, query_input, top_k_slider):
    _query = query_input.value
    _k = top_k_slider.value

    _empty = not _query.strip()

    if _empty:
        table_output = mo.md("Type something above to search!")
    else:
        _results = matcher.match(_query, top_k=_k)
        if isinstance(_results, list):
            _df = pd.DataFrame(_results)
        else:
            _df = pd.DataFrame([_results])
        table_output = mo.ui.table(_df, label=f"Matches for '{_query}'")

    table_output
    return


@app.cell
def _(matcher, mo, query_input):
    _query = query_input.value
    if _query.strip():
        _result = matcher.match(_query)
        _entry = _result if isinstance(_result, dict) else _result
        _best_id = _entry.get("id", "N/A") if isinstance(_entry, dict) else "N/A"
        _best_score = _entry.get("score", 0) if isinstance(_entry, dict) else 0
        summary = mo.md(
            f"**Best match:** `{_best_id}` with confidence **{_best_score:.2%}**"
        )
    else:
        summary = mo.md("")

    summary
    return


if __name__ == "__main__":
    app.run()
