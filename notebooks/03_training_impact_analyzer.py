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

    from novelentitymatcher import Matcher

    mo.md(
        """
    # Training Impact Analyzer

    Compare **zero-shot** vs **trained** matching side-by-side. Adjust the number
    of training samples and see how accuracy changes for known and tricky inputs.
    """
    )
    return Matcher, mo, plt


@app.cell
def _(Matcher):
    entities = [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
        {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
        {"id": "JP", "name": "Japan", "aliases": ["Nippon"]},
        {"id": "CN", "name": "China", "aliases": ["Zhongguo"]},
    ]

    full_training = [
        {"text": "Germany", "label": "DE"},
        {"text": "Deutschland", "label": "DE"},
        {"text": "Deutchland", "label": "DE"},
        {"text": "GER", "label": "DE"},
        {"text": "France", "label": "FR"},
        {"text": "French Republic", "label": "FR"},
        {"text": "La France", "label": "FR"},
        {"text": "FRA", "label": "FR"},
        {"text": "United States", "label": "US"},
        {"text": "USA", "label": "US"},
        {"text": "America", "label": "US"},
        {"text": "U.S.A.", "label": "US"},
        {"text": "Japan", "label": "JP"},
        {"text": "Nippon", "label": "JP"},
        {"text": "Nihon", "label": "JP"},
        {"text": "China", "label": "CN"},
        {"text": "Zhongguo", "label": "CN"},
        {"text": "PRC", "label": "CN"},
    ]

    test_queries = [
        ("Deutchland", "DE"),
        ("America", "US"),
        ("Frankreich", "FR"),
        ("Nihon", "JP"),
        ("PRC", "CN"),
        ("Bundesrepublik", "DE"),
        ("U.S. of A", "US"),
        ("La Republique", "FR"),
    ]

    zero_matcher = Matcher(entities=entities, mode="zero-shot")
    zero_matcher.fit()
    return entities, full_training, test_queries, zero_matcher


@app.cell
def _(mo):
    n_samples = mo.ui.slider(0, 18, value=6, label="Training samples (0 = zero-shot)")
    n_samples
    return (n_samples,)


@app.cell
def _(
    Matcher,
    entities,
    full_training,
    mo,
    n_samples,
    test_queries,
    zero_matcher,
):
    _n = n_samples.value

    if _n > 0:
        _training_subset = full_training[:_n]
        trained_matcher = Matcher(entities=entities, verbose=False)
        trained_matcher.fit(training_data=_training_subset, num_epochs=1)
        _trained_mode = "trained"
    else:
        trained_matcher = zero_matcher
        _trained_mode = "same (zero-shot)"

    _rows = []
    for _query, _expected in test_queries:
        _zr = zero_matcher.match(_query)
        _z_entry = _zr if isinstance(_zr, dict) else _zr
        _z_id = _z_entry.get("id", "?") if isinstance(_z_entry, dict) else "?"
        _z_score = _z_entry.get("score", 0) if isinstance(_z_entry, dict) else 0

        _tr = trained_matcher.match(_query)
        _t_entry = _tr if isinstance(_tr, dict) else _tr
        _t_id = _t_entry.get("id", "?") if isinstance(_t_entry, dict) else "?"
        _t_score = _t_entry.get("score", 0) if isinstance(_t_entry, dict) else 0

        _rows.append(
            {
                "query": _query,
                "expected": _expected,
                "zero_shot_id": _z_id,
                "zero_shot_score": f"{_z_score:.2%}",
                "zero_shot_correct": "OK" if _z_id == _expected else "MISS",
                "trained_id": _t_id,
                "trained_score": f"{_t_score:.2%}",
                "trained_correct": "OK" if _t_id == _expected else "MISS",
            }
        )

    mo.ui.table(_rows, label=f"Comparison: zero-shot vs {_trained_mode} ({_n} samples)")
    return


@app.cell
def _(
    Matcher,
    entities,
    full_training,
    n_samples,
    plt,
    test_queries,
    zero_matcher,
):
    _n = n_samples.value

    _sample_counts = list(range(min(_n + 1, len(full_training) + 1)))
    _zero_acc = []
    _trained_acc = []

    for _count in _sample_counts:
        _z_correct = sum(
            1
            for _q, _exp in test_queries
            if (_entry := zero_matcher.match(_q))
            and (_e := _entry if isinstance(_entry, dict) else _entry)
            and (_e.get("id", "?") if isinstance(_e, dict) else "?") == _exp
        )
        _zero_acc.append(_z_correct / len(test_queries))

        if _count > 0:
            _subset = full_training[:_count]
            try:
                _tm = Matcher(entities=entities, verbose=False)
                _tm.fit(training_data=_subset, num_epochs=1)
                _t_correct = sum(
                    1
                    for _q, _exp in test_queries
                    if (_entry := _tm.match(_q))
                    and (_e := _entry if isinstance(_entry, dict) else _entry)
                    and (_e.get("id", "?") if isinstance(_e, dict) else "?") == _exp
                )
                _trained_acc.append(_t_correct / len(test_queries))
            except Exception:
                _trained_acc.append(_zero_acc[-1])
        else:
            _trained_acc.append(_zero_acc[-1])

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_sample_counts, _zero_acc, "o--", label="Zero-shot", color="#3498db")
    _ax.plot(_sample_counts, _trained_acc, "s-", label="Trained", color="#e74c3c")
    _ax.set_xlabel("Number of Training Samples")
    _ax.set_ylabel("Accuracy")
    _ax.set_title("Zero-Shot vs Trained Matching Accuracy")
    _ax.set_ylim(0, 1.05)
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
