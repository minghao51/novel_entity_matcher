# Quick Start Guide

Related docs: [`index.md`](./index.md) | [`examples.md`](./examples.md) | [`troubleshooting.md`](./troubleshooting.md)

This guide covers the main `novelentitymatcher.Matcher` workflow. Use it when you want to map messy input text to canonical entity IDs.

## Install

### From PyPI

```bash
uv add novel-entity-matcher
```

### For local development

```bash
uv sync --group dev
```

If you are not using `uv`, `pip install novel-entity-matcher` also works.

## Basic Zero-Shot Matching

Use zero-shot mode when you do not have labeled training data yet. The async route is the recommended default for new integrations.

```python
import asyncio
from novelentitymatcher import Matcher

async def main():
    entities = [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
        {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
    ]

    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()

        print(await matcher.match_async("Deutschland"))
        # {'id': 'DE', 'score': 0.9..., 'text': 'Germany'}

        print(await matcher.match_async("America"))
        # {'id': 'US', 'score': 0.9..., 'text': 'United States'}

asyncio.run(main())
```

What happens here:

- `Matcher(...)` validates and stores your entity catalog.
- `fit_async()` builds the embedding index without blocking the event loop.
- `match_async()` returns match objects with scores.
- `predict()` remains available for sync-first scripts.

Sync alternative:

```python
matcher = Matcher(entities=entities)
matcher.fit()
print(matcher.predict("America"))
```

## Entity Format

Each entity must include:

- `id`: stable canonical ID
- `name`: primary display name

Optional fields:

- `aliases`: alternate names, abbreviations, common misspellings

Example:

```python
entities = [
    {
        "id": "GB",
        "name": "United Kingdom",
        "aliases": ["UK", "Great Britain", "Britain"],
    }
]
```

## Add Training Data

If you have labeled examples, pass them to `fit(training_data=...)`.

```python
from novelentitymatcher import Matcher

entities = [
    {"id": "DE", "name": "Germany"},
    {"id": "US", "name": "United States"},
]

training_data = [
    {"text": "Germany", "label": "DE"},
    {"text": "Deutschland", "label": "DE"},
    {"text": "USA", "label": "US"},
    {"text": "America", "label": "US"},
]

matcher = Matcher(entities=entities)
matcher.fit(training_data=training_data, num_epochs=1)

print(matcher.match("United States"))
print(matcher.predict(["Deutschland", "America"]))
```

Auto-selection rules:

- No training data: `zero-shot`
- Fewer than 3 examples for the most represented entity: `head-only`
- At least 3 examples for some entity: `full`

## Choose a Mode Explicitly

Override auto-selection when you want deterministic behavior.

```python
matcher = Matcher(entities=entities, mode="zero-shot")
matcher.fit()

matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data=training_data, num_epochs=1)

matcher = Matcher(entities=entities, mode="hybrid")
matcher.fit()
```

Supported modes:

- `zero-shot`: embedding similarity, no training
- `head-only`: lightweight SetFit training path
- `full`: full training path for higher accuracy
- `hybrid`: blocking + retrieval + reranking

## Return Shapes

Single input with default `top_k=1`:

```python
result = matcher.match("USA")
# {'id': 'US', 'score': 0.9..., 'text': 'United States'}
```

Single input with `top_k > 1`:

```python
results = matcher.match("United", top_k=3)
# [{'id': ...}, {'id': ...}, ...]
```

Batch input:

```python
results = matcher.match(["USA", "Deutschland"])
# [{'id': 'US', ...}, {'id': 'DE', ...}]
```

If nothing clears the threshold, the matcher returns `None` for `top_k=1` or `[]` for multi-result queries.

## Useful Parameters

```python
matcher = Matcher(
    entities=entities,
    model="default",
    threshold=0.7,
    normalize=True,
    verbose=False,
)
```

Common options:

- `model`: model alias or full sentence-transformer model name
  - `default` uses static embeddings (fast, good for retrieval)
  - `mpnet` or `bge-base` for better accuracy with training
  - See [`models.md`](./models.md) for all options
- `threshold`: minimum score required for a match
- `normalize`: normalize text before matching
- `verbose`: print mode and fit diagnostics

## Candidate Filtering

Restrict matching to a subset of known candidates.

```python
candidates = [
    {"id": "DE", "name": "Germany"},
    {"id": "US", "name": "United States"},
]

print(matcher.match("America", candidates=candidates))
```

This is useful when another upstream system has already narrowed the search space.

## Inspect Matcher State

```python
info = matcher.get_training_info()
stats = matcher.get_statistics()

print(info)
print(stats)
```

## Diagnostic Tools

### explain_match()

Get detailed information about why a query matched (or didn't match):

```python
explanation = matcher.explain_match("Deutchland", top_k=5)

print(explanation["matched"])       # True/False
print(explanation["best_match"])    # Top result
print(explanation["top_k"])         # All candidates
print(explanation["threshold"])     # Current threshold
print(explanation["mode"])          # Active mode
```

**Example output:**
```python
{
    "query": "Deutchland",
    "query_normalized": "deutschland",
    "matched": True,
    "best_match": {"id": "DE", "score": 0.92, "text": "Germany"},
    "top_k": [
        {"id": "DE", "score": 0.92},
        {"id": "US", "score": 0.75},
        {"id": "FR", "score": 0.68}
    ],
    "threshold": 0.7,
    "mode": "zero-shot"
}
```

**Use cases:**
- Debug low match rates
- Understand score distribution
- Verify threshold settings
- Investigate edge cases

### diagnose()

Get actionable suggestions when matches fail:

```python
diagnosis = matcher.diagnose("UnknownPlace")

print(diagnosis["issue"])       # What's wrong
print(diagnosis["suggestion"])  # How to fix it
```

**Example output:**
```python
{
    "query": "UnknownPlace",
    "matcher_ready": True,
    "active_matcher": "EmbeddingMatcher",
    "issue": "Score 0.45 below threshold 0.7",
    "suggestion": "Lower threshold with matcher.set_threshold(0.6) or add more training examples"
}
```

**Common diagnoses:**
- Score below threshold → Suggests lowering threshold
- No candidates found → Suggests checking entity data
- Matcher not ready → Suggests calling `fit()`

For debugging a specific query:

```python
print(matcher.explain_match("Deutchland"))
print(matcher.diagnose("UnknownPlace"))
```

## Common First-Run Notes

- First run may download model weights, so it can take longer than later runs.
- `fit()` is required before calling `match()` if you want explicit setup, but `match()` will auto-call `fit()` in the default flow.
- Lower `threshold` if likely matches are being filtered out.
- Add aliases or training examples if close variants are missing.

## Run Examples

Project examples live in [`examples.md`](./examples.md).

From the repository root:

```bash
uv run python examples/current/basic_matcher.py
uv run python examples/current/trained_matcher.py
```

## Next Steps

- See [`examples.md`](./examples.md) for the organized example catalog.
- See [`troubleshooting.md`](./troubleshooting.md) if model downloads or imports fail.
