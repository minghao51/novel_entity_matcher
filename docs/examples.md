# Examples

Related docs: [`index.md`](./index.md) | [`quickstart.md`](./quickstart.md) | [`experiments/index.md`](./experiments/index.md)

This page catalogs the organized example layout under `examples/`.

## Current Examples

These are the maintained examples for the unified `Matcher` API.

| Example | Difficulty | Runtime | What it covers |
|---|---|---|---|
| [`current/basic_matcher.py`](../examples/current/basic_matcher.py) | Beginner | 30s | Zero-shot matching with the unified `Matcher` API |
| [`current/trained_matcher.py`](../examples/current/trained_matcher.py) | Beginner | 1-2 min | Few-shot training with `Matcher.fit(training_data=...)` |
| [`current/hierarchical_matching.py`](../examples/current/hierarchical_matching.py) | Intermediate | 30s | Hierarchical matching with `HierarchicalMatcher` |

## Legacy Examples

These files are intentionally kept in `examples/legacy/` for migration help only. They use deprecated classes and should not be the default path for new users.

| Example | Status | Notes |
|---|---|---|
| [`legacy/embedding_matcher_demo.py`](../examples/legacy/embedding_matcher_demo.py) | Deprecated | Uses `EmbeddingMatcher` |
| [`legacy/entity_matcher_demo.py`](../examples/legacy/entity_matcher_demo.py) | Deprecated | Uses `EntityMatcher` |
| [`legacy/matcher_comparison.py`](../examples/legacy/matcher_comparison.py) | Deprecated | Compares deprecated APIs |
| [`legacy/model_persistence.py`](../examples/legacy/model_persistence.py) | Deprecated | References legacy classifier persistence flows |
| [`legacy/batch_processing.py`](../examples/legacy/batch_processing.py) | Deprecated | Uses legacy bulk APIs |
| [`legacy/threshold_tuning.py`](../examples/legacy/threshold_tuning.py) | Deprecated | Threshold tuning examples on deprecated matchers |
| [`legacy/hybrid_matching_demo.py`](../examples/legacy/hybrid_matching_demo.py) | Deprecated | Uses `HybridMatcher` directly |

For migration details, see [`migration-guide.md`](./migration-guide.md).

## Raw Examples

These files use lower-level libraries directly and are intended for advanced experimentation.

| Example | Category | What it demonstrates |
|---|---|---|
| [`raw/basic_usage.py`](../examples/raw/basic_usage.py) | Raw SetFit training | Minimal direct SetFit workflow |
| [`raw/country_matching.py`](../examples/raw/country_matching.py) | Raw SetFit training | Country-code matching outside wrapper APIs |
| [`raw/custom_backend.py`](../examples/raw/custom_backend.py) | Backend exploration | Model/backend tradeoffs via direct library usage |
| [`raw/zero_shot_classification.py`](../examples/raw/zero_shot_classification.py) | Generic classification | Non-entity SetFit classification examples |

## Recommended Order

1. Read [`quickstart.md`](./quickstart.md).
2. Run [`current/basic_matcher.py`](../examples/current/basic_matcher.py).
3. Run [`current/trained_matcher.py`](../examples/current/trained_matcher.py) if you have labels.
4. Use `examples/legacy/` only when migrating older code.
5. Use `examples/raw/` when you need lower-level control.
