# Implementation Notes

Archived note: this page captures a point-in-time maintenance pass and is kept only for historical reference.

**Date:** 2026-03-26

## Recent Changes

### Mypy Type Error Fixes (0472ebe)

Fixed type annotations and mypy compatibility issues across 11 files:

- `core/hierarchy.py` - Added type annotations for `_cache`, `entity_embeddings`, `entity_texts`
- `core/matcher.py` - Typed `_async_executor` and results lists
- `core/embedding_matcher.py` - Same async executor and list typing fixes
- `core/bert_classifier.py` - Fixed tokenizer closure capture, improved isinstance narrowing
- `utils/validation.py` - Restructured duplicate detection loop for mypy
- `utils/benchmark_dataset.py` - Added type annotations for empty_split, base_pairs, etc.
- `benchmarks/classification/evaluator.py` - Moved `per_class_f1` to `details` dict
- `benchmarks/novelty/evaluator.py` - Added assertion for `detector_fn` narrowing
- `benchmarks/loader.py` - Added `Dict` import
- `ingestion/cli.py` - Added None check for func
- `backends/litellm.py` - Removed unused type:ignore

### Refactor (1863bc5)

- Renamed `EntityMatcher` to `_EntityMatcher` (private class merged into matcher.py)
- Added state machine validation for review workflow transitions
- Fixed Pydantic `model_dump()` serialization in persistence layer
- Added `networkx` dependency for future hierarchy support
- Cleaned up archive and experimental files

## Notes

- Remaining mypy errors in `benchmarks/runner.py` and `benchmarks/cli.py` are pre-existing
- Type stub errors for external libraries (yaml, requests, pandas) are non-blocking
