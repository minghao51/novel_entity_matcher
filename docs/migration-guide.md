# Migration Guide: Unified Matcher API

This guide helps you migrate from the old multi-class API (`EntityMatcher`, `EmbeddingMatcher`, `HybridMatcher`) to the new unified `Matcher` class.

## What Changed?

### Before (Deprecated)
You had to choose between three different matcher classes:
- `EntityMatcher` - SetFit few-shot learning (requires training)
- `EmbeddingMatcher` - Zero-shot similarity matching (no training)
- `HybridMatcher` - Three-stage pipeline (complex setup)

### After (New)
Single `Matcher` class that **auto-detects** the best approach:
- No training data → zero-shot (embedding similarity)
- < 3 examples/entity → head-only training (~30s)
- ≥ 3 examples/entity → full training (~3min)

---

## Quick Migration Examples

### Example 1: Zero-Shot Matching (No Training)

**Before:**
```python
from semanticmatcher import EmbeddingMatcher

matcher = EmbeddingMatcher(entities=[
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
    {"id": "US", "name": "United States", "aliases": ["USA"]},
])
matcher.build_index()
result = matcher.match("America")  # {"id": "US", "score": 0.95}
```

**After:**
```python
from semanticmatcher import Matcher

matcher = Matcher(entities=[
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
    {"id": "US", "name": "United States", "aliases": ["USA"]},
])
matcher.fit()  # No training data = zero-shot mode
result = matcher.match("America")  # {"id": "US", "score": 0.95}
```

---

### Example 2: Few-Shot Training

**Before:**
```python
from semanticmatcher import EntityMatcher

matcher = EntityMatcher(entities=[
    {"id": "DE", "name": "Germany"},
    {"id": "US", "name": "United States"},
])

training_data = [
    {"text": "Germany", "label": "DE"},
    {"text": "Deutschland", "label": "DE"},
    {"text": "USA", "label": "US"},
    {"text": "America", "label": "US"},
]

matcher.train(training_data)
result = matcher.predict("Deutschland")  # "DE"
```

**After:**
```python
from semanticmatcher import Matcher

matcher = Matcher(entities=[
    {"id": "DE", "name": "Germany"},
    {"id": "US", "name": "United States"},
])

training_data = [
    {"text": "Germany", "label": "DE"},
    {"text": "Deutschland", "label": "DE"},
    {"text": "USA", "label": "US"},
    {"text": "America", "label": "US"},
]

matcher.fit(training_data)  # Auto-detects training mode
result = matcher.match("Deutschland")  # {"id": "DE", "score": 1.0}

# Or use predict() for just entity IDs:
result = matcher.predict("Deutschland")  # "DE"
```

---

### Example 3: Explicit Mode Selection

**Before:**
```python
# Had to choose which class to use
from semanticmatcher import EmbeddingMatcher

matcher = EmbeddingMatcher(entities=entities)
matcher.build_index()
```

**After:**
```python
# Can explicitly specify mode if needed
from semanticmatcher import Matcher

matcher = Matcher(entities=entities, mode="zero-shot")
matcher.fit()
```

---

## API Differences

### Method Changes

| Old API | New API | Notes |
|---------|---------|-------|
| `EmbeddingMatcher(entities)` | `Matcher(entities)` | Unified class |
| `build_index()` | `fit()` | Consistent method name |
| `match(texts)` | `match(texts)` | Same signature |
| `EntityMatcher.train(data)` | `Matcher.fit(data)` | Consistent method name |
| `EntityMatcher.predict(texts)` | `Matcher.match(texts)` | Returns full match dict |
| N/A | `Matcher.predict(texts)` | Convenience for entity IDs only |

### Return Value Changes

**`EmbeddingMatcher.match()`** (unchanged):
```python
result = matcher.match("America")
# Returns: {"id": "US", "score": 0.95, "text": "United States"}
```

**`EntityMatcher.predict()`** → **`Matcher.match()`**:
```python
# Old API
result = matcher.predict("America")
# Returns: "US" (just the entity ID)

# New API
result = matcher.match("America")
# Returns: {"id": "US", "score": 1.0, "text": "United States"} (full dict)

# New API (convenience method)
result = matcher.predict("America")
# Returns: "US" (just the entity ID, like old API)
```

---

## Parameter Mapping

### Common Parameters

| Parameter | Old API | New API |
|-----------|---------|---------|
| Entities | `entities=[...]` | `entities=[...]` (same) |
| Model | `model_name="..."` | `model="..."` (supports aliases) |
| Threshold | `threshold=0.7` | `threshold=0.7` (same) |
| Normalization | `normalize=True` | `normalize=True` (same) |
| Training data | `train(data, num_epochs=4)` | `fit(data, num_epochs=4)` (same) |

### New Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `mode` | Explicitly set matching mode | `mode="zero-shot"` |
| `top_k` | Number of results (new API) | `match(query, top_k=3)` |

---

## Model Selection

### Old API
```python
matcher = EmbeddingMatcher(
    entities=entities,
    model_name="sentence-transformers/all-mpnet-base-v2"  # Full model name
)
```

### New API (with aliases)
```python
matcher = Matcher(
    entities=entities,
    model="mpnet"  # Short alias, or use full model name
)

# Available aliases:
# - "default" → sentence-transformers/all-mpnet-base-v2
# - "mpnet" → sentence-transformers/all-mpnet-base-v2
# - "minilm" → sentence-transformers/all-MiniLM-L6-v2
# - "bge-base" → BAAI/bge-base-en-v1.5
# - "bge-m3" → BAAI/bge-m3
# - "nomic" → nomic-ai/nomic-embed-text-v1
```

---

## Common Migration Patterns

### Pattern 1: Migrating EmbeddingMatcher

**Old Code:**
```python
from semanticmatcher import EmbeddingMatcher

matcher = EmbeddingMatcher(entities=entities)
matcher.build_index()
result = matcher.match(query, top_k=5)
```

**New Code:**
```python
from semanticmatcher import Matcher

matcher = Matcher(entities=entities)
matcher.fit()  # Auto zero-shot
result = matcher.match(query, top_k=5)
```

### Pattern 2: Migrating EntityMatcher

**Old Code:**
```python
from semanticmatcher import EntityMatcher

matcher = EntityMatcher(entities=entities)
matcher.train(training_data, num_epochs=4)
entity_id = matcher.predict(query)
```

**New Code:**
```python
from semanticmatcher import Matcher

matcher = Matcher(entities=entities)
matcher.fit(training_data, num_epochs=4)
entity_id = matcher.predict(query)  # Convenience method

# OR get full match info:
match = matcher.match(query)  # {"id": "...", "score": ..., "text": "..."}
```

### Pattern 3: Batch Matching

**Old Code:**
```python
results = matcher.match(["query1", "query2", "query3"])
# Returns list of dicts
```

**New Code:**
```python
results = matcher.match(["query1", "query2", "query3"])
# Returns list of dicts (same)
```

---

## Deprecation Warnings

When you use the old classes, you'll see a deprecation warning:

```python
from semanticmatcher import EntityMatcher

# DeprecationWarning: EntityMatcher is deprecated and will be removed
# in a future version. Use the unified Matcher class instead.
# See documentation for migration guide.
```

### Suppressing Warnings (Temporary)

If you need to temporarily suppress warnings while migrating:

```python
import warnings
from semanticmatcher import EntityMatcher

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    matcher = EntityMatcher(entities=entities)
```

---

## Troubleshooting

### Issue: "No module named 'semanticmatcher.core.matcher'"

**Solution:** Make sure you're using the latest version of semantic-matcher:
```bash
pip install --upgrade semantic-matcher
```

### Issue: TypeErrors or AttributeError after migration

**Solution:** Check that you're using the correct method names:
- Old: `train()` → New: `fit()`
- Old: `predict()` → New: `match()` (or `predict()` for IDs only)
- Old: `build_index()` → New: `fit()`

### Issue: Different return values

**Solution:** The new `match()` method always returns a dict with keys `id`, `score`, and `text`. Use `predict()` if you just want entity IDs.

---

## Best Practices with New API

### 1. Let Auto-Detection Work

```python
# Good - let Matcher decide
matcher = Matcher(entities=entities)
matcher.fit(training_data)  # Auto-detects mode

# Also OK - explicit mode
matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data)
```

### 2. Use Model Aliases

```python
# Good - short and readable
matcher = Matcher(entities=entities, model="bge-base")

# Also works - full model name
matcher = Matcher(entities=entities, model="BAAI/bge-base-en-v1.5")
```

### 3. Leverage Convenience Methods

```python
# For just entity IDs (like old API)
entity_id = matcher.predict(query)  # Returns: "US"

# For full match info
match = matcher.match(query)  # Returns: {"id": "US", "score": 0.95, ...}
```

### 4. Batch Processing

```python
# Efficient batch matching
queries = ["query1", "query2", "query3", ...]
results = matcher.match(queries)  # Single call for all queries
```

---

## Need Help?

- **Documentation:** See [quickstart.md](quickstart.md) for more examples
- **Examples:** Check the `examples/` directory for complete scripts
- **Issues:** Report problems on GitHub

---

## Summary of Changes

| Aspect | Old API | New API |
|--------|---------|---------|
| **Classes** | 3 classes (EntityMatcher, EmbeddingMatcher, HybridMatcher) | 1 class (Matcher) |
| **Selection** | Manual (user chooses class) | Auto-detection based on data |
| **Method Names** | `train()`, `predict()`, `build_index()` | Unified: `fit()`, `match()`, `predict()` |
| **Model Selection** | Full model names required | Aliases supported (`"mpnet"`, `"bge-base"`) |
| **Return Values** | Varies by class | Consistent dict format |
| **Deprecation** | Current (showing warnings) | Future (will be removed) |
