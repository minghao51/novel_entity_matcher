# Troubleshooting

Related docs: [`quickstart.md`](./quickstart.md) | [`examples.md`](./examples.md) | [`experiments/index.md`](./experiments/index.md) | [`index.md`](./index.md)

This page covers common setup and first-run issues for the package plus exploratory scripts/notebooks.

## Import Errors (`setfit`, `datasets`, `torch`, `sentence-transformers`)

Symptoms:

- `ImportError: setfit is required...`
- `ModuleNotFoundError` for `datasets`, `torch`, or `sentence_transformers`

What to check:

1. Install the project dependencies in the active environment.
2. Make sure your Jupyter kernel matches the environment where the package is installed.
3. Re-run from the repo root if using `PYTHONPATH=.`

## Slow First Run / Model Downloads

First run often downloads model weights from Hugging Face. This can take time depending on model size and network speed.

What to expect:

- Small examples: may still pause while downloading
- Some advanced experiments/notebooks may require larger model downloads and slower startup
- Typical model sizes: 100-500MB

## CPU vs GPU Expectations

- CPU works for basic testing and small examples.
- SetFit training and large embedding models can be significantly slower on CPU.
- GPU is optional but helpful for the country classifier experiments and larger embedding experiments.

## `EmbeddingMatcher` Error: Index Not Built

Symptom:

- `RuntimeError("Index not built. Call build_index() first.")`

Fix:

```python
matcher = EmbeddingMatcher(entities=entities)
matcher.build_index()
result = matcher.match("query")
```

## `EntityMatcher` Error: Model Not Trained

Symptom:

- `RuntimeError("Model not trained. Call train() first.")`

Fix:

```python
matcher = EntityMatcher(entities=entities)
matcher.train(training_data)
result = matcher.predict("query")
```

## Low-Confidence Matches (Returns `None`)

Symptoms:

- `match()` or `predict()` returns `None` for valid queries
- Fewer matches than expected

Causes:

- **Threshold too high**: Confidence score below threshold
- **Insufficient training data**: Model hasn't learned variations
- **Wrong matcher type**: Using EmbeddingMatcher for complex variations

Solutions:

1. **Lower the threshold**:
```python
# Default threshold is 0.7, try lowering
matcher = EmbeddingMatcher(entities, threshold=0.5)
```

2. **Check confidence scores**:
```python
# For EntityMatcher, see probability distribution
probs = matcher.classifier.predict_proba("query")
print({label: prob for label, prob in zip(matcher.classifier.labels, probs)})
```

3. **Use EntityMatcher instead**:
```python
# EmbeddingMatcher struggles with variations
# EntityMatcher handles them better with training
entity_matcher = EntityMatcher(entities)
entity_matcher.train(training_data)  # Include variations in training
```

## Threshold Tuning Guidance

If you're getting **too many matches** (low precision):

- Raise threshold: `0.7` → `0.8` or `0.9`
- Use validation data to find optimal threshold

If you're getting **too few matches** (low recall):

- Lower threshold: `0.7` → `0.6` or `0.5`
- Check if queries have typos or extreme variations
- Consider adding more training examples (EntityMatcher)

**Recommended thresholds by use case**:

- **High precision** (0.8-0.9): Database lookups, exact matching
- **Balanced** (0.7, default): General purpose
- **High recall** (0.5-0.6): Fuzzy search, data cleaning

See [`examples/legacy/threshold_tuning.py`](../examples/legacy/threshold_tuning.py) for a legacy threshold tuning walkthrough.

## Model Selection Issues

**Problem**: Model not working well for your language/domain

**Solutions**:

1. **Try multilingual models**:
```python
# For non-English text
matcher = EntityMatcher(entities, model_name="sentence-transformers/LaBSE")
# or
matcher = EmbeddingMatcher(entities, model_name="paraphrase-multilingual-mpnet-base-v2")
```

2. **Use domain-specific models**:
```python
# For high-accuracy English
matcher = EmbeddingMatcher(entities, model_name="bge-base")

# For speed
matcher = EmbeddingMatcher(entities, model_name="minilm")
```

## Notebook Dependency Issues (`jupyter`, `geograpy`)

### Jupyter

- Install Jupyter in the same environment as `semanticmatcher`
- Launch from repo root to avoid path confusion

### `geograpy`

- If you add a local `geograpy` notebook experiment, expect extra installs and dependency troubleshooting beyond the core project

## Path Migration Note (`notebook/` -> `experiments/`)

The old experiment script path `notebook/...` was moved into `experiments/...`.

Updated examples:

- `experiments/country_classifier/country_classifier.py`
- `experiments/country_classifier/country_classifier_quick.py`
- `experiments/country_classifier/country_classifier_advanced.py`

## Static Embedding Issues

### model2vec Import Error

**Symptom:**
- `ModuleNotFoundError: No module named 'model2vec'`

**Cause:**
- Trying to use potion models without model2vec installed

**Fix:**
```bash
# Install model2vec
uv pip install model2vec

# Or with extras
uv pip install semantic-matcher[static]
```

### MRL Model Loading Error

**Symptom:**
- `Failed to load static embedding model`
- `AttributeError: 'StaticEmbedding' module not found`

**Cause:**
- RikkaBotan MRL models require `trust_remote_code=True`

**Fix:**
```python
from semanticmatcher.backends.static_embedding import StaticEmbeddingBackend

# Automatically handled by Matcher
from semanticmatcher import Matcher
matcher = Matcher(model="mrl-en")  # Works correctly
```

### MPS Fallback Warning (Apple Silicon)

**Symptom:**
- Warning about MPS fallback on Apple Silicon

**Cause:**
- RikkaBotan MRL models use operations not supported by MPS

**Fix:**
- Already handled automatically - library sets `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Warning is informational, not an error

### Static Model Auto-Fallback

**Symptom:**
- Training with `potion-8m` uses `mpnet` instead

**Cause:**
- Static models don't support SetFit training
- Library auto-falls back to training-compatible model

**Fix:**
```python
# Explicitly use training-compatible model
matcher = Matcher(model="mpnet")  # Not potion-8m
matcher.fit(training_data, mode="full")

# Or accept the fallback
matcher = Matcher(model="potion-8m")
matcher.fit(training_data, mode="full")  # Will use mpnet for training
```

See [`static-embeddings.md`](./static-embeddings.md) for more details.

## Matcher Mode Issues

### Auto-Detection Not Working as Expected

**Symptom:**
- Wrong mode selected by auto-detection
- Expected `head-only` but got `full`

**Cause:**
- Auto-detection counts examples per entity
- ≥ 3 examples per entity triggers full training

**Fix:**
```python
# Check detected mode
matcher = Matcher(entities=entities, mode="auto")
matcher.fit(training_data)
print(matcher.get_training_info()["detected_mode"])

# Override mode explicitly
matcher = Matcher(entities=entities, mode="head-only")
matcher.fit(training_data)
```

### Training Data Required Error

**Symptom:**
- `ValueError: training_data is required for modes 'head-only' and 'full'`

**Cause:**
- Requested training mode without providing training data

**Fix:**
```python
# Wrong
matcher = Matcher(mode="full")
matcher.fit()  # Error!

# Right
matcher = Matcher(mode="full")
matcher.fit(training_data)  # Provide training data

# Or use zero-shot for no training
matcher = Matcher(mode="zero-shot")
matcher.fit()  # OK
```

### Hybrid Mode Not Working

**Symptom:**
- Hybrid mode returns no results
- Very slow matching

**Causes:**
1. **Blocking too aggressive** - Filters out all candidates
2. **Dataset too small** - Hybrid overkill for <10k entities

**Fix:**
```python
# 1. Try different blocking strategy
from semanticmatcher.core.blocking import NoOpBlocking

matcher = Matcher(
    entities=entities,
    mode="hybrid",
    blocking_strategy=NoOpBlocking()  # No filtering
)

# 2. Increase blocking_top_k
result = matcher.match(
    "query",
    blocking_top_k=5000  # More candidates
)

# 3. Use simpler mode for small datasets
matcher = Matcher(entities=entities, mode="zero-shot")
```

### Mode Not Supported Error

**Symptom:**
- `ModeError: Invalid mode: 'invalid_mode'`

**Cause:**
- Typos or invalid mode names

**Valid modes:**
- `zero-shot`
- `head-only`
- `full`
- `hybrid`
- `auto`

**Fix:**
```python
# Check mode spelling
matcher = Matcher(entities=entities, mode="zero-shot")  # Correct
# Not "zeroshot" or "Zero-Shot"
```

See [`matcher-modes.md`](./matcher-modes.md) for complete mode guide.

## Performance Issues

**Problem**: Matching is too slow

**Solutions**:

1. **Use faster model**:
```python
# Use minilm for speed (3-5x faster)
matcher = EmbeddingMatcher(entities, model_name="minilm")
```

2. **Use batch processing**:
```python
# For multiple queries
results = matcher.match_bulk(queries)  # HybridMatcher
# Or loop for EmbeddingMatcher (still fast)
results = [matcher.match(q) for q in queries]
```

3. **Reduce embedding dimension** (Matryoshka):
```python
# Use smaller embeddings for faster matching
matcher = EmbeddingMatcher(entities, embedding_dim=256)
```

## Common Errors by Matcher

### EmbeddingMatcher

| Error | Cause | Fix |
|---|---|---|
| `Index not built` | Didn't call `build_index()` | Call `matcher.build_index()` |
| Returns `None` | Threshold too high | Lower `threshold` |
| Slow matching | Large model or dataset | Use `minilm` model |

### EntityMatcher

| Error | Cause | Fix |
|---|---|---|
| `Model not trained` | Didn't call `train()` | Call `matcher.train(data)` |
| Returns `None` | Threshold too high or insufficient training | Lower threshold or add training examples |
| Training slow | Large model or many epochs | Use `minilm` and fewer epochs |

### HybridMatcher

| Error | Cause | Fix |
|---|---|---|
| Empty results | Blocking stage filters everything | Try `NoOpBlocking()` or increase `blocking_top_k` |
| Slow | Small dataset | Use `EmbeddingMatcher` instead |

## Getting More Help

If you're still stuck:

1. **Check diagnostic tools**:
   ```python
   diagnosis = matcher.diagnose("problematic query")
   print(diagnosis["suggestion"])
   ```

2. **Review documentation**:
   - [`quickstart.md`](./quickstart.md) - Basic usage
   - [`examples.md`](./examples.md) - Example catalog
   - [`models.md`](./models.md) - Model selection guide
   - [`matcher-modes.md`](./matcher-modes.md) - Mode system
   - [`static-embeddings.md`](./static-embeddings.md) - Static embedding details
   - [`configuration.md`](./configuration.md) - Configuration options

3. **Check diagnostics**:
   ```python
   explanation = matcher.explain_match("query", top_k=5)
   print(explanation)
   ```

4. **Search issues**: [GitHub Issues](https://github.com/anomalyco/semantic_matcher/issues)

5. **Create an issue** with:
   - Code snippet
   - Data sample (sanitized)
   - Error message
   - `diagnose()` output
   - `get_training_info()` output
