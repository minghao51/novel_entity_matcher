# Static Embeddings

Related docs: [`index.md`](./index.md) | [`quickstart.md`](./quickstart.md) | [`models.md`](./models.md) | [`architecture.md`](./architecture.md)

## Overview

Static embeddings use pre-computed lookup tables instead of on-the-fly encoding, providing **10-100x faster** matching with minimal accuracy tradeoffs.

## How It Works

### Dynamic Embeddings (Traditional)

```
Input Text → Neural Network → Embedding Vector
   └──────────── ~50-500ms ────────────┘
```

### Static Embeddings

```
Input Text → Token Lookup → Pre-computed Vector
   └────────── ~1-5ms ──────────┘
```

Static embeddings skip the neural network forward pass by storing pre-computed vectors for each token in the vocabulary.

## Supported Static Backends

### 1. model2vec (minishlab potion models)

Fast, lightweight models distilled from larger sentence transformers.

**Models:**
- `potion-8m` - 8M parameters, ultra-fast (default retrieval model)
- `potion-32m` - 32M parameters, better quality

**Characteristics:**
- Best for: English general-purpose retrieval
- Backend: `StaticModel.from_pretrained()`
- Dimension: 256-384 (configurable)

**Usage:**
```python
from novelentitymatcher import Matcher

matcher = Matcher(entities=entities, model="potion-8m")
matcher.fit()
result = matcher.match("query")  # ~1-5ms per query
```

### 2. StaticEmbedding (RikkaBotan MRL models)

Matryoshka Representation Learning (MRL) models with configurable dimensionality.

**Models:**
- `mrl-en` - English-only with MRL support
- `mrl-multi` - Multilingual with MRL support

**Characteristics:**
- Best for: Multilingual retrieval or dimension-constrained scenarios
- Backend: SentenceTransformer with StaticEmbedding module
- Dimension: Variable (truncate at runtime for efficiency)

**Usage with MRL dimension reduction:**
```python
# Use full dimension
matcher = Matcher(entities=entities, model="mrl-en")

# Use reduced dimension (faster, less memory)
matcher = Matcher(
    entities=entities,
    model="mrl-en",
    embedding_dim=256  # Truncate to 256 dimensions
)
```

## When to Use Static vs Dynamic

| Use Case | Recommended | Why |
|---|---|---|
| High-throughput retrieval | Static (`potion-8m`) | 40-100x faster, sufficient accuracy |
| Multilingual retrieval | Static (`mrl-multi`) | Fast multilingual support |
| Training with few-shot data | Dynamic (`mpnet`) | SetFit requires trainable backbone |
| Context-heavy queries | Dynamic (`bge-base`) | Better contextual understanding |
| Resource-constrained | Static (`potion-8m`) | Lower CPU/memory usage |

## Performance Comparison

Benchmark results (queries per second):

| Model | Type | Throughput | Speedup vs minilm |
|---|---|---|---|
| potion-8m | Static | ~4000 QPS | 39x faster |
| potion-32m | Static | ~3500 QPS | 34x faster |
| mrl-en | Static | ~1800 QPS | 17x faster |
| minilm | Dynamic | ~100 QPS | baseline |

*Results from [`experiments/benchmarking.md`](./experiments/benchmarking.md) - actual performance varies by hardware.*

## Dimension Reduction with MRL

MRL (Matryoshka Representation Learning) models allow runtime dimensionality reduction:

```python
# Full dimension (768d)
matcher_full = Matcher(entities=entities, model="mrl-en")

# Reduced dimension (256d) - faster, less memory
matcher_reduced = Matcher(
    entities=entities,
    model="mrl-en",
    embedding_dim=256
)

# Both work, reduced is 3x faster with minimal accuracy loss
```

**Benefits:**
- Faster similarity computation
- Lower memory footprint
- Tunable speed/accuracy tradeoff

**How to choose dimension:**
- Start with model's default (usually best)
- Reduce if memory/speed constrained
- Test accuracy impact on your data

## Auto-Detection

The library automatically detects static embedding models:

```python
# These all use static embeddings automatically
Matcher(model="potion-8m")      # model2vec
Matcher(model="mrl-en")         # StaticEmbedding
Matcher(model="minishlab/potion-base-8M")  # Full name
```

Detection logic:
1. Try `model2vec.StaticModel.from_pretrained()`
2. Fall back to `SentenceTransformer` with StaticEmbedding
3. Use dynamic SentenceTransformer if neither works

## Fallback for Training

Static models don't support SetFit training. When training is requested:

```python
# Static model requested for training
matcher = Matcher(entities=entities, model="potion-8m")
matcher.fit(training_data, mode="full")  # ⚠️ Fallback to mpnet

# Training happens with mpnet (trainable), not potion-8m
```

**Why:** Static embeddings lack the trainable parameters required for SetFit.

**See:** [`models.md`](./models.md) for training-compatible model options.

## Technical Details

### model2vec Backend

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8M")
embeddings = model.encode(["text1", "text2"])
# Returns pre-computed vectors via lookup
```

### StaticEmbedding Backend

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en",
    trust_remote_code=True  # Required for custom StaticEmbedding module
)
embeddings = model.encode(["text1", "text2"])
# Returns static lookup results
```

## Troubleshooting

### "Failed to load static embedding model"

**Cause:** Model requires dependencies not installed.

**Solution:**
```bash
# For model2vec models
uv pip install model2vec

# For RikkaBotan MRL models
uv pip install sentence-transformers
```

### "MPS fallback" warning on Apple Silicon

**Cause:** RikkaBotan MRL models use operations not supported by MPS.

**Solution:** Already handled - library sets `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically.

### Model loading errors

**Cause:** Trying to use static model for training without fallback.

**Solution:** Specify a training-compatible model:
```python
# Wrong
matcher = Matcher(model="potion-8m")
matcher.fit(training_data, mode="full")  # Falls back to mpnet

# Right
matcher = Matcher(model="mpnet")  # Training-compatible
matcher.fit(training_data, mode="full")
```

## Next Steps

- See [`models.md`](./models.md) for complete model registry
- See [`matcher-modes.md`](./matcher-modes.md) for mode selection
- See [`experiments/benchmarking.md`](./experiments/benchmarking.md) for performance comparisons
- See [`configuration.md`](./configuration.md) for custom model registration
