# Models Guide

Related docs: [`index.md`](./index.md) | [`quickstart.md`](./quickstart.md) | [`static-embeddings.md`](./static-embeddings.md) | [`architecture.md`](./architecture.md)

## Overview

Novel Entity Matcher supports multiple embedding models through a registry system. This guide helps you choose the right model for your use case.

## Quick Reference

### By Use Case

| Use Case | Recommended Model | Type | Why |
|---|---|---|---|
| **General English retrieval** | `potion-8m` | Static | Fastest, good accuracy |
| **Multilingual retrieval** | `mrl-multi` or `bge-m3` | Static/Dynamic | Fast multilingual or better quality |
| **Training with data** | `mpnet` | Dynamic | SetFit-compatible, reliable |
| **High accuracy** | `bge-base` | Dynamic | Best contextual understanding |
| **Resource-constrained** | `potion-8m` | Static | Lowest memory/CPU |

## Model Registry

### Static Embedding Models

Pre-computed embeddings for ultra-fast retrieval.

| Alias | Full Model Name | Backend | Language | Training | Notes |
|---|---|---|---|---|---|
| `potion-8m` | minishlab/potion-base-8M | model2vec | en | ❌ | **Default retrieval**, 39x faster than minilm |
| `potion-32m` | minishlab/potion-base-32M | model2vec | en | ❌ | Better quality than potion-8m |
| `mrl-en` | RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en | StaticEmbedding | en | ❌ | MRL support, dimension reduction |
| `mrl-multi` | sentence-transformers/static-similarity-mrl-multilingual-v1 | StaticEmbedding | multilingual | ❌ | Multilingual static |

**See:** [`static-embeddings.md`](./static-embeddings.md) for details on static embeddings.

### Dynamic Embedding Models

Contextual embeddings that support SetFit training.

| Alias | Full Model Name | Language | Training | Notes |
|---|---|---|---|---|
| `bge-base` | BAAI/bge-base-en-v1.5 | en | ✅ | High accuracy English |
| `bge-m3` | BAAI/bge-m3 | multilingual | ✅ | Best multilingual quality |
| `nomic` | nomic-ai/nomic-embed-text-v1 | en | ✅ | Long context support |
| `mpnet` | sentence-transformers/all-mpnet-base-v2 | en | ✅ | **Default training** |
| `minilm` | sentence-transformers/all-MiniLM-L6-v2 | en | ✅ | Fast dynamic baseline |

### Reranker Models

Cross-encoder models for precise candidate reranking (used in hybrid mode).

| Alias | Full Model Name | Notes |
|---|---|---|
| `bge-m3` | BAAI/bge-reranker-v2-m3 | **Default reranker** |
| `bge-large` | BAAI/bge-reranker-large | Higher accuracy, slower |
| `ms-marco` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Lightweight alternative |

## Model Selection Logic

### Automatic Selection

The library has smart defaults based on your workflow:

```python
# Zero-shot mode → static embedding by default
Matcher(mode="zero-shot")  # Uses potion-8m

# Training mode → training-compatible model
Matcher(mode="full")  # Uses mpnet for training

# Explicit model selection
Matcher(model="bge-m3")  # Uses specified model
```

### Resolution Priority

1. **User-specified model** - If you provide a model name, it's used directly
2. **Default for mode** - Zero-shot uses static, training uses dynamic
3. **Fallback** - Static models for training auto-fallback to mpnet

### Example Resolution

```python
# Input: potion-8m
# Output: minishlab/potion-base-8M (model2vec backend)

# Input: default (zero-shot)
# Output: minishlab/potion-base-8M (retrieval default)

# Input: default (training)
# Output: sentence-transformers/all-mpnet-base-v2 (training default)

# Input: potion-8m (with training)
# Output: sentence-transformers/all-mpnet-base-v2 (auto-fallback)
```

## Using Model Aliases

### Short Aliases (Recommended)

```python
# Static embeddings
Matcher(model="potion-8m")
Matcher(model="mrl-en")

# Dynamic embeddings
Matcher(model="mpnet")
Matcher(model="bge-base")
Matcher(model="bge-m3")
```

### Full Model Names

```python
# Equivalent to above
Matcher(model="minishlab/potion-base-8M")
Matcher(model="RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en")
Matcher(model="sentence-transformers/all-mpnet-base-v2")
Matcher(model="BAAI/bge-base-en-v1.5")
Matcher(model="BAAI/bge-m3")
```

### Custom Models

Use any HuggingFace sentence-transformer model:

```python
Matcher(model="your-org/your-model")
```

**Note:** Custom models must be sentence-transformer compatible.

## Performance Characteristics

### Latency Comparison

Lower is better (single query latency):

| Model | Latency | Relative to minilm |
|---|---|---|
| potion-8m | ~0.25ms | **39x faster** |
| potion-32m | ~0.29ms | 34x faster |
| mrl-en | ~0.56ms | 17x faster |
| minilm | ~10ms | baseline |
| bge-base | ~25ms | 2.5x slower |
| mpnet | ~30ms | 3x slower |

*Results from [`experiments/benchmarking.md`](./experiments/benchmarking.md) - varies by hardware.*

### Memory Usage

Approximate model memory footprint:

| Model | Memory | Notes |
|---|---|---|
| potion-8m | ~30MB | Smallest |
| potion-32m | ~120MB | Small |
| minilm | ~100MB | Small |
| mpnet | ~400MB | Medium |
| bge-base | ~400MB | Medium |
| bge-m3 | ~2GB | Large (multilingual) |

## Accuracy vs Speed Tradeoffs

### Retrieval (Zero-Shot)

| Priority | Model | Why |
|---|---|---|
| **Speed** | potion-8m | 39x faster than minilm |
| **Balance** | minilm | Good accuracy, reasonable speed |
| **Accuracy** | bge-base | Best contextual understanding |

### Training (Few-Shot)

| Priority | Model | Why |
|---|---|---|
| **Speed** | mpnet | Good balance of training speed and accuracy |
| **Accuracy** | bge-base | Best fine-tuning results |
| **Multilingual** | bge-m3 | Supports 100+ languages |

## Multilingual Models

### Static Multilingual

```python
# Fast multilingual retrieval
Matcher(model="mrl-multi", mode="zero-shot")
```

- Supports 50+ languages
- Static embeddings (fast)
- Slight accuracy tradeoff vs dynamic

### Dynamic Multilingual

```python
# High-quality multilingual
Matcher(model="bge-m3", mode="zero-shot")
```

- Supports 100+ languages
- Better accuracy than static
- Slower but more contextual

**Language Support Coverage:**
- `mrl-multi`: 50+ languages (major world languages)
- `bge-m3`: 100+ languages (broader coverage)

## Training Compatibility

### Why Static Models Don't Support Training

Static embeddings use pre-computed lookup tables without trainable parameters:

```python
# This works for retrieval
Matcher(model="potion-8m", mode="zero-shot")  # ✅ OK

# This triggers auto-fallback
Matcher(model="potion-8m", mode="full")  # ⚠️ Falls back to mpnet
```

**Why:** SetFit requires a trainable embedding backbone (SentenceTransformer).

### Training-Compatible Models

All dynamic models support SetFit training:

```python
# All of these work for training
Matcher(model="mpnet", mode="full")      # ✅
Matcher(model="bge-base", mode="full")   # ✅
Matcher(model="bge-m3", mode="full")     # ✅
Matcher(model="minilm", mode="full")     # ✅
```

## Custom Model Registration

Add your own models to the registry:

```python
from novelentitymatcher.config import MODEL_SPECS

MODEL_SPECS["my-model"] = {
    "name": "my-org/my-model",
    "backend": "sentence-transformers",
    "supports_training": True,
    "language": "en",
}

# Now you can use the alias
Matcher(model="my-model")
```

## Recommendations by Scenario

### High-Throughput API

```python
matcher = Matcher(entities=entities, model="potion-8m")
# 4000+ queries per second
```

### Batch Processing

```python
matcher = Matcher(entities=entities, model="minilm")
# Good balance of speed and accuracy
```

### Research & Experimentation

```python
matcher = Matcher(entities=entities, model="bge-base")
# Best accuracy for analysis
```

### Production Training

```python
matcher = Matcher(entities=entities, model="mpnet")
matcher.fit(training_data, mode="full")
# Reliable training with good results
```

### Multilingual Production

```python
# Fast multilingual
matcher = Matcher(entities=entities, model="mrl-multi")

# High-quality multilingual
matcher = Matcher(entities=entities, model="bge-m3")
```

## Troubleshooting

### Model Download Failures

**Cause:** Network issues or model not found.

**Solution:**
```python
# Check model alias is valid
from novelentitymatcher.config import MODEL_SPECS
print(MODEL_SPECS.get("your-model"))

# Try full model name
Matcher(model="org/model-name")
```

### CUDA/MPS Memory Errors

**Cause:** Model too large for GPU memory.

**Solution:**
```python
# Use smaller model
Matcher(model="minilm")  # ~100MB vs 400MB for mpnet

# Or force CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### Training Not Improving

**Cause:** Using static model or insufficient training data.

**Solution:**
```python
# Ensure training-compatible model
from novelentitymatcher.config import supports_training_model
print(supports_training_model("your-model"))  # Should be True

# Add more training examples (aim for 3+ per entity)
matcher.fit(training_data, mode="full", num_epochs=4)
```

## Next Steps

- See [`static-embeddings.md`](./static-embeddings.md) for static model details
- See [`matcher-modes.md`](./matcher-modes.md) for mode selection
- See [`experiments/benchmarking.md`](./experiments/benchmarking.md) for performance benchmarks
- See [`configuration.md`](./configuration.md) for custom model registration
