# Classification Methods

Individual documentation for each classification method in Phase 1 of the Novel Entity Matcher pipeline.

---

## Methods

| Method | Benchmark Accuracy | Training Time | Best For |
|--------|-------------------|---------------|----------|
| [SetFit Full](setfit-full.md) | **91.2%** | ~64s | Production, ≥ 3 examples/entity |
| [SetFit Head-Only](setfit-head-only.md) | 54.7% | ~5s | Few-shot, < 3 examples/entity |
| [Zero-Shot](zero-shot.md) | 73.3% | ~3s (index) | No training data |
| [BERT](bert.md) | 88-98% | ~5min | High-stakes, ≥ 100 examples |
| [Hybrid](hybrid.md) | 90-95% | None (index) | 10k+ entities |
| Auto | — | — | Automatic mode selection based on data |

### Auto Mode

When `mode="auto"` is specified, the system automatically selects the optimal classification mode based on training data characteristics:

```
No training data → zero-shot
< 3 examples/entity → head-only
≥ 3 examples/entity, < 100 total → full (SetFit)
≥ 100 total, ≥ 8 examples/entity → bert
```

This is the recommended default for users who don't want to manually select a mode.

---

## Selection Guide

```
Do you have training data?
│
├─ No → [Zero-Shot](zero-shot.md)
│
└─ Yes → How many examples per entity?
          │
          ├─ < 3 → [SetFit Head-Only](setfit-head-only.md)
          │
          ├─ ≥ 3, < 100 total → [SetFit Full](setfit-full.md) ← RECOMMENDED
          │
          ├─ ≥ 100 total, ≥ 8 per entity → [BERT](bert.md)
          │
          ├─ 10k+ entities → [Hybrid](hybrid.md)
          │
          └─ Unsure → Use auto mode
```

---

## Model Backends

The system supports three embedding model backends, selected based on mode and model choice:

| Backend | Models | Training Support | Use Case |
|---------|--------|-----------------|----------|
| **static** | potion-8m, potion-32m, mrl-en, mrl-multi | No | Fast retrieval, zero-shot |
| **sentence-transformers** | bge-base, bge-m3, nomic, mpnet, minilm | Yes | SetFit training, general use |
| **bert** | distilbert, tinybert, roberta-base, deberta-v3, bert-multilingual | Yes | BERT classification mode |

---

## Each method page contains:

- **Mathematical Formulation**: Equations and formulas
- **DAG**: Inputs, components/techniques, outputs
- **Approach/Methodology**: Process and key design decisions
- **Findings**: Benchmark performance, when to use, when not to use, characteristics, strengths, weaknesses

---

**Related**: [Novelty Detection Methods](../novelty-detection/) | [Methodology Overview](../overview.md)
