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
          └─ 10k+ entities → [Hybrid](hybrid.md)
```

---

## Each method page contains:

- **Mathematical Formulation**: Equations and formulas
- **DAG**: Inputs, components/techniques, outputs
- **Approach/Methodology**: Process and key design decisions
- **Findings**: Benchmark performance, when to use, when not to use, characteristics, strengths, weaknesses

---

**Related**: [Novelty Detection Methods](../novelty-detection/) | [Methodology Overview](../overview.md)
