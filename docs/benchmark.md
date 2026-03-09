# Benchmark Results

This document was refreshed on March 9, 2026 after switching the benchmark
suite to use the real processed datasets under `data/processed/` instead of the
old synthetic fixtures.

## What Changed

- Benchmarks now load benchmark sections directly from `data/processed/*/*.csv`
- Each processed dataset is reported as its own `section`
- Current results below come from:
  - `languages/languages`
  - `universities/universities`
- Current model set below includes:
  - `potion-8m`
  - `potion-32m`
  - `mrl-en`
  - `mrl-multi`
  - `minilm`
- The benchmark CLI used is `scripts/benchmark_embeddings.py`

## Commands

```bash
uv run python scripts/benchmark_embeddings.py \
  --track embeddings \
  --sections languages/languages universities/universities \
  --embedding-models potion-8m potion-32m mrl-en mrl-multi minilm \
  --max-entities-per-section 30 \
  --max-queries-per-section 10 \
  --output /tmp/processed-benchmark-refresh.json
```

## Key Findings

1. `potion-8m` remains the fastest model in the measured retrieval set.
2. `potion-32m` also benchmarks very well and stays far ahead of the dynamic
   baseline, but it is still slower than `potion-8m` in both measured sections.
3. On `languages/languages`, `potion-8m` reached `17582.85 qps` and
   `potion-32m` reached `14714.90 qps`, versus `432.41 qps` for `minilm`.
4. On `universities/universities`, `potion-8m` reached `14246.99 qps` and
   `potion-32m` reached `14113.22 qps`, versus `463.91 qps` for `minilm`.
5. With MPS fallback enabled by default, both SSE/MRL models now run:
   `mrl-en` is clearly faster than `minilm`, while `mrl-multi` is faster than
   `minilm` but well behind the potion models.
6. Accuracy was `1.00` for all completed models on these generated top-1 lookup
   queries, so the main difference in this run is latency and throughput.

## Embedding Benchmark

### Section: `languages/languages`

| Model | Backend | Build Time (s) | Cold Query (s) | Avg Latency (s/query) | Throughput (qps) | Accuracy | Speedup vs MiniLM |
|---|---|---:|---:|---:|---:|---:|---:|
| `potion-8m` | static | 1.158 | 0.000905 | 0.000208 | 17177.62 | 1.00 | 39.72x |
| `potion-32m` | static | 1.520 | 0.000442 | 0.000201 | 14714.90 | 1.00 | 34.03x |
| `mrl-en` | static | 3.734 | 0.007973 | 0.002864 | 1957.45 | 1.00 | 4.53x |
| `mrl-multi` | static | 2.789 | 0.008540 | 0.007301 | 1291.12 | 1.00 | 2.99x |
| `minilm` | sentence-transformers | 3.719 | 0.067272 | 0.005633 | 432.41 | 1.00 | 1.00x |

### Section: `universities/universities`

| Model | Backend | Build Time (s) | Cold Query (s) | Avg Latency (s/query) | Throughput (qps) | Accuracy | Speedup vs MiniLM |
|---|---|---:|---:|---:|---:|---:|---:|
| `potion-8m` | static | 1.091 | 0.000360 | 0.000180 | 16822.81 | 1.00 | 36.26x |
| `potion-32m` | static | 1.388 | 0.000395 | 0.000205 | 14113.22 | 1.00 | 30.42x |
| `mrl-en` | static | 3.455 | 0.003809 | 0.003062 | 3074.79 | 1.00 | 6.63x |
| `mrl-multi` | static | 2.811 | 0.009700 | 0.007395 | 1267.58 | 1.00 | 2.73x |
| `minilm` | sentence-transformers | 0.130 | 0.005708 | 0.005550 | 463.91 | 1.00 | 1.00x |

## Comparison Summary

### `potion-8m` vs `minilm`

- `languages/languages`
  - Throughput: `39.7x` faster
  - Average latency: `27.1x` lower
  - Cold query: `74.3x` lower
- `universities/universities`
  - Throughput: `36.3x` faster
  - Average latency: `30.8x` lower
  - Cold query: `15.9x` lower

### `potion-32m` vs `minilm`

- `languages/languages`
  - Throughput: `34.0x` faster
  - Average latency: `28.0x` lower
  - Cold query: `152.2x` lower
- `universities/universities`
  - Throughput: `30.4x` faster
  - Average latency: `27.1x` lower
  - Cold query: `14.4x` lower

### SSE / MRL Models

- `mrl-en` now runs successfully with automatic MPS CPU fallback
- `mrl-multi` also runs successfully with the same fallback behavior
- `mrl-en` throughput:
  - `1957.45 qps` on `languages/languages`
  - `3074.79 qps` on `universities/universities`
- `mrl-multi` throughput:
  - `1291.12 qps` on `languages/languages`
  - `1267.58 qps` on `universities/universities`
- Both MRL models are still faster than `minilm`, but they are materially slower
  than the potion static models on this machine
- Because they rely on MPS fallback for unsupported ops, they may show different
  relative behavior on CPU-only or CUDA environments

## Interpretation

- The move to static embeddings for retrieval is still strongly supported when
  benchmarked on the repo's actual processed datasets.
- The speedup is not tied to one synthetic fixture; it persists across at least
  two very different entity sets and across both potion static models.
- `potion-8m` is still the best speed default among the models that completed.
- `potion-32m` looks like a viable higher-capacity static option if a small
  throughput tradeoff is acceptable.
- `mrl-en` is now a working English MRL option on Apple Silicon, but it is
  closer to a middle tier than a top-speed option in this environment.
- `mrl-multi` is the slowest completed static model in this run, but it still
  outperformed the dynamic MiniLM baseline.
- The benchmark is now section-aware, which makes it easier to compare model
  behavior by dataset family instead of collapsing everything into one number.

## Trained Benchmark Status

The benchmark suite now supports section-based trained benchmarks using the same
processed datasets, but those runs are materially slower because each section
re-trains SetFit models. The document above reflects the latest completed
processed-data embedding run. To refresh the trained section as well, run:

```bash
uv run python scripts/benchmark_embeddings.py \
  --track trained \
  --sections languages/languages universities/universities \
  --training-models minilm mpnet \
  --max-entities-per-section 20 \
  --max-queries-per-section 10 \
  --output /tmp/processed-trained.json
```
