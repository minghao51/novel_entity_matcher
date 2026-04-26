# Benchmarking Guide

Related docs: [`../index.md`](../index.md) | [`../models.md`](../models.md) | [`../static-embeddings.md`](../static-embeddings.md) | [`benchmark-results.md`](./benchmark-results.md) | [`speed-benchmark-results.md`](./speed-benchmark-results.md) | [`novelty-detection-benchmark.md`](./novelty-detection-benchmark.md)

## Overview

Novel Entity Matcher includes a comprehensive benchmarking suite accessed via the `novelentitymatcher-bench` CLI. It covers accuracy, latency, throughput, and novelty detection across multiple strategies and datasets.

All benchmarks live in `src/novelentitymatcher/benchmarks/` and are registered as the `novelentitymatcher-bench` entry point in `pyproject.toml`.

## CLI Subcommands

| Subcommand | Purpose |
|------------|---------|
| `run` | Run entity resolution, classification, and novelty benchmarks on HuggingFace datasets |
| `bench-classifier` | Benchmark BERT vs SetFit classifiers head-to-head |
| `bench-novelty` | Benchmark novelty detection strategies at quick/standard/full depth |
| `bench-async` | Benchmark sync vs async matcher API throughput |
| `render` | Render benchmark JSON as markdown tables |
| `plot` | Generate charts from benchmark JSON |
| `load` | Download/cache HuggingFace datasets |
| `list` | List available datasets |
| `clear` | Clear cached datasets |
| `sweep` | Parameter sweep (threshold, k, distance) |

## Running Benchmarks

### HuggingFace Benchmark Suite (`run`)

```bash
# Run all benchmarks (ER + classification + novelty)
uv run novelentitymatcher-bench run --task all --models potion-8m

# Entity resolution only
uv run novelentitymatcher-bench run --task er --models potion-8m --thresholds 0.5 0.7 0.9

# Classification with trained modes
uv run novelentitymatcher-bench run \
  --task classification \
  --models all-MiniLM-L6-v2 \
  --modes zero-shot head-only \
  --class-counts 4 10 28 \
  --max-train-samples 200

# Novelty detection
uv run novelentitymatcher-bench run --task novelty --ood-ratio 0.2

# Save results to JSON
uv run novelentitymatcher-bench run --task all --output data/hf_benchmarks/results.json
```

### Classifier Comparison (`bench-classifier`)

```bash
# BERT vs SetFit head-to-head
uv run novelentitymatcher-bench bench-classifier --mode compare

# Multi-model BERT sweep
uv run novelentitymatcher-bench bench-classifier \
  --mode sweep-models \
  --models distilbert tinybert roberta-base \
  --num-entities 10 --num-samples 50

# Save results
uv run novelentitymatcher-bench bench-classifier --mode compare --output /tmp/clf_results.md
```

### Novelty Strategy Benchmark (`bench-novelty`)

Three depth levels control strategy coverage:

| Depth | Strategies |
|-------|-----------|
| `quick` | KNN, Mahalanobis, LOF, OneClassSVM, IsolationForest |
| `standard` | quick + Pattern, SetFit Centroid, ensembles |
| `full` | standard + SignalCombiner, meta-learner |

```bash
# Quick benchmark (fastest)
uv run novelentitymatcher-bench bench-novelty --depth quick

# Standard with specific datasets
uv run novelentitymatcher-bench bench-novelty \
  --depth standard \
  --datasets ag_news go_emotions \
  --max-train 200 --max-test 500

# Full depth
uv run novelentitymatcher-bench bench-novelty --depth full --output /tmp/novelty_results.csv
```

### Async Speed Benchmark (`bench-async`)

```bash
uv run novelentitymatcher-bench bench-async \
  --section languages/languages \
  --model default \
  --modes zero-shot \
  --max-entities 50 \
  --max-queries 25 \
  --multiplier 20 \
  --concurrency 8 \
  --output artifacts/benchmarks/speed-routes.json
```

### Rendering and Plotting

```bash
# Render benchmark JSON as markdown
uv run novelentitymatcher-bench render artifacts/benchmarks/results.json

# Generate charts from benchmark results
uv run novelentitymatcher-bench plot \
  --embedding-results results/embeddings.json \
  --training-results results/training.json \
  --bert-results results/bert.json \
  --output-dir docs/images/benchmarks
```

### Dataset Management

```bash
# List available datasets
uv run novelentitymatcher-bench list

# Download/cache specific datasets
uv run novelentitymatcher-bench load --datasets ag_news go_emotions

# Force re-download
uv run novelentitymatcher-bench load --datasets ag_news --force

# Clear cache
uv run novelentitymatcher-bench clear --dataset ag_news
uv run novelentitymatcher-bench clear
```

## Understanding the Output

### Console Output

```
BENCHMARK RESULTS

[embedding]
<section: languages/languages>
           model backend  status  throughput_qps  accuracy_split  base_accuracy  val_accuracy  test_accuracy
potion-8m    static       ok        4032.12       val             0.9500         0.7250        1.0000
minilm       dynamic      ok         102.45       val             0.9500         0.7750        0.9474
bge-base     dynamic      ok          41.23       val             0.9600         0.7900        0.9500
```

**Key metrics:**
- `throughput_qps` — Queries per second (higher is better)
- `accuracy` — Top-1 accuracy on the preferred populated split
- `accuracy_split` — Which split that top-line accuracy came from
- Perturbation metrics such as `typo_accuracy` — robustness by transformation type
- `speedup_vs_minilm` — Relative speed vs minilm baseline
- `status` — "ok" or "skipped" (with skip_reason)

### Novelty Benchmark Output

```
Strategy                   Val AUROC   Test AUROC   Test DR@1%
------------------------------------------------------------
knn_distance                   0.875        0.851        0.160
mahalanobis                    0.826        0.822        0.090
lof                            0.817        0.799        0.250
oneclass_svm                   0.830        0.825        0.290
isolation_forest               0.644        0.539        0.050
```

## Benchmark Metrics Explained

### Throughput (QPS)

Queries per second — how many matches the system can process.

- **Higher is better**
- **potion-8m**: ~4000 QPS (39x faster than minilm)
- **minilm**: ~100 QPS (baseline)
- **bge-base**: ~40 QPS (2.5x slower than minilm)

### Accuracy

Top-1 match accuracy — percentage of queries that match the correct entity.

- Typical range: 0.80–0.95 (80–95%)
- Tradeoff with speed: static models trade slight accuracy for huge speed gains

### Latency

Time per query — `avg_latency`, `p95_latency`, `p99_latency`.

### AUROC (Novelty)

Area Under ROC Curve — overall discrimination ability for novelty detection. 1.0 = perfect, 0.5 = random.

### DR@1% (Novelty)

Detection Rate at 1% False Positive — what fraction of novel samples are caught when only 1% of known samples are incorrectly flagged. Measures practical detection capability.

## Benchmark Datasets

### HuggingFace Datasets (`run` command)

| Task | Datasets |
|------|----------|
| Entity Resolution | walmart_amazon, amazon_google, fodors_zagats, beer, dblp_acm, dblp_googlescholar, itunes_amazon |
| Classification | ag_news, yahoo_answers, goemotions |
| Novelty Detection | ag_news, goemotions (with 20% OOD class split) |

Datasets are cached as parquet at `data/hf_benchmarks/`.

### Benchmark Download Security

Some entity-resolution benchmark sources currently resolve to legacy `http://` URLs.
The benchmark loader now emits a warning when insecure transport is used.

Migration guidance:
- Prefer HTTPS mirrors for benchmark assets whenever available.
- Update `download_url` values in dataset registry entries to trusted HTTPS sources.
- Treat HTTP benchmark downloads as non-production and integrity-risky until migrated.

### Processed Sections (`bench-async`)

Custom CSV sections in `data/processed/*/*.csv`:

```
data/processed/
├── languages/
│   └── languages.csv
├── universities/
│   └── universities.csv
└── currencies/
    └── currencies.csv
```

CSV columns: `id`, `name`, `aliases` (pipe-separated), `type` (optional).

## Interpreting Results for Model Selection

### Speed-Critical: `potion-8m`
- 39x faster than minilm, minimal accuracy tradeoff (~92% vs 93%)
- Use for high-traffic APIs (>1000 req/s), tight latency budgets (<10ms)

### Accuracy-Critical: `bge-base`
- Highest accuracy (~94–95%), better contextual understanding
- Use when accuracy is paramount, lower traffic volumes

### Balanced: `minilm`
- Good accuracy (~93%), reasonable speed (~100 QPS)
- Safe default for moderate traffic

### Multilingual: `mrl-multi` or `bge-m3`
- Static (fast) or dynamic (accurate) multilingual options

## Programmatic Usage

```python
from novelentitymatcher.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()

# Load datasets
runner.load_all()

# Run specific benchmarks
er_results = runner.run_entity_resolution_benchmark(model="potion-8m")
clf_results = runner.run_classification(model="potion-8m", mode="zero-shot")
novelty_results = runner.run_novelty(model="potion-8m", ood_ratio=0.2)

# Run everything
all_results = runner.run_all()
```

## Troubleshooting

### "No benchmark sections found"
No processed data in `data/processed/`. Check with `ls data/processed/*/*.csv` or specify datasets explicitly.

### Model loading errors
Test model loading: `from novelentitymatcher import Matcher; m = Matcher(model="your-model", entities=[{"id":"1","name":"test"}]); m.fit()`

### Out of memory
Benchmark one model at a time or reduce data: `--max-train-samples 100`.

## Next Steps

- See [`benchmark-results.md`](./benchmark-results.md) for latest published results
- See [`speed-benchmark-results.md`](./speed-benchmark-results.md) for route speed results
- See [`novelty-detection-benchmark.md`](./novelty-detection-benchmark.md) for novelty detection strategy results
- See [`../models.md`](../models.md) for model selection guidance
- See [`../matcher-modes.md`](../matcher-modes.md) for mode selection
