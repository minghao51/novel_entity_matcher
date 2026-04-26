# Experiments Index

Related docs: [`../index.md`](../index.md) | [`country-classifier-scripts.md`](./country-classifier-scripts.md) | [`benchmarking.md`](./benchmarking.md) | [`benchmark-results.md`](./benchmark-results.md) | [`novelty-detection-benchmark.md`](./novelty-detection-benchmark.md) | [`speed-benchmark-results.md`](./speed-benchmark-results.md)

This section documents exploratory assets and benchmark results under `experiments/`.

## Benchmark Docs

| File | Content |
|------|---------|
| [`benchmarking.md`](./benchmarking.md) | How to run benchmarks via `novelentitymatcher-bench` CLI |
| [`benchmark-results.md`](./benchmark-results.md) | ER, classification, and novelty detection results |
| [`novelty-detection-benchmark.md`](./novelty-detection-benchmark.md) | Novelty strategy results (traditional + SetFit-based) |
| [`speed-benchmark-results.md`](./speed-benchmark-results.md) | Sync vs async route speed results |

## Experiment Scripts

| File | Purpose | Runtime | Notes |
|---|---|---|---|
| `experiments/country_classifier/country_classifier.py` | Baseline country-code classifier comparison (A/B/C scenarios) | Minutes (CPU/GPU dependent) | Best first experiment |
| `experiments/country_classifier/country_classifier_quick.py` | Fast optimization checks beyond baseline | Several minutes | Quick iteration path |
| `experiments/country_classifier/country_classifier_advanced.py` | Broader optimization search across models/heads | Longer (~10-15+ min) | Advanced tuning |

For a deeper explanation of the country classifier scripts, see [`country-classifier-scripts.md`](./country-classifier-scripts.md).

## How to Run

```bash
uv run python experiments/<path_to_script>.py
```

The current experiment scripts inject `src/` automatically, so `PYTHONPATH=.` is no longer required.
