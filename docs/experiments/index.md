# Experiments Index

Related docs: [`../index.md`](../index.md) | [`country-classifier-scripts.md`](./country-classifier-scripts.md) | [`benchmarking.md`](./benchmarking.md)

This section documents exploratory assets under `experiments/`.

## Scope

- `experiments/` is for script-style exploratory work that is not part of the stable package API.
- There are currently no tracked Jupyter notebooks in this repository.

## How to Run

Run from the repository root:

```bash
uv run python experiments/<path_to_script>.py
```

The current experiment scripts inject `src/` automatically, so `PYTHONPATH=.` is no longer required.

## Script Inventory

| File | Purpose | Runtime | Notes |
|---|---|---|---|
| `experiments/country_classifier/country_classifier.py` | Baseline country-code classifier comparison (A/B/C scenarios) | Minutes (CPU/GPU dependent) | Best first experiment |
| `experiments/country_classifier/country_classifier_quick.py` | Fast optimization checks beyond baseline | Several minutes | Quick iteration path |
| `experiments/country_classifier/country_classifier_advanced.py` | Broader optimization search across models/heads | Longer (~10-15+ min) | Advanced tuning |

For a deeper explanation of the country classifier scripts, see [`country-classifier-scripts.md`](./country-classifier-scripts.md).
