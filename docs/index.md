# Documentation Index

This folder is split into stable user guides, experiment docs, and archived project history.

## Guides

- [`quickstart.md`](./quickstart.md): unified `Matcher` class with smart auto-selection
- [`async-guide.md`](./async-guide.md): async/await API for high-concurrency scenarios
- [`examples.md`](./examples.md): maintained example inventory across `examples/current`, `examples/legacy`, and `examples/raw`
- [`troubleshooting.md`](./troubleshooting.md): common install and first-run errors
- [`migration-guide.md`](./migration-guide.md): migrating from deprecated classes
- [`models.md`](./models.md): model registry, aliases, and selection guidance
- [`matcher-modes.md`](./matcher-modes.md): matcher mode system (zero-shot, head-only, full, hybrid)
- [`static-embeddings.md`](./static-embeddings.md): static embedding backend notes
- [`configuration.md`](./configuration.md): configuration system and model registries

## Experiments

- [`experiments/index.md`](./experiments/index.md): experiment inventory and execution conventions
- [`experiments/country-classifier-scripts.md`](./experiments/country-classifier-scripts.md): country classifier experiment walkthrough
- [`experiments/benchmarking.md`](./experiments/benchmarking.md): how to run and interpret benchmarks
- [`experiments/benchmark-results.md`](./experiments/benchmark-results.md): latest published benchmark summary

## Internals

- [`architecture.md`](./architecture.md): module layout and internals
- [`architecture/hierarchical-matching.md`](./architecture/hierarchical-matching.md): hierarchy-specific design notes
- [`bert-classifier.md`](./bert-classifier.md): BERT classifier details
- [`classifier-routes-comparison.md`](./classifier-routes-comparison.md): classifier route tradeoffs

## Archive

- [`archive/index.md`](./archive/index.md): entry point for dated reports, plans, and research

## Reading Paths

### I want to use the library

1. Read [`quickstart.md`](./quickstart.md).
2. If processing large batches (1K+ queries), read [`async-guide.md`](./async-guide.md).
3. Run one of the maintained examples from `examples/current/`.
4. Use [`models.md`](./models.md) and [`matcher-modes.md`](./matcher-modes.md) to refine behavior.
5. Use [`troubleshooting.md`](./troubleshooting.md) if setup/runtime issues appear.

### I want to reproduce experiments

1. Read [`experiments/index.md`](./experiments/index.md).
2. Use the runnable scripts in `experiments/`.
3. Review [`experiments/benchmarking.md`](./experiments/benchmarking.md) or [`experiments/country-classifier-scripts.md`](./experiments/country-classifier-scripts.md) as needed.

### I want lower-level control

1. Read [`examples.md`](./examples.md).
2. Start from `examples/raw/`.
3. Refer to [`architecture.md`](./architecture.md) for project internals.

## Notes

- The package code lives in `src/semanticmatcher/` (src-layout).
- Script experiments live in `experiments/`.
- Local generated outputs should go under `artifacts/`, not the repository root.
