# Speed Benchmark Results

Related docs: [`benchmarking.md`](./benchmarking.md) | [`benchmark-results.md`](./benchmark-results.md)

This document captures a route-level speed comparison between the sync and async
matcher APIs on the real `products/products_mcc` processed-data section.

The benchmark includes:
- `zero-shot`
- `head-only`
- `full`

It reports separate timings for matcher construction, fit/training, cold query,
steady-state matching, and end-to-end wall time.

This document was refreshed on March 14, 2026.

## Command

```bash
uv run novelentitymatcher-bench bench-async \
  --section products/products_mcc \
  --model default \
  --modes zero-shot head-only full \
  --max-entities 50 \
  --max-queries 25 \
  --multiplier 20 \
  --concurrency 8 \
  --output artifacts/benchmarks/speed-routes-products-mcc.json
```

## Key Findings

- `zero-shot` remains in a completely different speed tier from trained modes:
  `sync.match.bulk` reached `49,670.59 qps` and `async.match_batch_async`
  reached `34,530.44 qps`.
- `head-only` and `full` have very similar steady-state inference throughput on
  this workload, both landing around `90-104 qps`.
- The real difference between trained modes here is setup cost:
  `head-only` fit ranged from `87.73s` to `184.05s`, while `full` fit ranged
  from `127.91s` to `132.38s`.
- For async integration, `match_batch_async()` remains the best async route, but
  the trained-mode end-to-end cost is dominated by training rather than by route
  selection.

## `products/products_mcc`

| Mode | Route | Construct (s) | Fit/Train (s) | Cold Query (s) | Match (s) | End-to-End (s) | Throughput (qps) | Match Avg (ms/query) | End-to-End Avg (ms/query) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `zero-shot` | `sync.match.single` | `0.0003` | `1.1894` | `0.0013` | `0.0306` | `1.2216` | `5884.40` | `0.1699` | `6.7866` |
| `zero-shot` | `sync.match.bulk` | `0.0003` | `1.1894` | `0.0013` | `0.0036` | `1.1946` | `49670.59` | `0.0201` | `6.6368` |
| `zero-shot` | `async.match_async.sequential` | `0.0009` | `1.1022` | `0.0004` | `0.0395` | `1.1431` | `4552.03` | `0.2197` | `6.3503` |
| `zero-shot` | `async.match_async.concurrent_8` | `0.0009` | `1.1022` | `0.0004` | `0.0423` | `1.1458` | `4260.19` | `0.2347` | `6.3654` |
| `zero-shot` | `async.match_batch_async` | `0.0009` | `1.1022` | `0.0004` | `0.0052` | `1.1087` | `34530.44` | `0.0290` | `6.1596` |
| `head-only` | `sync.match.single` | `0.0002` | `87.7325` | `0.1120` | `2.1270` | `89.9716` | `84.63` | `11.8165` | `499.8424` |
| `head-only` | `sync.match.bulk` | `0.0002` | `87.7325` | `0.1120` | `1.7507` | `89.5954` | `102.82` | `9.7262` | `497.7521` |
| `head-only` | `async.match_async.sequential` | `0.0004` | `184.0497` | `0.0297` | `1.9778` | `186.0576` | `91.01` | `10.9878` | `1033.6535` |
| `head-only` | `async.match_async.concurrent_8` | `0.0004` | `184.0497` | `0.0297` | `1.7896` | `185.8695` | `100.58` | `9.9424` | `1032.6081` |
| `head-only` | `async.match_batch_async` | `0.0004` | `184.0497` | `0.0297` | `1.7269` | `185.8067` | `104.23` | `9.5940` | `1032.2597` |
| `full` | `sync.match.single` | `0.0004` | `132.3848` | `0.0269` | `1.9176` | `134.3297` | `93.87` | `10.6532` | `746.2763` |
| `full` | `sync.match.bulk` | `0.0004` | `132.3848` | `0.0269` | `1.7503` | `134.1625` | `102.84` | `9.7241` | `745.3471` |
| `full` | `async.match_async.sequential` | `0.0004` | `127.9124` | `0.0321` | `1.9819` | `129.9268` | `90.82` | `11.0108` | `721.8156` |
| `full` | `async.match_async.concurrent_8` | `0.0004` | `127.9124` | `0.0321` | `1.8276` | `129.7725` | `98.49` | `10.1536` | `720.9584` |
| `full` | `async.match_batch_async` | `0.0004` | `127.9124` | `0.0321` | `1.7336` | `129.6785` | `103.83` | `9.6312` | `720.4360` |

## Interpretation

- If you care about pure retrieval speed, `zero-shot` is still the practical
  default by a very wide margin.
- If you need supervised behavior, choose between `head-only` and `full`
  primarily on quality and training-budget grounds, not inference speed, because
  their steady-state route throughput is now very similar in this benchmark.
- `sync.match.bulk` and `async.match_batch_async` remain the right throughput
  routes for batched workloads across all modes.
- End-to-end cost changes the story substantially: trained modes can spend two
  to three orders of magnitude more time in setup/training than in the actual
  match loop.
