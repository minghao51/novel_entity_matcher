# Benchmark Organization

## Current Structure

The benchmark code is organized into three layers:

### 1. Utility Functions (`src/novelentitymatcher/utils/`)
Internal helper functions used by benchmark scripts:
- `benchmarks.py` - Core benchmark utilities
- `benchmark_dataset.py` - Dataset loading for benchmarks
- `benchmark_reporting.py` - Report formatting and CLI helpers

These are **internal utilities** not exported in the public API.

### 2. Benchmark Scripts (`scripts/`)
Executable scripts that run benchmarks:
- `benchmark_embeddings.py` - Embedding model benchmarks
- `benchmark_async.py` - Async API benchmarks
- `benchmark_bert.py` - BERT classifier benchmarks
- `visualize_benchmarks.py` - Visualization tools
- `render_benchmark_report.py` - Report generation

### 3. Results (`artifacts/benchmarks/`)
Benchmark output data:
- JSON result files
- Generated reports

## Design Rationale

**Why keep utilities in `src/novelentitymatcher/utils/`?**

1. **Internal vs. External**: Benchmark utilities are internal helpers, not user-facing features
2. **Import Path**: Scripts in `scripts/` can import from `novelentitymatcher.utils.benchmark*`
3. **Testing**: Tests are already in `tests/test_utils/test_benchmarks.py`
4. **Clarity**: The `utils/` directory clearly indicates these are helper functions

**Why not create a top-level `benchmarks/` directory?**

1. **Scope**: These are utilities, not standalone benchmarks
2. **Imports**: Would require updating all import paths in scripts/
3. **Consistency**: Other utilities (validation, logging, etc.) are in `utils/`
4. **Separation**: Actual benchmark scripts are already separated in `scripts/`

## Usage

### Running Benchmarks

```bash
# Embedding benchmarks
uv run python scripts/benchmark_embeddings.py

# Async benchmarks
uv run python scripts/benchmark_async.py

# BERT benchmarks
uv run python scripts/benchmark_bert.py
```

### Using Utilities Directly

```python
from novelentitymatcher.utils.benchmarks import benchmark_accuracy
from novelentitymatcher.utils.benchmark_dataset import load_processed_sections
from novelentitymatcher.utils.benchmark_reporting import format_benchmark_summary
```

## Future Considerations

If the benchmark utilities become part of the public API, consider:
1. Creating `novelentitymatcher.benchmarks` module
2. Exporting in main `__init__.py`
3. Adding documentation for public benchmark API

For now, the current structure is optimal for internal use.
