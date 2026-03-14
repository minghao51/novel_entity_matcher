# Async API Guide

The SemanticMatcher now supports async/await operations for high-concurrency scenarios. This guide covers when and how to use the async API.

## When to Use Async

**Use async when:**
- Processing large batches (1K-100K items)
- Running multiple matchers concurrently
- Integrating with async frameworks (FastAPI, asyncio)
- Need progress tracking for long-running operations
- Want cancellation support

**Use sync when:**
- Simple single-match operations
- Small batches (< 100 items)
- Code simplicity is priority
- Not using async frameworks

## Basic Usage

### Single Matcher

```python
import asyncio
from semanticmatcher import Matcher

async def main():
    entities = [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "US", "name": "United States", "aliases": ["USA"]},
    ]

    # Use async context manager for automatic cleanup
    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()
        result = await matcher.match_async("USA")
        print(result)  # {"id": "US", "score": 0.95}

asyncio.run(main())
```

### Batch Processing with Progress

```python
async def process_large_batch():
    entities = [{"id": str(i), "name": f"Product {i}"} for i in range(1000)]
    queries = [f"product {i}" for i in range(10000)]

    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()

        async def show_progress(completed, total):
            percent = (completed / total) * 100
            print(f"Progress: {completed}/{total} ({percent:.1f}%)")

        results = await matcher.match_batch_async(
            queries,
            batch_size=100,
            on_progress=show_progress
        )

    return results
```

### Concurrent Matchers

```python
async def match_concurrently():
    # Multiple matchers running in parallel
    async def match_category(category):
        entities = load_entities_for_category(category)
        async with Matcher(entities=entities) as matcher:
            await matcher.fit_async()
            return await matcher.match_async("query")

    results = await asyncio.gather(
        match_category("products"),
        match_category("users"),
        match_category("locations"),
    )

    return results
```

## API Reference

### Matcher Methods

#### `async fit_async(training_data=None, mode=None, show_progress=True, **kwargs)`

Async version of `fit()`. Trains the matcher if needed.

**Parameters:**
- `training_data`: Optional training examples
- `mode`: Training mode (None, 'zero-shot', 'head-only', 'full', 'hybrid', 'bert')
- `show_progress`: Show progress bar
- `**kwargs`: Additional training arguments (num_epochs, batch_size)

**Returns:** `self` for method chaining

#### `async match_async(texts, top_k=1, **kwargs)`

Async version of `match()`. Match texts against entities.

**Parameters:**
- `texts`: Query text(s) - string or list of strings
- `top_k`: Number of top results to return
- `**kwargs`: Additional arguments (candidates, batch_size)

**Returns:** Match result(s) with scores

#### `async match_batch_async(queries, threshold=None, top_k=1, batch_size=32, on_progress=None, **kwargs)`

Async batch matching with progress tracking.

**Parameters:**
- `queries`: List of query texts
- `threshold`: Optional threshold override
- `top_k`: Number of top results per query
- `batch_size`: Queries per batch
- `on_progress`: Callback(completed, total) for progress updates

**Returns:** List of match results (one per query)

#### `async explain_match_async(query, top_k=5)`

Async version of `explain_match()`. Debug matching results.

#### `async diagnose_async(query)`

Async version of `diagnose()`. Run diagnostics on a query.

### Context Manager

```python
async with Matcher(entities=entities) as matcher:
    # Use matcher
    pass
# Resources automatically cleaned up
```

### Manual Cleanup

```python
matcher = Matcher(entities=entities)
await matcher.fit_async()
try:
    result = await matcher.match_async("query")
finally:
    await matcher.aclose()
```

## Cancellation

Async operations support cancellation:

```python
async def cancellable_match():
    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()

        # Create a long-running task
        task = asyncio.create_task(
            matcher.match_batch_async(large_query_list)
        )

        # Cancel if needed
        if some_condition:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                print("Match was cancelled")
```

## Performance Considerations

### Thread Pool Size

The async executor uses `CPU_COUNT * 2` workers (capped at 32) by default. Adjust if needed:

```python
from semanticmatcher.core.async_utils import AsyncExecutor

# Custom executor
executor = AsyncExecutor(max_workers=16)

# Note: Matcher doesn't yet support custom executors
# This is for illustration only
```

### Batch Size Guidelines

- Small batches (10-50): Low latency, lower throughput
- Medium batches (50-200): Balanced
- Large batches (200-1000): High throughput, higher latency

### Memory Management

For very large datasets (100K+ queries), process in chunks:

```python
async def match_large_dataset(matcher, queries, chunk_size=10000):
    all_results = []
    for i in range(0, len(queries), chunk_size):
        chunk = queries[i:i+chunk_size]
        results = await matcher.match_batch_async(chunk)
        all_results.extend(results)
        # Allow GC to run
        await asyncio.sleep(0)
    return all_results
```

## FastAPI Integration

```python
from fastapi import FastAPI
from semanticmatcher import Matcher

app = FastAPI()
matcher: Matcher = None

@app.on_event("startup")
async def startup():
    global matcher
    entities = load_entities()
    matcher = Matcher(entities=entities)
    await matcher.fit_async()

@app.on_event("shutdown")
async def shutdown():
    global matcher
    if matcher:
        await matcher.aclose()

@app.post("/match")
async def match_query(query: str):
    result = await matcher.match_async(query)
    return {"matched": result}

@app.post("/batch")
async def match_batch(queries: List[str]):
    results = await matcher.match_batch_async(queries)
    return {"results": results}
```

## Migration from Sync

Converting sync code to async:

**Before:**
```python
matcher = Matcher(entities=entities)
matcher.fit()
result = matcher.match("query")
```

**After:**
```python
async with Matcher(entities=entities) as matcher:
    await matcher.fit_async()
    result = await matcher.match_async("query")
```

That's it! The async API mirrors the sync API exactly.

## Best Practices

1. **Always use context managers** - Ensures proper cleanup
2. **Reuse matchers** - Don't recreate for each query
3. **Batch aggressively** - Larger batches = better throughput
4. **Handle cancellation** - Be prepared for CancelledError
5. **Report progress** - Use on_progress for long-running operations

## Troubleshooting

### "RuntimeError: no running event loop"

Make sure you're running async code with `asyncio.run()` or inside an async context.

### "Matcher not ready"

Call `await matcher.fit_async()` before matching.

### High memory usage

Process data in smaller chunks or reduce batch_size.

### Slow performance

- Increase batch_size
- Use static embeddings for faster matching
- Consider hybrid mode for large entity sets
