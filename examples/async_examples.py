"""
Async API Examples for SemanticMatcher

This file demonstrates various async usage patterns.
Run with: uv run python examples/async_examples.py
"""

import asyncio
from semanticmatcher import Matcher


# Sample entities
SAMPLE_ENTITIES = [
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
    {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
    {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
    {"id": "JP", "name": "Japan", "aliases": ["日本"]},
]


async def example_basic_usage():
    """Basic async matcher usage."""
    print("\n=== Basic Usage ===")

    async with Matcher(entities=SAMPLE_ENTITIES) as matcher:
        await matcher.fit_async()

        # Single match
        result = await matcher.match_async("USA")
        print(f"Match: {result}")

        # Multiple matches
        results = await matcher.match_async(["USA", "Germany", "Japan"])
        print(f"Batch: {results}")


async def example_with_progress():
    """Batch matching with progress tracking."""
    print("\n=== Batch with Progress ===")

    # Create a larger entity set
    entities = [{"id": str(i), "name": f"Product {i}"} for i in range(100)]
    queries = [f"product {i}" for i in range(1000)]

    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()

        # Progress callback
        async def show_progress(completed, total):
            percent = (completed / total) * 100
            if completed % 100 == 0:  # Print every 100
                print(f"Progress: {completed}/{total} ({percent:.1f}%)")

        results = await matcher.match_batch_async(
            queries,
            batch_size=50,
            on_progress=show_progress
        )

        print(f"Matched {len(results)} queries")


async def example_concurrent_matchers():
    """Multiple matchers running concurrently."""
    print("\n=== Concurrent Matchers ===")

    async def match_category(category_name, entities):
        async with Matcher(entities=entities) as matcher:
            await matcher.fit_async()
            result = await matcher.match_async(entities[0]["name"])
            return category_name, result

    # Run multiple matchers in parallel
    results = await asyncio.gather(
        match_category("Countries", SAMPLE_ENTITIES),
        match_category("Products", [
            {"id": "1", "name": "Laptop"},
            {"id": "2", "name": "Mouse"},
        ]),
        match_category("Cities", [
            {"id": "1", "name": "Berlin"},
            {"id": "2", "name": "Paris"},
        ]),
    )

    for category, result in results:
        print(f"{category}: {result}")


async def example_with_training():
    """Async matcher with training data."""
    print("\n=== With Training ===")

    entities = [
        {"id": "tech", "name": "Technology"},
        {"id": "finance", "name": "Finance"},
        {"id": "health", "name": "Healthcare"},
    ]

    training_data = [
        {"text": "software", "label": "tech"},
        {"text": "hardware", "label": "tech"},
        {"text": "banking", "label": "finance"},
        {"text": "investment", "label": "finance"},
        {"text": "hospital", "label": "health"},
        {"text": "medicine", "label": "health"},
    ]

    async with Matcher(entities=entities) as matcher:
        # Train with head-only mode (fast)
        await matcher.fit_async(training_data, mode="head-only", num_epochs=1)

        # Test
        result = await matcher.match_async("software development")
        print(f"Matched: {result}")


async def example_explain_and_diagnose():
    """Using explain_match_async and diagnose_async."""
    print("\n=== Explain and Diagnose ===")

    async with Matcher(entities=SAMPLE_ENTITIES) as matcher:
        await matcher.fit_async()

        # Explain match
        explanation = await matcher.explain_match_async("USA", top_k=3)
        print(f"Explanation: {explanation['best_match']}")
        print(f"Top matches: {[r['id'] for r in explanation['top_k']]}")

        # Diagnose
        diagnosis = await matcher.diagnose_async("United States")
        print(f"Diagnosis: matched={diagnosis['matched']}")


async def example_cancellation():
    """Demonstrating cancellation support."""
    print("\n=== Cancellation ===")

    entities = [{"id": str(i), "name": f"Entity {i}"} for i in range(100)]

    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()

        # Create a long-running task
        queries = [f"Entity {i}" for i in range(10000)]
        task = asyncio.create_task(
            matcher.match_batch_async(queries, batch_size=10)
        )

        # Cancel after a short delay
        await asyncio.sleep(0.01)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            print("Task was successfully cancelled")


async def example_threshold_override():
    """Using threshold override in batch operations."""
    print("\n=== Threshold Override ===")

    async with Matcher(entities=SAMPLE_ENTITIES, threshold=0.8) as matcher:
        await matcher.fit_async()

        # High threshold - fewer matches
        results_strict = await matcher.match_batch_async(
            ["USA", "Deutschland", "Unknown Country"],
            threshold=0.9
        )
        print(f"Strict threshold: {sum(1 for r in results_strict if r)} matches")

        # Low threshold - more matches
        results_lenient = await matcher.match_batch_async(
            ["USA", "Deutschland", "Unknown Country"],
            threshold=0.5
        )
        print(f"Lenient threshold: {sum(1 for r in results_lenient if r)} matches")


async def main():
    """Run all examples."""
    print("SemanticMatcher Async API Examples")
    print("=" * 50)

    await example_basic_usage()
    await example_with_progress()
    await example_concurrent_matchers()
    await example_with_training()
    await example_explain_and_diagnose()
    await example_cancellation()
    await example_threshold_override()

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
