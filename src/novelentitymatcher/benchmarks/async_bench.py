"""Async/sync performance benchmark for matcher APIs.

Consolidated from: benchmark_async.py

Benchmarks sync vs async matcher APIs across zero-shot, head-only, and full modes,
measuring construct time, fit time, cold-query latency, steady-state match latency,
QPS, and end-to-end wall time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .. import Matcher
from ..utils.benchmarks import load_processed_sections


def _build_queries(seed_queries: List[str], multiplier: int) -> List[str]:
    return seed_queries * multiplier


def _resolve_training_data(
    section_data: Dict[str, Any],
    mode: str,
) -> Optional[List[Dict[str, Any]]]:
    if mode == "zero-shot":
        return None
    return section_data.get("training_data", [])


def _metric(
    *,
    label: str,
    mode: str,
    elapsed: float,
    query_count: int,
    section: str,
    model: str,
    construct_elapsed: float,
    fit_elapsed: float,
    cold_elapsed: float,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    end_to_end = construct_elapsed + fit_elapsed + cold_elapsed + elapsed
    return {
        "route": label,
        "mode": mode,
        "section": section,
        "model": model,
        "construct_seconds": round(construct_elapsed, 6),
        "fit_seconds": round(fit_elapsed, 6),
        "cold_query_seconds": round(cold_elapsed, 6),
        "match_seconds": round(elapsed, 6),
        "end_to_end_seconds": round(end_to_end, 6),
        "queries": query_count,
        "qps": round(query_count / elapsed, 2) if elapsed > 0 else float("inf"),
        "avg_ms_per_query": round((elapsed / query_count) * 1000, 4) if query_count else 0.0,
        "end_to_end_ms_per_query": round((end_to_end / query_count) * 1000, 4) if query_count else 0.0,
        "concurrency": concurrency,
    }


def benchmark_sync(
    entities: List[Dict[str, Any]],
    queries: List[str],
    training_data: Optional[List[Dict[str, Any]]],
    section: str,
    model: str,
    mode: str,
) -> List[Dict[str, Any]]:
    construct_start = time.perf_counter()
    matcher = Matcher(entities=entities, model=model, mode=mode)
    construct_elapsed = time.perf_counter() - construct_start

    fit_start = time.perf_counter()
    if training_data:
        matcher.fit(training_data, mode=mode, show_progress=False)
    else:
        matcher.fit()
    fit_elapsed = time.perf_counter() - fit_start

    cold_start = time.perf_counter()
    matcher.match(queries[0])
    cold_elapsed = time.perf_counter() - cold_start

    start = time.perf_counter()
    [matcher.match(query) for query in queries]
    sync_single = _metric(
        label="sync.match.single",
        mode=mode,
        elapsed=time.perf_counter() - start,
        query_count=len(queries),
        section=section,
        model=model,
        construct_elapsed=construct_elapsed,
        fit_elapsed=fit_elapsed,
        cold_elapsed=cold_elapsed,
    )

    start = time.perf_counter()
    matcher.match(queries)
    sync_bulk = _metric(
        label="sync.match.bulk",
        mode=mode,
        elapsed=time.perf_counter() - start,
        query_count=len(queries),
        section=section,
        model=model,
        construct_elapsed=construct_elapsed,
        fit_elapsed=fit_elapsed,
        cold_elapsed=cold_elapsed,
    )
    return [sync_single, sync_bulk]


async def benchmark_async(
    entities: List[Dict[str, Any]],
    queries: List[str],
    training_data: Optional[List[Dict[str, Any]]],
    concurrency: int,
    section: str,
    model: str,
    mode: str,
) -> List[Dict[str, Any]]:
    construct_start = time.perf_counter()
    matcher = Matcher(entities=entities, model=model, mode=mode)
    construct_elapsed = time.perf_counter() - construct_start

    async with matcher:
        fit_start = time.perf_counter()
        if training_data:
            await matcher.fit_async(training_data, mode=mode, show_progress=False)
        else:
            await matcher.fit_async()
        fit_elapsed = time.perf_counter() - fit_start

        cold_start = time.perf_counter()
        await matcher.match_async(queries[0])
        cold_elapsed = time.perf_counter() - cold_start

        start = time.perf_counter()
        for query in queries:
            await matcher.match_async(query)
        sequential = _metric(
            label="async.match_async.sequential",
            mode=mode,
            elapsed=time.perf_counter() - start,
            query_count=len(queries),
            section=section,
            model=model,
            construct_elapsed=construct_elapsed,
            fit_elapsed=fit_elapsed,
            cold_elapsed=cold_elapsed,
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def run_query(query: str):
            async with semaphore:
                return await matcher.match_async(query)

        start = time.perf_counter()
        await asyncio.gather(*(run_query(query) for query in queries))
        concurrent = _metric(
            label=f"async.match_async.concurrent_{concurrency}",
            mode=mode,
            elapsed=time.perf_counter() - start,
            query_count=len(queries),
            section=section,
            model=model,
            construct_elapsed=construct_elapsed,
            fit_elapsed=fit_elapsed,
            cold_elapsed=cold_elapsed,
            concurrency=concurrency,
        )

        start = time.perf_counter()
        await matcher.match_batch_async(queries, batch_size=min(32, len(queries)))
        batch = _metric(
            label="async.match_batch_async",
            mode=mode,
            elapsed=time.perf_counter() - start,
            query_count=len(queries),
            section=section,
            model=model,
            construct_elapsed=construct_elapsed,
            fit_elapsed=fit_elapsed,
            cold_elapsed=cold_elapsed,
        )

    return [sequential, concurrent, batch]


def _iter_modes(raw_modes: Iterable[str]) -> List[str]:
    valid = []
    seen = set()
    for mode in raw_modes:
        if mode not in {"zero-shot", "head-only", "full"}:
            raise SystemExit(f"Unsupported mode: {mode}")
        if mode not in seen:
            valid.append(mode)
            seen.add(mode)
    return valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiplier", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--section", default="languages/languages")
    parser.add_argument("--model", default="default")
    parser.add_argument("--modes", nargs="+", default=["zero-shot"])
    parser.add_argument("--max-entities", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    sections = load_processed_sections(
        sections=[args.section],
        max_entities_per_section=args.max_entities,
        max_queries_per_section=args.max_queries,
    )
    if not sections:
        raise SystemExit(f"No processed benchmark section found for {args.section}")

    section_data = sections[0]
    queries = _build_queries(section_data["queries"], args.multiplier)
    results = []

    for mode in _iter_modes(args.modes):
        training_data = _resolve_training_data(section_data, mode)
        if mode != "zero-shot" and not training_data:
            raise SystemExit(
                f"No training data available for mode '{mode}' in section {section_data['section']}"
            )

        results.extend(
            benchmark_sync(
                section_data["entities"], queries, training_data,
                section_data["section"], args.model, mode,
            )
        )
        results.extend(
            asyncio.run(
                benchmark_async(
                    section_data["entities"], queries, training_data,
                    args.concurrency, section_data["section"], args.model, mode,
                )
            )
        )

    print(f"Benchmark section: {section_data['section']}")
    print(f"Benchmark model: {args.model}")
    print(f"Benchmark queries: {len(queries)}")
    for result in results:
        print(
            f"{result['mode']} {result['route']}: "
            f"construct={result['construct_seconds']:.4f}s "
            f"fit={result['fit_seconds']:.4f}s "
            f"cold={result['cold_query_seconds']:.4f}s "
            f"match={result['match_seconds']:.4f}s "
            f"end_to_end={result['end_to_end_seconds']:.4f}s "
            f"({result['qps']:.2f} qps)"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved results to {args.output}")

    return 0
