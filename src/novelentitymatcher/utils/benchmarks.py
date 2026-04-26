"""Benchmark utilities for comparing retrieval and trained matching models."""

from __future__ import annotations

import time
from collections.abc import Iterable
from statistics import mean
from typing import Any

import pandas as pd

from novelentitymatcher.utils.logging_config import get_logger

from ..config import (
    get_embedding_model_aliases,
    get_model_spec,
    get_training_model_aliases,
)
from ..core.matcher import EmbeddingMatcher, Matcher
from .benchmark_dataset import load_processed_sections
from .benchmark_reporting import (
    format_benchmark_summary,
    print_benchmark_report,
    save_benchmark_report,
)
from .benchmark_reporting import (
    parse_benchmark_args as _parse_args,
)

logger = get_logger(__name__)

__all__ = [
    "format_benchmark_summary",
    "print_benchmark_report",
    "save_benchmark_report",
]

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency

    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable


def _top_level_match_id(result: Any) -> str | None:
    if isinstance(result, dict):
        return result.get("id")
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first.get("id")
    return None


def benchmark_accuracy(
    matcher: Any,
    test_pairs: list[dict[str, Any]],
) -> dict[str, float]:
    correct = 0
    scores = []

    for pair in test_pairs:
        result = matcher.match(pair["query"])
        if _top_level_match_id(result) == pair["expected_id"]:
            correct += 1
        if isinstance(result, dict):
            scores.append(float(result.get("score", 0.0)))

    return {
        "accuracy": correct / len(test_pairs) if test_pairs else 0.0,
        "avg_score": mean(scores) if scores else 0.0,
        "total_pairs": len(test_pairs),
    }


def benchmark_latency(
    matcher: Any,
    queries: list[str],
    iterations: int = 5,
    warmup_iterations: int = 1,
) -> dict[str, float]:
    for _ in range(warmup_iterations):
        for query in queries:
            matcher.match(query)

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        for query in queries:
            matcher.match(query)
        elapsed = time.perf_counter() - start
        timings.append(elapsed / len(queries))

    timings_sorted = sorted(timings)
    return {
        "avg_time": mean(timings),
        "min_time": min(timings),
        "max_time": max(timings),
        "p50_time": timings_sorted[len(timings_sorted) // 2],
        "p95_time": timings_sorted[
            min(len(timings_sorted) - 1, int(len(timings_sorted) * 0.95))
        ],
        "p99_time": timings_sorted[
            min(len(timings_sorted) - 1, int(len(timings_sorted) * 0.99))
        ],
        "total_time": sum(timings) * len(queries),
    }


def compare_models(
    entities: list[dict[str, Any]],
    queries: list[str],
    model_names: list[str],
    num_iterations: int = 3,
) -> pd.DataFrame:
    accuracy_pairs = [
        {"query": query, "expected_id": entities[index]["id"]}
        for index, query in enumerate(queries[: len(entities)])
    ]
    return benchmark_embedding_models(
        entities=entities,
        queries=queries,
        accuracy_pairs=accuracy_pairs,
        model_names=model_names,
        iterations=num_iterations,
    )


def _split_accuracy_fields(
    matcher: Any,
    split_pairs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}

    metric_splits = (
        "base",
        "train",
        "val",
        "test",
        "typo",
        "remove_parenthetical",
        "ampersand_expanded",
        "first_clause",
        "normalized_verbatim",
    )
    for split_name in metric_splits:
        metrics = benchmark_accuracy(matcher, split_pairs.get(split_name, []))
        fields[f"{split_name}_accuracy"] = metrics["accuracy"]
        fields[f"{split_name}_avg_score"] = metrics["avg_score"]
        fields[f"{split_name}_total_pairs"] = metrics["total_pairs"]

    for preferred in ("test", "val", "train", "base"):
        if fields[f"{preferred}_total_pairs"] > 0:
            fields["accuracy"] = fields[f"{preferred}_accuracy"]
            fields["avg_score"] = fields[f"{preferred}_avg_score"]
            fields["accuracy_split"] = preferred
            break
    else:
        fields["accuracy"] = 0.0
        fields["avg_score"] = 0.0
        fields["accuracy_split"] = "none"

    return fields


def benchmark_embedding_models(
    entities: list[dict[str, Any]] | None = None,
    queries: list[str] | None = None,
    accuracy_pairs: list[dict[str, Any]] | None = None,
    model_names: list[str] | None = None,
    iterations: int = 3,
    batch_size: int | None = None,
    sections_data: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    model_names = model_names or get_embedding_model_aliases()
    if sections_data is None:
        if entities is None or queries is None or accuracy_pairs is None:
            sections_data = load_processed_sections()
        else:
            sections_data = [
                {
                    "section": "custom",
                    "entities": entities,
                    "queries": queries,
                    "accuracy_pairs": accuracy_pairs,
                }
            ]

    records: list[dict[str, Any]] = []

    for section_data in sections_data:
        section_name = section_data["section"]
        section_entities = section_data["entities"]
        section_queries = section_data["queries"]
        split_pairs = {
            "base": section_data.get(
                "base_pairs", section_data.get("accuracy_pairs", [])
            ),
            "train": section_data.get("train_pairs", []),
            "val": section_data.get("val_pairs", []),
            "test": section_data.get("test_pairs", []),
            "typo": section_data.get("typo_pairs", []),
            "remove_parenthetical": section_data.get("remove_parenthetical_pairs", []),
            "ampersand_expanded": section_data.get("ampersand_expanded_pairs", []),
            "first_clause": section_data.get("first_clause_pairs", []),
            "normalized_verbatim": section_data.get("normalized_verbatim_pairs", []),
        }

        for alias in tqdm(model_names, desc=f"Embedding benchmarks [{section_name}]"):
            spec = get_model_spec(alias) or {}
            backend = spec.get("backend", "sentence-transformers")

            try:
                matcher = EmbeddingMatcher(
                    section_entities, model_name=alias, threshold=0.0
                )
                build_start = time.perf_counter()
                matcher.build_index(batch_size=batch_size)
                build_time = time.perf_counter() - build_start

                cold_start = time.perf_counter()
                matcher.match(section_queries[0], batch_size=batch_size)
                cold_query_time = time.perf_counter() - cold_start

                latency = benchmark_latency(
                    matcher, section_queries, iterations=iterations, warmup_iterations=1
                )
                accuracy_fields = _split_accuracy_fields(matcher, split_pairs)

                bulk_times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    matcher.match(section_queries, batch_size=batch_size)
                    bulk_times.append(time.perf_counter() - start)
                avg_bulk_time = mean(bulk_times)

                records.append(
                    {
                        "track": "embedding",
                        "section": section_name,
                        "model": alias,
                        "resolved_model": matcher.model_name,
                        "backend": backend,
                        "status": "ok",
                        "build_time": build_time,
                        "cold_query_time": cold_query_time,
                        "avg_latency": latency["avg_time"],
                        "p95_latency": latency["p95_time"],
                        "throughput_qps": len(section_queries) / avg_bulk_time
                        if avg_bulk_time
                        else 0.0,
                        "bulk_time": avg_bulk_time,
                        **accuracy_fields,
                        "skip_reason": "",
                    }
                )
            except Exception as exc:  # pragma: no cover - exercised via monkeypatch
                logger.warning("Trained benchmark skipped for %s: %s", alias, exc)
                spec = get_model_spec(alias) or {}
                records.append(
                    {
                        "track": "trained",
                        "section": section_name,
                        "model": alias,
                        "resolved_model": spec.get("name", alias) if spec else alias,
                        "status": "skipped",
                        "build_time": None,
                        "cold_query_time": None,
                        "avg_latency": None,
                        "p95_latency": None,
                        "throughput_qps": None,
                        "bulk_time": None,
                        "accuracy": None,
                        "avg_score": None,
                        "accuracy_split": None,
                        "base_accuracy": None,
                        "train_accuracy": None,
                        "val_accuracy": None,
                        "test_accuracy": None,
                        "base_avg_score": None,
                        "train_avg_score": None,
                        "val_avg_score": None,
                        "test_avg_score": None,
                        "base_total_pairs": None,
                        "train_total_pairs": None,
                        "val_total_pairs": None,
                        "test_total_pairs": None,
                        "typo_accuracy": None,
                        "remove_parenthetical_accuracy": None,
                        "ampersand_expanded_accuracy": None,
                        "first_clause_accuracy": None,
                        "normalized_verbatim_accuracy": None,
                        "typo_avg_score": None,
                        "remove_parenthetical_avg_score": None,
                        "ampersand_expanded_avg_score": None,
                        "first_clause_avg_score": None,
                        "normalized_verbatim_avg_score": None,
                        "typo_total_pairs": None,
                        "remove_parenthetical_total_pairs": None,
                        "ampersand_expanded_total_pairs": None,
                        "first_clause_total_pairs": None,
                        "normalized_verbatim_total_pairs": None,
                        "skip_reason": str(exc),
                    }
                )

        section_baseline = next(
            (
                row["throughput_qps"]
                for row in records
                if row["section"] == section_name
                and row["status"] == "ok"
                and row["model"] == "minilm"
            ),
            None,
        )
        for row in records:
            if row["section"] != section_name:
                continue
            if section_baseline and row["status"] == "ok" and row["throughput_qps"]:
                row["speedup_vs_minilm"] = row["throughput_qps"] / section_baseline
            else:
                row["speedup_vs_minilm"] = None

    return pd.DataFrame(records)


def benchmark_trained_modes(
    entities: list[dict[str, Any]] | None = None,
    training_data: list[dict[str, Any]] | None = None,
    evaluation_pairs: list[dict[str, Any]] | None = None,
    queries: list[str] | None = None,
    model_names: list[str] | None = None,
    modes: Iterable[str] | None = None,
    num_epochs: int = 1,
    sections_data: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    model_names = model_names or get_training_model_aliases()
    modes = list(modes or ("head-only", "full"))
    if sections_data is None:
        if (
            entities is None
            or training_data is None
            or evaluation_pairs is None
            or queries is None
        ):
            sections_data = load_processed_sections(
                max_entities_per_section=40, max_queries_per_section=20
            )
        else:
            sections_data = [
                {
                    "section": "custom",
                    "entities": entities,
                    "training_data": training_data,
                    "evaluation_pairs": evaluation_pairs,
                    "queries": queries,
                }
            ]

    records: list[dict[str, Any]] = []

    for section_data in sections_data:
        section_name = section_data["section"]
        section_entities = section_data["entities"]
        section_training = section_data["training_data"]
        section_queries = section_data["queries"]
        split_pairs = {
            "base": section_data.get("base_pairs", []),
            "train": section_data.get("train_pairs", []),
            "val": section_data.get("val_pairs", []),
            "test": section_data.get(
                "test_pairs", section_data.get("evaluation_pairs", [])
            ),
            "typo": section_data.get("typo_pairs", []),
            "remove_parenthetical": section_data.get("remove_parenthetical_pairs", []),
            "ampersand_expanded": section_data.get("ampersand_expanded_pairs", []),
            "first_clause": section_data.get("first_clause_pairs", []),
            "normalized_verbatim": section_data.get("normalized_verbatim_pairs", []),
        }

        if (
            not section_training
            or not (split_pairs["val"] or split_pairs["test"])
            or not section_queries
        ):
            continue

        for alias in tqdm(model_names, desc=f"Training benchmarks [{section_name}]"):
            for mode in modes:
                try:
                    matcher = Matcher(
                        entities=section_entities,
                        model=alias,
                        mode=mode,
                        threshold=0.0,
                    )
                    train_start = time.perf_counter()
                    matcher.fit(
                        section_training,
                        num_epochs=num_epochs,
                        show_progress=False,
                    )
                    train_time = time.perf_counter() - train_start

                    latency = benchmark_latency(
                        matcher, section_queries, iterations=3, warmup_iterations=1
                    )
                    accuracy_fields = _split_accuracy_fields(matcher, split_pairs)

                    bulk_times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        matcher.match(section_queries)
                        bulk_times.append(time.perf_counter() - start)
                    avg_bulk_time = mean(bulk_times)

                    records.append(
                        {
                            "track": "trained",
                            "section": section_name,
                            "mode": mode,
                            "model": alias,
                            "resolved_model": matcher.entity_matcher.model_name,
                            "status": "ok",
                            "training_time": train_time,
                            "avg_latency": latency["avg_time"],
                            "p95_latency": latency["p95_time"],
                            "throughput_qps": len(section_queries) / avg_bulk_time
                            if avg_bulk_time
                            else 0.0,
                            **accuracy_fields,
                            "skip_reason": "",
                        }
                    )
                except Exception as exc:  # pragma: no cover - exercised via monkeypatch
                    logger.warning("Trained benchmark skipped for %s: %s", alias, exc)
                    spec = get_model_spec(alias)
                    records.append(
                        {
                            "track": "trained",
                            "section": section_name,
                            "mode": mode,
                            "model": alias,
                            "resolved_model": spec.get("name", alias)
                            if spec
                            else alias,
                            "status": "skipped",
                            "training_time": None,
                            "avg_latency": None,
                            "p95_latency": None,
                            "throughput_qps": None,
                            "accuracy": None,
                            "avg_score": None,
                            "accuracy_split": None,
                            "base_accuracy": None,
                            "train_accuracy": None,
                            "val_accuracy": None,
                            "test_accuracy": None,
                            "base_avg_score": None,
                            "train_avg_score": None,
                            "val_avg_score": None,
                            "test_avg_score": None,
                            "base_total_pairs": None,
                            "train_total_pairs": None,
                            "val_total_pairs": None,
                            "test_total_pairs": None,
                            "typo_accuracy": None,
                            "remove_parenthetical_accuracy": None,
                            "ampersand_expanded_accuracy": None,
                            "first_clause_accuracy": None,
                            "normalized_verbatim_accuracy": None,
                            "typo_avg_score": None,
                            "remove_parenthetical_avg_score": None,
                            "ampersand_expanded_avg_score": None,
                            "first_clause_avg_score": None,
                            "normalized_verbatim_avg_score": None,
                            "typo_total_pairs": None,
                            "remove_parenthetical_total_pairs": None,
                            "ampersand_expanded_total_pairs": None,
                            "first_clause_total_pairs": None,
                            "normalized_verbatim_total_pairs": None,
                            "skip_reason": str(exc),
                        }
                    )

    return pd.DataFrame(records)


def run_benchmark_suite(
    track: str = "all",
    embedding_models: list[str] | None = None,
    training_models: list[str] | None = None,
    output_path: str | None = None,
    sections: list[str] | None = None,
    max_entities_per_section: int = 200,
    max_queries_per_section: int = 50,
) -> dict[str, pd.DataFrame]:
    suite: dict[str, pd.DataFrame] = {}
    loaded_sections = load_processed_sections(
        sections=sections,
        max_entities_per_section=max_entities_per_section,
        max_queries_per_section=max_queries_per_section,
    )

    if track in ("embeddings", "all"):
        suite["embeddings"] = benchmark_embedding_models(
            model_names=embedding_models,
            sections_data=loaded_sections,
        )

    if track in ("trained", "all"):
        suite["trained"] = benchmark_trained_modes(
            model_names=training_models,
            sections_data=loaded_sections,
        )

    if output_path:
        combined = (
            pd.concat(suite.values(), ignore_index=True) if suite else pd.DataFrame()
        )
        save_benchmark_report(combined, output_path)

    return suite


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    suite = run_benchmark_suite(
        track=args.track,
        embedding_models=args.embedding_models,
        training_models=args.training_models,
        output_path=args.output,
        sections=args.sections,
        max_entities_per_section=args.max_entities_per_section,
        max_queries_per_section=args.max_queries_per_section,
    )

    if args.track in ("embeddings", "all") and "embeddings" in suite:
        print_benchmark_report(suite["embeddings"])
    if args.track in ("trained", "all") and "trained" in suite:
        logger.info("")
        print_benchmark_report(suite["trained"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
