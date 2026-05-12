"""Benchmark utilities for comparing retrieval and trained matching models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
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
from .preprocessing import clean_text

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


def parse_aliases(raw_aliases: str) -> list[str]:
    if not raw_aliases:
        return []
    return [alias.strip() for alias in raw_aliases.split("|") if alias.strip()]


def dataset_section_name(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def row_to_entity(
    row: dict[str, str],
    alias_counts: Counter[str] | None = None,
) -> dict[str, Any]:
    aliases = parse_aliases(row.get("aliases", ""))
    if alias_counts is not None:
        aliases = [
            alias
            for alias in aliases
            if alias_counts.get(alias, 0) == 1 and alias != row["id"]
        ]
    entity: dict[str, Any] = {"id": row["id"], "name": row["name"]}
    if aliases:
        entity["aliases"] = aliases
    if row.get("type"):
        entity["type"] = row["type"]
    return entity


def dedupe_texts(entity: dict[str, Any], include_id: bool = True) -> list[str]:
    texts = [entity["name"], *entity.get("aliases", [])]
    if include_id:
        texts.append(entity["id"])
    unique_texts = []
    for text in texts:
        if text and text not in unique_texts:
            unique_texts.append(text)
    return unique_texts


def normalize_eval_text(text: str) -> str:
    return clean_text(text, lowercase=True, remove_punct=True)


def remove_parenthetical(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def first_clause(text: str) -> str:
    return re.split(r"[,;]", text, maxsplit=1)[0]


def introduce_typo(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text

    longest_idx = max(range(len(tokens)), key=lambda idx: len(tokens[idx]))
    token = tokens[longest_idx]
    if len(token) < 5:
        return text

    midpoint = len(token) // 2
    typo = token[:midpoint] + token[midpoint + 1 :]
    if typo == token:
        return text

    tokens[longest_idx] = typo
    return " ".join(tokens)


def generate_holdout_queries(
    source_text: str,
    excluded_texts: Iterable[str],
) -> list[tuple[str, str]]:
    excluded = {normalize_eval_text(text) for text in excluded_texts if text}
    candidates: list[tuple[str, str]] = []
    raw_candidates = [
        ("typo", normalize_eval_text(introduce_typo(normalize_eval_text(source_text)))),
        (
            "remove_parenthetical",
            normalize_eval_text(remove_parenthetical(source_text)),
        ),
        ("ampersand_expanded", normalize_eval_text(source_text.replace("&", " and "))),
        ("first_clause", normalize_eval_text(first_clause(source_text))),
        ("normalized_verbatim", normalize_eval_text(source_text)),
    ]

    seen_candidates = set()
    for label, candidate in raw_candidates:
        if not candidate or candidate in excluded or candidate in seen_candidates:
            continue
        candidates.append((label, candidate))
        seen_candidates.add(candidate)

    return candidates


def build_split_pairs(entity: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    indexed_texts = dedupe_texts(entity, include_id=False)
    empty_split: dict[str, list[dict[str, Any]]] = {
        "base": [],
        "train": [],
        "val": [],
        "test": [],
        "typo": [],
        "remove_parenthetical": [],
        "ampersand_expanded": [],
        "first_clause": [],
        "normalized_verbatim": [],
    }
    if not indexed_texts:
        return empty_split

    base_query = indexed_texts[0]
    base_pairs = [{"query": base_query, "expected_id": entity["id"]}]
    train_pairs = [
        {"query": text, "expected_id": entity["id"]}
        for text in indexed_texts[1:3]
        if text != base_query
    ]

    holdout_source = indexed_texts[-1]
    holdout_queries = generate_holdout_queries(holdout_source, indexed_texts)
    perturbation_pairs = {
        label: [{"query": query, "expected_id": entity["id"]}]
        for label, query in holdout_queries
    }
    val_pairs = (
        [{"query": holdout_queries[0][1], "expected_id": entity["id"]}]
        if len(holdout_queries) >= 1
        else []
    )
    test_pairs = (
        [{"query": holdout_queries[1][1], "expected_id": entity["id"]}]
        if len(holdout_queries) >= 2
        else []
    )

    return {
        "base": base_pairs,
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
        "typo": perturbation_pairs.get("typo", []),
        "remove_parenthetical": perturbation_pairs.get("remove_parenthetical", []),
        "ampersand_expanded": perturbation_pairs.get("ampersand_expanded", []),
        "first_clause": perturbation_pairs.get("first_clause", []),
        "normalized_verbatim": perturbation_pairs.get("normalized_verbatim", []),
    }


def select_primary_queries(split_pairs: dict[str, list[dict[str, Any]]]) -> list[str]:
    for split in ("test", "val", "train", "base"):
        pairs = split_pairs.get(split, [])
        if pairs:
            return [pair["query"] for pair in pairs]
    return []


def iter_processed_dataset_paths(
    processed_dir: Path = PROCESSED_DATA_DIR,
    sections: Iterable[str] | None = None,
) -> list[Path]:
    selected = set(sections) if sections else None
    paths = sorted(processed_dir.glob("*/*.csv"))
    if selected is None:
        return paths
    return [path for path in paths if dataset_section_name(path) in selected]


def load_processed_sections(
    processed_dir: Path = PROCESSED_DATA_DIR,
    sections: Iterable[str] | None = None,
    max_entities_per_section: int = 200,
    max_queries_per_section: int = 50,
) -> list[dict[str, Any]]:
    loaded_sections: list[dict[str, Any]] = []

    for path in iter_processed_dataset_paths(
        processed_dir=processed_dir, sections=sections
    ):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        alias_counts: Counter[str] = Counter()
        for row in rows:
            alias_counts.update(parse_aliases(row.get("aliases", "")))

        entities = []
        training_data = []
        base_pairs: list[dict[str, Any]] = []
        train_pairs: list[dict[str, Any]] = []
        val_pairs: list[dict[str, Any]] = []
        test_pairs: list[dict[str, Any]] = []
        perturbation_pairs: dict[str, list[dict[str, Any]]] = {
            "typo": [],
            "remove_parenthetical": [],
            "ampersand_expanded": [],
            "first_clause": [],
            "normalized_verbatim": [],
        }

        for row in rows[:max_entities_per_section]:
            entity = row_to_entity(row, alias_counts=alias_counts)
            entities.append(entity)
            split_pairs = build_split_pairs(entity)

            if len(base_pairs) < max_queries_per_section:
                base_pairs.extend(split_pairs["base"])
            if len(train_pairs) < max_queries_per_section:
                train_pairs.extend(split_pairs["train"])
            if len(val_pairs) < max_queries_per_section:
                val_pairs.extend(split_pairs["val"])
            if len(test_pairs) < max_queries_per_section:
                test_pairs.extend(split_pairs["test"])
            for label in perturbation_pairs:
                if len(perturbation_pairs[label]) < max_queries_per_section:
                    perturbation_pairs[label].extend(split_pairs[label])

            training_texts = [base_query["query"] for base_query in split_pairs["base"]]
            training_texts.extend(pair["query"] for pair in split_pairs["train"])
            for text in training_texts[:3]:
                training_data.append({"text": text, "label": entity["id"]})

        split_map = {
            "base": base_pairs[:max_queries_per_section],
            "train": train_pairs[:max_queries_per_section],
            "val": val_pairs[:max_queries_per_section],
            "test": test_pairs[:max_queries_per_section],
        }
        queries = select_primary_queries(split_map)
        accuracy_pairs = split_map["base"]
        evaluation_pairs = split_map["test"] or split_map["val"]

        if not entities or not queries:
            continue

        loaded_sections.append(
            {
                "section": dataset_section_name(path),
                "path": path,
                "entities": entities,
                "queries": queries,
                "accuracy_pairs": accuracy_pairs,
                "training_data": training_data[: max_queries_per_section * 3],
                "evaluation_pairs": evaluation_pairs[:max_queries_per_section],
                "base_pairs": split_map["base"],
                "train_pairs": split_map["train"],
                "val_pairs": split_map["val"],
                "test_pairs": split_map["test"],
                "typo_pairs": perturbation_pairs["typo"][:max_queries_per_section],
                "remove_parenthetical_pairs": perturbation_pairs[
                    "remove_parenthetical"
                ][:max_queries_per_section],
                "ampersand_expanded_pairs": perturbation_pairs["ampersand_expanded"][
                    :max_queries_per_section
                ],
                "first_clause_pairs": perturbation_pairs["first_clause"][
                    :max_queries_per_section
                ],
                "normalized_verbatim_pairs": perturbation_pairs["normalized_verbatim"][
                    :max_queries_per_section
                ],
            }
        )

    return loaded_sections


def build_embedding_benchmark_dataset(
    num_entities: int = 200,
    num_queries: int = 50,
) -> dict[str, Any]:
    section = load_processed_sections(
        max_entities_per_section=num_entities,
        max_queries_per_section=num_queries,
    )[0]
    return {
        "entities": section["entities"],
        "queries": section["queries"],
        "accuracy_pairs": section["accuracy_pairs"],
    }


def build_trained_benchmark_dataset() -> dict[str, Any]:
    sections = load_processed_sections(
        max_entities_per_section=40, max_queries_per_section=20
    )
    section = next(
        (
            item
            for item in sections
            if item["training_data"] and item["evaluation_pairs"]
        ),
        None,
    )
    if section is None:
        raise ValueError(
            "No processed sections contain a non-overlapping train/eval split"
        )
    return {
        "entities": section["entities"],
        "training_data": section["training_data"],
        "evaluation_pairs": section["evaluation_pairs"],
        "queries": [pair["query"] for pair in section["evaluation_pairs"]],
    }


def _training_texts_from_split_pairs(
    split_pairs: dict[str, list[dict[str, Any]]],
) -> list[str]:
    training_texts = [pair["query"] for pair in split_pairs.get("base", [])]
    training_texts.extend(pair["query"] for pair in split_pairs.get("train", []))
    deduped: list[str] = []
    for text in training_texts:
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _first_available_pairs(
    split_pairs: dict[str, list[dict[str, Any]]],
    preferred_splits: tuple[str, ...],
) -> list[dict[str, Any]]:
    for split_name in preferred_splits:
        pairs = split_pairs.get(split_name, [])
        if pairs:
            return pairs
    return []


def build_processed_ood_sections(
    processed_dir: Path = PROCESSED_DATA_DIR,
    sections: Iterable[str] | None = None,
    max_entities_per_section: int = 200,
    max_queries_per_section: int = 50,
    ood_ratio: float = 0.2,
    min_known_classes: int = 3,
) -> list[dict[str, Any]]:
    loaded_sections: list[dict[str, Any]] = []

    for section in load_processed_sections(
        processed_dir=processed_dir,
        sections=sections,
        max_entities_per_section=max_entities_per_section,
        max_queries_per_section=max_queries_per_section,
    ):
        entities = section["entities"]
        if len(entities) < (min_known_classes + 1):
            continue

        entity_splits = [(entity, build_split_pairs(entity)) for entity in entities]
        entity_splits = [
            (entity, split_pairs)
            for entity, split_pairs in entity_splits
            if dedupe_texts(entity, include_id=False)
        ]
        if len(entity_splits) < (min_known_classes + 1):
            continue

        holdout_count = max(1, math.ceil(len(entity_splits) * ood_ratio))
        holdout_count = min(holdout_count, len(entity_splits) - min_known_classes)
        if holdout_count <= 0:
            continue

        known_items = entity_splits[:-holdout_count]
        heldout_items = entity_splits[-holdout_count:]
        if len(known_items) < min_known_classes or not heldout_items:
            continue

        known_entities: list[dict[str, Any]] = []
        training_data: list[dict[str, str]] = []
        known_val_pairs: list[dict[str, Any]] = []
        known_test_pairs: list[dict[str, Any]] = []
        novel_val_pairs: list[dict[str, Any]] = []
        novel_test_pairs: list[dict[str, Any]] = []

        for entity, split_pairs in known_items:
            known_entities.append(entity)
            for text in _training_texts_from_split_pairs(split_pairs)[:3]:
                training_data.append({"text": text, "label": entity["id"]})

            for pair in _first_available_pairs(split_pairs, ("val", "train", "base"))[
                :1
            ]:
                known_val_pairs.append(
                    {
                        "query": pair["query"],
                        "expected_id": entity["id"],
                        "label": entity["id"],
                        "is_novel": False,
                        "split": "val_known",
                    }
                )
            for pair in _first_available_pairs(
                split_pairs, ("test", "val", "train", "base")
            )[:1]:
                known_test_pairs.append(
                    {
                        "query": pair["query"],
                        "expected_id": entity["id"],
                        "label": entity["id"],
                        "is_novel": False,
                        "split": "test_known",
                    }
                )

        for entity, split_pairs in heldout_items:
            for pair in _first_available_pairs(split_pairs, ("val", "train", "base"))[
                :1
            ]:
                novel_val_pairs.append(
                    {
                        "query": pair["query"],
                        "expected_id": entity["id"],
                        "label": entity["id"],
                        "is_novel": True,
                        "split": "val_novel",
                    }
                )
            for pair in _first_available_pairs(
                split_pairs, ("test", "val", "train", "base")
            )[:1]:
                novel_test_pairs.append(
                    {
                        "query": pair["query"],
                        "expected_id": entity["id"],
                        "label": entity["id"],
                        "is_novel": True,
                        "split": "test_novel",
                    }
                )

        if not known_entities or not training_data:
            continue
        if not known_val_pairs or not novel_val_pairs:
            continue
        if not known_test_pairs or not novel_test_pairs:
            continue

        loaded_sections.append(
            {
                "section": section["section"],
                "path": section["path"],
                "track": "ood_novelty",
                "known_entities": known_entities,
                "training_data": training_data[: max_queries_per_section * 3],
                "known_class_ids": [entity["id"] for entity in known_entities],
                "heldout_class_ids": [entity["id"] for entity, _ in heldout_items],
                "val_known_pairs": known_val_pairs[:max_queries_per_section],
                "val_novel_pairs": novel_val_pairs[:max_queries_per_section],
                "test_known_pairs": known_test_pairs[:max_queries_per_section],
                "test_novel_pairs": novel_test_pairs[:max_queries_per_section],
            }
        )

    return loaded_sections


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def save_benchmark_report(results: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(results.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
    elif path.suffix.lower() == ".csv":
        results.to_csv(path, index=False)
    else:
        raise ValueError("output_path must end with .json or .csv")

    return path


def format_benchmark_summary(results: pd.DataFrame) -> str:
    if results.empty:
        return "No benchmark results collected."

    lines = ["BENCHMARK RESULTS"]
    for track in results["track"].dropna().unique():
        track_subset = results[results["track"] == track]
        lines.append("")
        lines.append(f"[{track}]")
        for section in track_subset["section"].dropna().unique():
            section_subset = track_subset[track_subset["section"] == section]
            lines.append(f"<section: {section}>")
            if track == "ood_novelty":
                columns = [
                    col
                    for col in [
                        "model",
                        "selected_threshold",
                        "num_threshold_candidates",
                        "num_known_classes",
                        "num_heldout_classes",
                        "validation_novel_f1",
                        "validation_known_accuracy",
                        "validation_false_positive_novel_rate",
                        "novel_precision",
                        "novel_recall",
                        "novel_f1",
                        "known_accuracy",
                        "false_positive_novel_rate",
                        "overall_accuracy",
                        "artifact_path",
                    ]
                    if col in section_subset.columns
                ]
            elif (
                "mode" in section_subset.columns
                and section_subset["mode"].notna().any()
            ):
                columns = [
                    col
                    for col in [
                        "mode",
                        "model",
                        "status",
                        "throughput_qps",
                        "base_accuracy",
                        "train_accuracy",
                        "val_accuracy",
                        "test_accuracy",
                        "skip_reason",
                    ]
                    if col in section_subset.columns
                ]
            else:
                columns = [
                    col
                    for col in [
                        "model",
                        "backend",
                        "status",
                        "throughput_qps",
                        "base_accuracy",
                        "train_accuracy",
                        "val_accuracy",
                        "test_accuracy",
                        "speedup_vs_minilm",
                        "skip_reason",
                    ]
                    if col in section_subset.columns
                ]
            lines.append(section_subset[columns].to_string(index=False))
            lines.append("")
    return "\n".join(line for line in lines if line is not None).rstrip()


def print_benchmark_report(results: pd.DataFrame):
    logger.info(format_benchmark_summary(results))


def parse_benchmark_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark semantic matcher models")
    parser.add_argument(
        "--track",
        choices=("embeddings", "trained", "all"),
        default="all",
        help="Benchmark embeddings, trained modes, or both",
    )
    parser.add_argument(
        "--output",
        help="Optional output path (.json or .csv) for the combined results",
    )
    parser.add_argument(
        "--embedding-models",
        nargs="*",
        default=None,
        help="Optional subset of embedding aliases to benchmark",
    )
    parser.add_argument(
        "--training-models",
        nargs="*",
        default=None,
        help="Optional subset of training-compatible aliases to benchmark",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="Optional subset of processed-data sections such as languages/languages",
    )
    parser.add_argument(
        "--max-entities-per-section",
        type=int,
        default=200,
        help="Maximum entities loaded from each processed dataset section",
    )
    parser.add_argument(
        "--max-queries-per-section",
        type=int,
        default=50,
        help="Maximum benchmark queries generated per processed dataset section",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

try:
    from tqdm.auto import tqdm
except ImportError:

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
            except Exception as exc:
                logger.warning("Embedding benchmark skipped for %s: %s", alias, exc)
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
                except Exception as exc:
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


__all__ = [
    "format_benchmark_summary",
    "print_benchmark_report",
    "save_benchmark_report",
]


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
    args = parse_benchmark_args(argv)
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
