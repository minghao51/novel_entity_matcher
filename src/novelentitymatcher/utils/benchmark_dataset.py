"""Dataset preparation helpers for semantic matcher benchmarks."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .preprocessing import clean_text

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
