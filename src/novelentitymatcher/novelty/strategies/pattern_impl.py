"""
Pattern-based novelty detection strategy implementation.

Extracts orthographic and linguistic patterns from known entities
and scores novelty based on pattern violations.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


class PatternScorer:
    """
    Detect novel entities using orthographic and linguistic pattern analysis.

    This strategy extracts character-level patterns from known entities and
    scores new entities based on how well they match these patterns. No training
    is required - it's purely rule-based.
    """

    def __init__(self, known_entities: list[str]):
        if not known_entities:
            raise ValueError("known_entities cannot be empty")

        self.known_entities = known_entities
        self.patterns = self._extract_patterns()

    def _extract_patterns(self) -> dict[str, Any]:
        patterns = {
            "char_ngrams": self._get_char_ngrams(self.known_entities, n=3),
            "char_4grams": self._get_char_ngrams(self.known_entities, n=4),
            "has_numbers": self._has_numbers(self.known_entities),
            "capitalization": self._get_capitalization_patterns(self.known_entities),
            "length_range": (
                min(len(e) for e in self.known_entities),
                max(len(e) for e in self.known_entities),
            ),
            "prefix_distribution": self._get_prefix_suffix_distribution(
                self.known_entities, prefix=True
            ),
            "suffix_distribution": self._get_prefix_suffix_distribution(
                self.known_entities, prefix=False
            ),
        }
        return patterns

    def score_novelty(self, entity: str) -> float:
        if not entity:
            return 1.0

        scores = []

        entity_ngrams = self._get_char_ngrams([entity], n=3)
        if len(entity_ngrams) > 0:
            overlap = len(entity_ngrams & self.patterns["char_ngrams"])
            ngram_score = overlap / len(entity_ngrams)
            novelty_from_ngrams = 1.0 - ngram_score
            scores.append(novelty_from_ngrams)

        entity_4grams = self._get_char_ngrams([entity], n=4)
        if len(entity_4grams) > 0:
            overlap = len(entity_4grams & self.patterns["char_4grams"])
            ngram_score = overlap / len(entity_4grams)
            novelty_from_4grams = 1.0 - ngram_score
            scores.append(novelty_from_4grams)

        min_len, max_len = self.patterns["length_range"]
        if max_len > min_len:
            if min_len <= len(entity) <= max_len:
                length_score = 0.0
            else:
                if len(entity) < min_len:
                    length_score = (min_len - len(entity)) / min_len
                else:
                    length_score = (len(entity) - max_len) / max_len
            length_score = min(length_score, 1.0)
            scores.append(length_score)

        entity_cap = self._get_capitalization_patterns([entity])
        if entity_cap in self.patterns["capitalization"]:
            cap_score = 0.0
        else:
            common_patterns = {"title_case", "uppercase", "lowercase", "mixed"}
            cap_score = 0.3 if entity_cap in common_patterns else 0.7
        scores.append(cap_score)

        entity_prefix = entity[:3] if len(entity) >= 3 else entity
        prefix_freq = self.patterns["prefix_distribution"].get(entity_prefix, 0)
        total = sum(self.patterns["prefix_distribution"].values())
        if total > 0:
            prefix_rarity = 1.0 - (prefix_freq / total)
            scores.append(prefix_rarity)

        entity_suffix = entity[-3:] if len(entity) >= 3 else entity
        suffix_freq = self.patterns["suffix_distribution"].get(entity_suffix, 0)
        if total > 0:
            suffix_rarity = 1.0 - (suffix_freq / total)
            scores.append(suffix_rarity)

        if scores:
            weights = [0.25, 0.2, 0.15, 0.1, 0.15, 0.15]
            if len(scores) < len(weights):
                weights = weights[: len(scores)]
                weights = [w / sum(weights) for w in weights]

            novelty_score = sum(s * w for s, w in zip(scores, weights, strict=False))
            return float(np.clip(novelty_score, 0.0, 1.0))

        return 0.5

    def _get_char_ngrams(self, entities: list[str], n: int = 3) -> set[str]:
        ngrams = set()
        for entity in entities:
            entity = entity.strip()
            if len(entity) >= n:
                for i in range(len(entity) - n + 1):
                    ngrams.add(entity[i : i + n])
        return ngrams

    def _has_numbers(self, entities: list[str]) -> float:
        count = sum(1 for e in entities if any(c.isdigit() for c in e))
        return count / len(entities) if entities else 0.0

    def _get_capitalization_patterns(self, entities: list[str]) -> set[str]:
        patterns = set()
        for entity in entities:
            if not entity:
                continue
            if entity.istitle():
                patterns.add("title_case")
            elif entity.isupper():
                patterns.add("uppercase")
            elif entity.islower():
                patterns.add("lowercase")
            else:
                patterns.add("mixed")
        return patterns

    def _get_prefix_suffix_distribution(
        self, entities: list[str], prefix: bool = True, n: int = 3
    ) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for entity in entities:
            entity = entity.strip()
            if len(entity) >= n:
                if prefix:
                    chunk = entity[:n]
                else:
                    chunk = entity[-n:]
                counter[chunk] += 1
        return dict(counter)


def score_batch_novelty(entities: list[str], scorer: PatternScorer) -> list[float]:
    return [scorer.score_novelty(entity) for entity in entities]
