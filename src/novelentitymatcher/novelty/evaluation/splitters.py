"""
Data splitters for novelty detection evaluation.

Provides utilities for creating OOD (Out-of-Distribution) splits
and gradual novelty scenarios for testing.
"""

from typing import Any

import numpy as np


class OODSplitter:
    """
    Creates OOD (Out-of-Distribution) splits for novelty detection evaluation.

    Splits data into known classes and unknown/novel classes to simulate
    the novelty detection scenario.
    """

    def __init__(
        self,
        known_ratio: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initialize OOD splitter.

        Args:
            known_ratio: Fraction of classes to keep as known (0-1)
            random_state: Random seed for reproducibility
        """
        self.known_ratio = known_ratio
        self.random_state = random_state

    def create_split(
        self,
        texts: list[str],
        labels: list[str],
    ) -> tuple[list[str], list[str], list[str], list[bool]]:
        """
        Create OOD train/test split.

        Args:
            texts: List of input texts
            labels: List of corresponding labels

        Returns:
            Tuple of (train_texts, train_labels, test_texts, test_is_novel)
            - test_is_novel: True for novel (previously unknown) classes
        """
        np.random.seed(self.random_state)

        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)
        n_known = max(1, int(n_classes * self.known_ratio))

        known_classes = set(np.random.choice(unique_labels, n_known, replace=False))

        train_texts = []
        train_labels = []
        test_texts = []
        test_is_novel = []

        for text, label in zip(texts, labels, strict=False):
            if label in known_classes:
                train_texts.append(text)
                train_labels.append(label)
            else:
                test_texts.append(text)
                test_is_novel.append(True)

        return train_texts, train_labels, test_texts, test_is_novel

    def create_split_with_indices(
        self,
        texts: list[str],
        labels: list[str],
    ) -> dict[str, Any]:
        """
        Create OOD split with additional metadata.

        Args:
            texts: List of input texts
            labels: List of corresponding labels

        Returns:
            Dict with split data and metadata
        """
        train_texts, train_labels, test_texts, test_is_novel = self.create_split(
            texts, labels
        )

        unique_labels = sorted(set(labels))
        known_classes = sorted(set(train_labels))
        novel_classes = sorted(set(unique_labels) - set(known_classes))

        return {
            "train_texts": train_texts,
            "train_labels": train_labels,
            "test_texts": test_texts,
            "test_is_novel": test_is_novel,
            "known_classes": known_classes,
            "novel_classes": novel_classes,
            "n_known": len(known_classes),
            "n_novel": len(novel_classes),
            "n_train": len(train_texts),
            "n_test": len(test_texts),
        }


class GradualNoveltySplitter:
    """
    Creates multiple splits with gradually increasing novelty.

    Useful for testing how novelty detection performance degrades
    as the number of novel classes increases.
    """

    def __init__(
        self,
        known_ratios: list[float] | None = None,
        random_state: int = 42,
    ):
        """
        Initialize gradual novelty splitter.

        Args:
            known_ratios: List of known ratios to create splits for
            random_state: Random seed for reproducibility
        """
        self.known_ratios = known_ratios or [0.95, 0.9, 0.8, 0.7, 0.5]
        self.random_state = random_state

    def create_splits(
        self,
        texts: list[str],
        labels: list[str],
    ) -> list[dict[str, Any]]:
        """
        Create multiple splits with different novelty levels.

        Args:
            texts: List of input texts
            labels: List of corresponding labels

        Returns:
            List of split dictionaries, one per known_ratio
        """
        splits = []

        for ratio in self.known_ratios:
            splitter = OODSplitter(known_ratio=ratio, random_state=self.random_state)
            split_data = splitter.create_split_with_indices(texts, labels)
            split_data["known_ratio"] = ratio
            splits.append(split_data)

        return splits

    def get_novelty_progression(
        self,
        texts: list[str],
        labels: list[str],
    ) -> dict[str, list]:
        """
        Get summary of novelty progression across splits.

        Args:
            texts: List of input texts
            labels: List of corresponding labels

        Returns:
            Dict with arrays for known_ratio, n_known, n_novel
        """
        splits = self.create_splits(texts, labels)

        return {
            "known_ratios": [s["known_ratio"] for s in splits],
            "n_known": [s["n_known"] for s in splits],
            "n_novel": [s["n_novel"] for s in splits],
            "n_train": [s["n_train"] for s in splits],
            "n_test": [s["n_test"] for s in splits],
        }
