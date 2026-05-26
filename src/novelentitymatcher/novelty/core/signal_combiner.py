"""
Signal combination for novelty detection.

This module handles the fusion of signals from multiple strategies
into final novelty decisions.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..config.base import DetectionConfig
from ..config.weights import WeightConfig

logger = logging.getLogger(__name__)

_signal_tables_built = False
_STRATEGY_SIGNALS: list[dict[str, str]] = []
_SIGNAL_BY_ID: dict[str, dict[str, str]] = {}
_SCORE_KEYS: list[str] = []
_FLAG_KEYS: list[str] = []


def _build_signal_tables() -> None:
    global \
        _signal_tables_built, \
        _STRATEGY_SIGNALS, \
        _SIGNAL_BY_ID, \
        _SCORE_KEYS, \
        _FLAG_KEYS
    if _signal_tables_built:
        return
    from .strategies import StrategyRegistry

    for sid in sorted(StrategyRegistry._strategies.keys()):
        cls = StrategyRegistry._strategies[sid]
        info = getattr(cls, "signal_info", None)
        if info is None:
            continue
        entry: dict[str, str] = {"id": sid}
        if info.score_key is not None:
            entry["score_key"] = info.score_key
        entry["flag_key"] = info.flag_key
        entry["weight"] = info.weight_name
        entry["kind"] = info.kind
        _STRATEGY_SIGNALS.append(entry)
    _SIGNAL_BY_ID = {s["id"]: s for s in _STRATEGY_SIGNALS}
    _SCORE_KEYS = [s["score_key"] for s in _STRATEGY_SIGNALS if s.get("score_key")]
    _FLAG_KEYS = [s["flag_key"] for s in _STRATEGY_SIGNALS]
    _signal_tables_built = True


class SignalCombiner:
    """
    Handles signal combination from multiple strategies.

    Supports several combination methods:
    - weighted: Weighted fusion of strategy scores
    - union: Flag if any strategy flags
    - intersection: Flag if all strategies flag
    - voting: Flag if majority of strategies flag
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.weights: WeightConfig = config.get_weight_config()
        self.combine_method = config.combine_method
        self._meta_model: Any | None = None
        _build_signal_tables()
        self._feature_names: list[str] = _SCORE_KEYS + _FLAG_KEYS

    def _weight_for_strategy(self, strategy_id: str) -> float:
        """Resolve the configured weight for a strategy id."""
        signal = _SIGNAL_BY_ID.get(strategy_id)
        if signal is None:
            return 0.0
        return getattr(self.weights, signal["weight"], 0.0)

    def combine(
        self,
        strategy_outputs: dict[str, tuple[set[int], dict]],
        all_metrics: dict[int, dict[str, Any]],
    ) -> tuple[set[int], dict[int, float]]:
        """
        Combine strategy signals into final novelty decisions.

        Args:
            strategy_outputs: Dict mapping strategy_id to (flags, metrics)
            all_metrics: Dict mapping sample index to all metrics

        Returns:
            (novel_indices, novelty_scores)
            - novel_indices: Set of indices flagged as novel
            - novelty_scores: Dict mapping index to final novelty score
        """
        if self.combine_method == "weighted":
            return self._weighted_combination(strategy_outputs, all_metrics)
        elif self.combine_method == "union":
            return self._union_combination(strategy_outputs)
        elif self.combine_method == "intersection":
            return self._intersection_combination(strategy_outputs)
        elif self.combine_method == "voting":
            return self._voting_combination(strategy_outputs)
        elif self.combine_method == "meta_learner":
            return self._meta_learner_combination(strategy_outputs, all_metrics)
        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")

    def _weighted_combination(
        self,
        strategy_outputs: dict[str, tuple[set[int], dict]],
        all_metrics: dict[int, dict[str, Any]],
    ) -> tuple[set[int], dict[int, float]]:
        """
        Weighted fusion of strategy scores.

        Computes a weighted average of strategy scores and applies
        heuristics for high-confidence detection.
        """
        novelty_scores: dict[int, float] = {}
        novel_indices: set[int] = set()

        # Collect all sample indices that were flagged by any strategy
        all_indices = set()
        for flags, _ in strategy_outputs.values():
            all_indices.update(flags)

        # Only include weights for strategies that are actually in use.
        active_strategy_ids = set(strategy_outputs.keys())
        total_weight = sum(
            self._weight_for_strategy(strategy_id)
            for strategy_id in active_strategy_ids
        )

        # Compute weighted score for each flagged sample
        for idx in all_indices:
            score = self._compute_weighted_score(
                idx,
                all_metrics,
                active_strategy_ids,
            )
            # Normalize by total weight of active strategies
            if total_weight > 0:
                score = score / total_weight
            novelty_scores[idx] = score

            if self._is_novel(idx, score, all_metrics):
                novel_indices.add(idx)

        return novel_indices, novelty_scores

    def _compute_weighted_score(
        self,
        idx: int,
        metrics: dict[int, dict[str, Any]],
        active_strategies: set[str] | None = None,
    ) -> float:
        if active_strategies is None:
            active_strategies = set()

        sample_metrics = metrics.get(idx, {})
        weighted_score = 0.0

        def _resolve_value(signal: dict[str, str]) -> float:
            src = sample_metrics
            kind = signal["kind"]
            if kind == "flag":
                return 1.0 if src.get(signal["flag_key"], False) else 0.0
            if kind == "score":
                return float(src.get(signal["score_key"], 0.0))
            if kind == "special":
                if signal["id"] == "energy_ood":
                    score = src.get("energy_score")
                    if score is not None:
                        return float(score > src.get("energy_threshold", float("inf")))
                    return 1.0 if src.get("energy_is_novel", False) else 0.0
                if signal["id"] == "mixture_gaussian":
                    score = src.get("log_likelihood")
                    if score is not None:
                        return float(
                            score < src.get("log_likelihood_threshold", float("-inf"))
                        )
                    return 1.0 if src.get("mixture_gaussian_is_novel", False) else 0.0
            return 0.0

        for strategy_id in active_strategies:
            signal = _SIGNAL_BY_ID.get(strategy_id)
            if signal is None:
                continue
            weight = getattr(self.weights, signal["weight"], 0.0)
            weighted_score += weight * _resolve_value(signal)

        return float(np.clip(weighted_score, 0.0, 1.0))

    def _is_novel(
        self, idx: int, score: float, metrics: dict[int, dict[str, Any]]
    ) -> bool:
        """
        Determine if a sample is novel based on score and heuristics.

        Applies several heuristics in addition to the weighted score:
        - Strong uncertainty threshold
        - Strong kNN threshold
        - Final novelty threshold
        """
        sample_metrics = metrics.get(idx, {})

        # Strong uncertainty heuristics
        uncertainty_score = sample_metrics.get("uncertainty_score", 0.0)
        if uncertainty_score >= self.weights.strong_uncertainty_threshold:
            return True

        # Strong kNN heuristics
        knn_score = sample_metrics.get("knn_novelty_score", 0.0)
        if knn_score >= self.weights.strong_knn_threshold:
            return True

        # kNN gate threshold
        if knn_score >= self.weights.knn_gate_threshold:
            return True

        # Final threshold check
        return score >= self.weights.novelty_threshold

    def _union_combination(
        self, strategy_outputs: dict[str, tuple[set[int], dict]]
    ) -> tuple[set[int], dict[int, float]]:
        """
        Union combination: flag if any strategy flags.

        Returns score of 1.0 for flagged samples.
        """
        novel_indices: set[int] = set()
        novelty_scores: dict[int, float] = {}

        for flags, _ in strategy_outputs.values():
            novel_indices.update(flags)

        for idx in novel_indices:
            novelty_scores[idx] = 1.0

        return novel_indices, novelty_scores

    def _intersection_combination(
        self, strategy_outputs: dict[str, tuple[set[int], dict]]
    ) -> tuple[set[int], dict[int, float]]:
        """
        Intersection combination: flag only if all strategies flag.

        Returns score of 1.0 for flagged samples.
        """
        if not strategy_outputs:
            return set(), {}

        # Get all flagged indices from first strategy
        first_flags = next(iter(strategy_outputs.values()))[0]

        # Intersect with all other strategies
        novel_indices = first_flags.copy()
        for flags, _ in strategy_outputs.values():
            novel_indices.intersection_update(flags)

        novelty_scores = dict.fromkeys(novel_indices, 1.0)

        return novel_indices, novelty_scores

    def _voting_combination(
        self, strategy_outputs: dict[str, tuple[set[int], dict]]
    ) -> tuple[set[int], dict[int, float]]:
        """
        Voting combination: flag if majority of strategies flag.

        Score represents the fraction of strategies that flagged the sample.
        """
        # Count votes for each sample
        vote_counts: dict[int, int] = {}
        num_strategies = len(strategy_outputs)

        for flags, _ in strategy_outputs.values():
            for idx in flags:
                vote_counts[idx] = vote_counts.get(idx, 0) + 1

        # Flag samples with majority votes
        majority_threshold = num_strategies // 2 + 1
        novel_indices = {
            idx for idx, count in vote_counts.items() if count >= majority_threshold
        }

        # Score is fraction of strategies that flagged
        novelty_scores = {
            idx: count / num_strategies for idx, count in vote_counts.items()
        }

        return novel_indices, novelty_scores

    # ------------------------------------------------------------------
    # Meta-learner combination
    # ------------------------------------------------------------------

    def _meta_learner_combination(
        self,
        strategy_outputs: dict[str, tuple[set[int], dict]],
        all_metrics: dict[int, dict[str, Any]],
    ) -> tuple[set[int], dict[int, float]]:
        """
        Learned fusion of strategy scores via a logistic regression meta-learner.

        Falls back to weighted combination when no trained model is available.
        """
        if self._meta_model is None:
            logger.warning(
                "meta_learner combine_method selected but no trained model found; "
                "falling back to weighted combination"
            )
            return self._weighted_combination(strategy_outputs, all_metrics)

        novelty_scores: dict[int, float] = {}
        novel_indices: set[int] = set()

        all_indices = set()
        for flags, _ in strategy_outputs.values():
            all_indices.update(flags)

        if not all_indices:
            return novel_indices, novelty_scores

        feature_matrix = np.array(
            [self._extract_features(idx, all_metrics) for idx in sorted(all_indices)]
        )
        predictions = self._meta_model.predict_proba(feature_matrix)[:, 1]

        for pos, idx in enumerate(sorted(all_indices)):
            score = float(predictions[pos])
            novelty_scores[idx] = score
            if score >= self.weights.novelty_threshold:
                novel_indices.add(idx)

        return novel_indices, novelty_scores

    def _extract_features(
        self, idx: int, metrics: dict[int, dict[str, Any]]
    ) -> list[float]:
        """Extract a fixed-length feature vector from per-sample metrics.

        Returns one float per key in ``_SCORE_KEYS + _FLAG_KEYS`` (29 features).
        Score keys yield the raw float; flag keys yield 0.0 or 1.0.
        """
        sample = metrics.get(idx, {})
        features: list[float] = []

        for key in _SCORE_KEYS:
            val = sample.get(key)
            features.append(float(val) if isinstance(val, (int, float)) else 0.0)

        for key in _FLAG_KEYS:
            features.append(1.0 if sample.get(key, False) else 0.0)

        return features

    # ------------------------------------------------------------------
    # Meta-learner training / persistence
    # ------------------------------------------------------------------

    def train_meta_learner(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Train the logistic regression meta-learner.

        Args:
            features: (n_samples, n_features) matrix of strategy scores
            labels: (n_samples,) binary novelty labels (1=novel, 0=known)

        Returns:
            Training accuracy
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as err:
            raise ImportError(
                "scikit-learn is required for meta-learner training. "
                "Install with: pip install scikit-learn"
            ) from err

        self._meta_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )
        self._meta_model.fit(features, labels)
        accuracy = float(self._meta_model.score(features, labels))
        logger.info("Meta-learner trained with accuracy=%.4f", accuracy)
        return accuracy

    def save_meta_learner(self, path: str) -> None:
        """Persist the trained meta-learner to disk."""
        if self._meta_model is None:
            raise RuntimeError("No trained meta-learner to save")

        import joblib

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._meta_model, p / "meta_learner.pkl")

        meta = {
            "feature_names": _SCORE_KEYS + _FLAG_KEYS,
            "n_features": len(_SCORE_KEYS) + len(_FLAG_KEYS),
            "novelty_threshold": self.weights.novelty_threshold,
        }
        with open(p / "meta_learner_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load_meta_learner(self, path: str) -> None:
        """Load a trained meta-learner from disk."""
        import joblib

        p = Path(path)
        self._meta_model = joblib.load(p / "meta_learner.pkl")

        with open(p / "meta_learner_meta.json") as f:
            meta = json.load(f)
        self._feature_names = meta.get("feature_names", _SCORE_KEYS + _FLAG_KEYS)
        logger.info("Meta-learner loaded from %s", path)
