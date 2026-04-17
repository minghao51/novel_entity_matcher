"""
Signal combination for novelty detection.

This module handles the fusion of signals from multiple strategies
into final novelty decisions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..config.base import DetectionConfig
from ..config.weights import WeightConfig

logger = logging.getLogger(__name__)

# Strategy metrics extracted for each sample during weighted fusion.
_SCORE_KEYS = [
    "confidence_score",
    "uncertainty_score",
    "knn_novelty_score",
    "cluster_support_score",
    "self_knowledge_score",
    "pattern_score",
    "oneclass_score",
    "prototypical_score",
    "setfit_score",
    "setfit_centroid_score",
]

_FLAG_KEYS = [
    "confidence_is_novel",
    "uncertainty_is_novel",
    "knn_is_novel",
    "cluster_is_novel",
    "self_knowledge_is_novel",
    "pattern_is_novel",
    "oneclass_is_novel",
    "prototypical_is_novel",
    "setfit_is_novel",
    "setfit_centroid_is_novel",
]


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
        """
        Initialize the signal combiner.

        Args:
            config: Detection configuration
        """
        self.config = config
        self.weights: WeightConfig = config.get_weight_config()
        self.combine_method = config.combine_method
        self._meta_model: Optional[Any] = None
        self._feature_names: List[str] = _SCORE_KEYS + _FLAG_KEYS

    def _weight_for_strategy(self, strategy_id: str) -> float:
        """Resolve the configured weight for a strategy id."""
        weight_map = {
            "confidence": getattr(self.weights, "confidence", 0.35),
            "uncertainty": self.weights.uncertainty,
            "knn_distance": self.weights.knn,
            "clustering": self.weights.cluster,
            "self_knowledge": self.weights.self_knowledge,
            "pattern": self.weights.pattern,
            "oneclass": self.weights.oneclass,
            "prototypical": self.weights.prototypical,
            "setfit": self.weights.setfit,
            "setfit_centroid": self.weights.setfit_centroid,
        }
        return weight_map.get(strategy_id, 0.0)

    def combine(
        self,
        strategy_outputs: Dict[str, tuple[Set[int], Dict]],
        all_metrics: Dict[int, Dict[str, Any]],
    ) -> tuple[Set[int], Dict[int, float]]:
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
        strategy_outputs: Dict[str, tuple[Set[int], Dict]],
        all_metrics: Dict[int, Dict[str, Any]],
    ) -> tuple[Set[int], Dict[int, float]]:
        """
        Weighted fusion of strategy scores.

        Computes a weighted average of strategy scores and applies
        heuristics for high-confidence detection.
        """
        novelty_scores: Dict[int, float] = {}
        novel_indices: Set[int] = set()

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
        metrics: Dict[int, Dict[str, Any]],
        active_strategies: Set[str] | None = None,
    ) -> float:
        """
        Compute weighted novelty score for a single sample.

        Extracts scores from each strategy and combines them using
        the configured weights.
        """
        if active_strategies is None:
            active_strategies = set()

        sample_metrics = metrics.get(idx, {})

        # For strategies with binary decisions, use 1.0 if flagged, 0.0 otherwise
        confidence = 1.0 if sample_metrics.get("confidence_is_novel", False) else 0.0
        pattern = 1.0 if sample_metrics.get("pattern_is_novel", False) else 0.0
        oneclass = 1.0 if sample_metrics.get("oneclass_is_novel", False) else 0.0
        prototypical = (
            1.0 if sample_metrics.get("prototypical_is_novel", False) else 0.0
        )
        setfit = 1.0 if sample_metrics.get("setfit_is_novel", False) else 0.0
        setfit_centroid = sample_metrics.get("setfit_centroid_novelty_score", 0.0)
        self_knowledge = (
            1.0 if sample_metrics.get("self_knowledge_is_novel", False) else 0.0
        )

        # For strategies with continuous scores, use the score directly
        uncertainty = sample_metrics.get("uncertainty_score", 0.0)
        knn_score = sample_metrics.get("knn_novelty_score", 0.0)
        cluster_score = sample_metrics.get("cluster_support_score", 0.0)

        # Get confidence weight from weights config if it exists
        confidence_weight = self._weight_for_strategy("confidence")

        # Compute weighted sum (only include active strategies)
        weighted_score = 0.0

        if "confidence" in active_strategies:
            weighted_score += confidence_weight * confidence
        if "uncertainty" in active_strategies:
            weighted_score += self.weights.uncertainty * uncertainty
        if "knn_distance" in active_strategies:
            weighted_score += self.weights.knn * knn_score
        if "clustering" in active_strategies:
            weighted_score += self.weights.cluster * cluster_score
        if "self_knowledge" in active_strategies:
            weighted_score += self.weights.self_knowledge * self_knowledge
        if "pattern" in active_strategies:
            weighted_score += self.weights.pattern * pattern
        if "oneclass" in active_strategies:
            weighted_score += self.weights.oneclass * oneclass
        if "prototypical" in active_strategies:
            weighted_score += self.weights.prototypical * prototypical
        if "setfit" in active_strategies:
            weighted_score += self.weights.setfit * setfit
        if "setfit_centroid" in active_strategies:
            weighted_score += self.weights.setfit_centroid * setfit_centroid

        return float(np.clip(weighted_score, 0.0, 1.0))

    def _is_novel(
        self, idx: int, score: float, metrics: Dict[int, Dict[str, Any]]
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
        self, strategy_outputs: Dict[str, tuple[Set[int], Dict]]
    ) -> tuple[Set[int], Dict[int, float]]:
        """
        Union combination: flag if any strategy flags.

        Returns score of 1.0 for flagged samples.
        """
        novel_indices: Set[int] = set()
        novelty_scores: Dict[int, float] = {}

        for flags, _ in strategy_outputs.values():
            novel_indices.update(flags)

        for idx in novel_indices:
            novelty_scores[idx] = 1.0

        return novel_indices, novelty_scores

    def _intersection_combination(
        self, strategy_outputs: Dict[str, tuple[Set[int], Dict]]
    ) -> tuple[Set[int], Dict[int, float]]:
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

        novelty_scores = {idx: 1.0 for idx in novel_indices}

        return novel_indices, novelty_scores

    def _voting_combination(
        self, strategy_outputs: Dict[str, tuple[Set[int], Dict]]
    ) -> tuple[Set[int], Dict[int, float]]:
        """
        Voting combination: flag if majority of strategies flag.

        Score represents the fraction of strategies that flagged the sample.
        """
        # Count votes for each sample
        vote_counts: Dict[int, int] = {}
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
        strategy_outputs: Dict[str, tuple[Set[int], Dict]],
        all_metrics: Dict[int, Dict[str, Any]],
    ) -> tuple[Set[int], Dict[int, float]]:
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

        novelty_scores: Dict[int, float] = {}
        novel_indices: Set[int] = set()

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
        self, idx: int, metrics: Dict[int, Dict[str, Any]]
    ) -> List[float]:
        """Extract a fixed-length feature vector from per-sample metrics."""
        sample = metrics.get(idx, {})
        features = []

        # Continuous scores
        features.append(1.0 if sample.get("confidence_is_novel", False) else 0.0)
        features.append(sample.get("uncertainty_score", 0.0))
        features.append(sample.get("knn_novelty_score", 0.0))
        features.append(sample.get("cluster_support_score", 0.0))
        features.append(1.0 if sample.get("self_knowledge_is_novel", False) else 0.0)
        features.append(1.0 if sample.get("pattern_is_novel", False) else 0.0)
        features.append(1.0 if sample.get("oneclass_is_novel", False) else 0.0)
        features.append(1.0 if sample.get("prototypical_is_novel", False) else 0.0)
        features.append(1.0 if sample.get("setfit_is_novel", False) else 0.0)
        features.append(sample.get("setfit_centroid_novelty_score", 0.0))

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
        except ImportError:
            raise ImportError(
                "scikit-learn is required for meta-learner training. "
                "Install with: pip install scikit-learn"
            )

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
