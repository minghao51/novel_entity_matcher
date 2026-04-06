"""
LOF (Local Outlier Factor) novelty detection strategy.

Flags samples based on their local outlier factor score computed
against the reference set using sklearn's LocalOutlierFactor.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Type

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ..config.strategies import LOFConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy

logger = logging.getLogger(__name__)


@StrategyRegistry.register
class LOFStrategy(NoveltyStrategy):
    """
    LOF strategy for novelty detection.

    Trains a Local Outlier Factor model on reference embeddings in
    novelty=True mode, then scores new samples. Samples with scores
    below the configurable threshold are flagged as novel.
    """

    strategy_id = "lof"

    def __init__(self):
        self._config: Optional[LOFConfig] = None
        self._lof_model: Optional[LocalOutlierFactor] = None
        self._fallback: bool = False

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: LOFConfig,
    ) -> None:
        """
        Initialize LOF strategy by fitting on reference embeddings.

        Args:
            reference_embeddings: Embeddings of known samples
            reference_labels: Labels of known samples
            config: LOFConfig with n_neighbors, contamination, metric, threshold
        """
        self._config = config or LOFConfig()
        self._fallback = False

        n_ref = len(reference_embeddings)
        n_neighbors = self._config.n_neighbors

        if n_ref < n_neighbors:
            logger.warning(
                "LOF: reference set too small (%d < %d neighbors). "
                "Falling back to flagging all samples.",
                n_ref,
                n_neighbors,
            )
            self._lof_model = None
            self._fallback = True
            return

        try:
            self._lof_model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self._config.contamination,
                metric=self._config.metric,
                novelty=True,
            )
            self._lof_model.fit(reference_embeddings)
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("LOF: failed to fit model: %s. Falling back.", exc)
            self._lof_model = None
            self._fallback = True

    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples using LOF anomaly scores.

        Args:
            texts: Input texts
            embeddings: Text embeddings
            predicted_classes: Predicted classes
            confidences: Prediction confidences
            **kwargs: Additional parameters

        Returns:
            (flags, metrics) - Flagged indices and per-sample metrics
        """
        flags: Set[int] = set()
        metrics: Dict[int, Dict[str, Any]] = {}

        if self._fallback or self._lof_model is None:
            for idx in range(len(embeddings)):
                metrics[idx] = {
                    "lof_score": 0.0,
                    "lof_novelty_score": 1.0,
                    "lof_is_outlier": True,
                }
                flags.add(idx)
            return flags, metrics

        try:
            raw_scores = self._lof_model.score_samples(embeddings)
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("LOF: score_samples failed: %s. Flagging all.", exc)
            for idx in range(len(embeddings)):
                metrics[idx] = {
                    "lof_score": 0.0,
                    "lof_novelty_score": 1.0,
                    "lof_is_outlier": True,
                }
                flags.add(idx)
            return flags, metrics

        threshold = self._config.score_threshold

        for idx in range(len(embeddings)):
            score = float(raw_scores[idx])
            novelty_score = -score
            is_outlier = score < threshold

            metrics[idx] = {
                "lof_score": score,
                "lof_novelty_score": novelty_score,
                "lof_is_outlier": is_outlier,
            }

            if is_outlier:
                flags.add(idx)

        return flags, metrics

    @property
    def config_schema(self) -> Type:
        """Return LOFConfig as the config schema."""
        return LOFConfig

    def get_weight(self) -> float:
        """Return weight for signal combination."""
        return 0.30
