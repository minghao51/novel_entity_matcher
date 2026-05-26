"""ReAct-style feature trimming utility for OOD strategies.

ReAct (Sun & Li, 2021) trims extreme activations before scoring to improve
OOD detection. This module provides a reusable wrapper that can be applied
to any strategy that operates on embeddings.
"""

from typing import Any

import numpy as np

from ...utils.logging_config import get_logger
from ..config.strategies import ReActConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy, SignalInfo
from .conformal_mixin import ConformalMixin
from .energy import EnergyOODStrategy

logger = get_logger(__name__)


def trim_activations(embeddings: np.ndarray, percentile: float) -> np.ndarray:
    threshold = np.percentile(embeddings, percentile * 100)
    trimmed = embeddings.copy()
    np.clip(trimmed, None, threshold, out=trimmed)
    return trimmed


@StrategyRegistry.register
class ReActEnergyStrategy(ConformalMixin, NoveltyStrategy):
    strategy_id = "react_energy"
    maturity = "experimental"
    score_keys = ("energy_score",)
    signal_info = SignalInfo(
        flag_key="react_energy_is_novel", weight_name="react_energy", kind="flag"
    )
    default_weight = 0.3

    def __init__(self):
        self._config: ReActConfig | None = None
        self._inner: EnergyOODStrategy | None = None
        self._calibrator: Any = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: ReActConfig,
    ) -> None:
        """Initialize ReAct wrapper and underlying energy strategy.

        When ``calibration_mode="conformal"``, trains a calibrator on
        trimmed-embedding energies.
        """
        self._config = config or ReActConfig()
        self._calibrator = None
        trimmed = trim_activations(reference_embeddings, self._config.trim_percentile)
        self._inner = EnergyOODStrategy()
        from ..config.strategies import EnergyConfig

        inner_config = EnergyConfig()
        self._inner.initialize(trimmed, reference_labels, inner_config)

        if self._config.calibration_mode == "conformal":
            self._initialize_conformal(trimmed, reference_labels)

        logger.info(
            "ReActEnergyStrategy initialized: trim_percentile=%.2f, inner=%s",
            self._config.trim_percentile,
            self._inner.strategy_id,
        )

    def _initialize_conformal(
        self,
        trimmed_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        self._run_conformal_calibration(
            trimmed_embeddings,
            reference_labels,
            lambda embs, _labels: self._inner._compute_energy(
                self._inner._compute_logits(embs)
            ),
            run_core_init=False,
        )

    def _detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        trimmed = trim_activations(embeddings, self._config.trim_percentile)

        if self._is_conformal_active():
            logits = self._inner._compute_logits(trimmed)
            energies = self._inner._compute_energy(logits)
            return self._conformal_detect_loop(
                energies,
                predicted_classes,
                lambda idx, energy, pv: {
                    "energy_score": energy,
                    "energy_threshold": self._inner._threshold,
                    "predicted_class": predicted_classes[idx],
                    "react_trim_percentile": self._config.trim_percentile,
                    "react_energy_is_novel": pv < self._config.calibration_alpha,
                },
            )

        flags, metrics = self._inner.detect(
            texts, trimmed, predicted_classes, confidences, **kwargs
        )
        for idx in metrics:
            metrics[idx]["react_trim_percentile"] = self._config.trim_percentile
            metrics[idx]["react_energy_is_novel"] = idx in flags
        return flags, metrics

    @property
    def config_schema(self) -> type:
        return ReActConfig
