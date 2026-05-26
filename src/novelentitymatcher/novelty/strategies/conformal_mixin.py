from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ConformalConfig(Protocol):
    calibration_mode: str
    calibration_alpha: float
    calibration_method: str
    calibration_set_fraction: float


class ConformalMixin:
    _calibrator: Any
    _config: ConformalConfig

    def _split_calibration_set(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> tuple[np.ndarray, list[str], np.ndarray, list[str]] | None:
        n = len(reference_embeddings)
        frac = getattr(self._config, "calibration_set_fraction", 0.2)
        n_cal = max(1, int(n * frac))
        if n_cal >= n:
            logger.warning(
                "%s conformal calibration disabled: calibration split "
                "would leave no core reference samples (n=%d, n_cal=%d)",
                type(self).__name__,
                n,
                n_cal,
            )
            return None
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        cal_indices = indices[:n_cal]
        core_indices = indices[n_cal:]
        return (
            reference_embeddings[core_indices],
            [reference_labels[i] for i in core_indices],
            reference_embeddings[cal_indices],
            [reference_labels[i] for i in cal_indices],
        )

    def _init_conformal_calibrator(
        self,
        cal_scores: np.ndarray,
        cal_labels: list[str] | np.ndarray,
        n_core: int,
        n_cal: int,
    ) -> None:
        from .conformal import ConformalCalibrator

        self._calibrator = ConformalCalibrator(
            alpha=self._config.calibration_alpha,
            method=self._config.calibration_method,
        )
        self._calibrator.calibrate(cal_scores, np.asarray(cal_labels))
        logger.info(
            "%s initialized with conformal calibration: n_core=%d, n_cal=%d, method=%s",
            type(self).__name__,
            n_core,
            n_cal,
            self._config.calibration_method,
        )

    def _is_conformal_active(self) -> bool:
        return (
            getattr(self._config, "calibration_mode", "none") == "conformal"
            and self._calibrator is not None
            and self._calibrator.is_calibrated
        )

    def _run_conformal_calibration(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        score_fn: Callable[[np.ndarray, list[str]], np.ndarray],
        *,
        run_core_init: bool = True,
    ) -> None:
        result = self._split_calibration_set(reference_embeddings, reference_labels)
        if result is None:
            if run_core_init:
                self._initialize_core(reference_embeddings, reference_labels)
            return
        core_embs, core_labels, cal_embs, cal_labels = result
        if run_core_init:
            self._initialize_core(core_embs, core_labels)
        cal_scores = score_fn(cal_embs, cal_labels)
        self._init_conformal_calibrator(
            cal_scores, cal_labels, len(core_embs), len(cal_embs)
        )

    def _get_conformal_pvalues(
        self,
        raw_scores: np.ndarray,
        predicted_classes: list[str],
    ) -> np.ndarray:
        if self._config.calibration_method == "mondrian":
            return self._calibrator.predict_pvalues_for_class(
                raw_scores, predicted_classes
            )
        return self._calibrator.predict_pvalues(raw_scores)

    def _conformal_detect_loop(
        self,
        raw_scores: np.ndarray,
        predicted_classes: list[str],
        build_metric_fn: Any,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        p_values = self._get_conformal_pvalues(raw_scores, predicted_classes)
        flags: set[int] = set()
        metrics: dict[int, dict[str, Any]] = {}
        for idx in range(len(raw_scores)):
            m = build_metric_fn(idx, float(raw_scores[idx]), float(p_values[idx]))
            m["p_value"] = float(p_values[idx])
            m["calibration_mode"] = "conformal"
            metrics[idx] = m
            if p_values[idx] < self._config.calibration_alpha:
                flags.add(idx)
        return flags, metrics
