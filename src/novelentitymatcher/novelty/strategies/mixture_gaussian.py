"""Mixture of Gaussians OOD detection strategy.

Fits a full-covariance multivariate Gaussian per class and scores samples
via log-likelihood under their predicted class. Richer class models catch
subtle near-OOD better than diagonal-covariance Mahalanobis.
"""

from typing import Any

import numpy as np

from ...utils.logging_config import get_logger
from ..config.strategies import MixtureGaussianConfig
from ..core.strategies import StrategyRegistry
from .base import NoveltyStrategy

logger = get_logger(__name__)


@StrategyRegistry.register
class MixtureGaussianStrategy(NoveltyStrategy):
    """Per-class multivariate Gaussian log-likelihood strategy.

    For each class, fits ``mean`` and ``cov`` from reference embeddings.
    Scores a sample by ``log p(x | predicted_class)`` plus optional log-prior.
    Samples with log-likelihood below a calibrated threshold are flagged.
    """

    strategy_id = "mixture_gaussian"
    maturity = "experimental"

    def __init__(self):
        self._config: MixtureGaussianConfig = None
        self._class_models: dict[str, dict[str, Any]] = {}
        self._dim: int = 0
        self._threshold: float = 0.0
        self._calibrator: Any = None

    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: MixtureGaussianConfig,
    ) -> None:
        """Initialize per-class Gaussian models from reference data.

        When ``calibration_mode="conformal"``, splits reference data into
        core and calibration sets, then wraps negated log-likelihoods with
        p-values.
        """
        self._config = config or MixtureGaussianConfig()
        self._dim = reference_embeddings.shape[1]
        self._class_models = {}
        self._calibrator = None

        if self._config.calibration_mode == "conformal":
            self._initialize_with_calibration(reference_embeddings, reference_labels)
        else:
            self._initialize_core(reference_embeddings, reference_labels)

    def _initialize_with_calibration(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        """Initialize with conformal calibration, splitting reference data."""
        from .conformal import ConformalCalibrator

        n = len(reference_embeddings)
        frac = self._config.calibration_set_fraction
        n_cal = max(1, int(n * frac))
        if n_cal >= n:
            logger.warning(
                "MixtureGaussian conformal calibration disabled: calibration split "
                "would leave no core reference samples (n=%d, n_cal=%d)",
                n,
                n_cal,
            )
            self._initialize_core(reference_embeddings, reference_labels)
            return

        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        cal_indices = indices[:n_cal]
        core_indices = indices[n_cal:]

        core_embeddings = reference_embeddings[core_indices]
        core_labels = [reference_labels[i] for i in core_indices]

        self._initialize_core(core_embeddings, core_labels)

        cal_embeddings = reference_embeddings[cal_indices]
        cal_labels = [reference_labels[i] for i in cal_indices]
        cal_scores = np.array(
            [
                -self._log_likelihood(cal_embeddings[i], cal_labels[i])
                for i in range(len(cal_embeddings))
            ]
        )

        self._calibrator = ConformalCalibrator(
            alpha=self._config.calibration_alpha,
            method=self._config.calibration_method,
        )
        self._calibrator.calibrate(cal_scores, np.array(cal_labels))
        logger.info(
            "MixtureGaussian strategy initialized with conformal calibration: "
            "n_core=%d, n_cal=%d, method=%s",
            len(core_embeddings),
            n_cal,
            self._config.calibration_method,
        )

    def _initialize_core(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        """Core initialization: fit per-class Gaussians and threshold."""
        unique_labels = set(reference_labels)
        n_total = len(reference_embeddings)

        for label in unique_labels:
            mask = np.array(reference_labels) == label
            class_embs = reference_embeddings[mask]
            cov = np.cov(class_embs, rowvar=False)
            if cov.ndim < 2:
                cov = np.array([[cov]])
            cov += self._config.regularization * np.eye(self._dim)

            self._class_models[label] = {
                "mean": class_embs.mean(axis=0),
                "cov": cov,
                "cov_inv": np.linalg.inv(cov),
                "prior": len(class_embs) / n_total,
            }

        ref_lls = np.array(
            [
                self._log_likelihood(reference_embeddings[i], reference_labels[i])
                for i in range(n_total)
            ]
        )
        self._threshold = float(
            np.mean(ref_lls) - self._config.threshold_std_multiplier * np.std(ref_lls)
        )

        logger.info(
            "MixtureGaussianStrategy initialized: n_classes=%d, dim=%d, threshold=%.4f",
            len(self._class_models),
            self._dim,
            self._threshold,
        )

    def _log_likelihood(self, x: np.ndarray, label: str) -> float:
        """Compute log-likelihood (plus log-prior) for sample x under class label."""
        model = self._class_models.get(label)
        if model is None:
            # Fallback: use global mean if unseen class
            all_means = np.array([m["mean"] for m in self._class_models.values()])
            mean = all_means.mean(axis=0)
            diff = x - mean
            cov = np.eye(self._dim) * self._config.regularization
            cov_inv = np.eye(self._dim) / self._config.regularization
            prior = 1.0 / max(len(self._class_models), 1)
        else:
            mean = model["mean"]
            cov = model["cov"]
            cov_inv = model["cov_inv"]
            prior = model["prior"]

        diff = x - mean
        mahal = float(diff @ cov_inv @ diff)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = float(np.log(max(self._config.regularization, 1e-12)) * self._dim)
        ll = -0.5 * (mahal + logdet + self._dim * np.log(2.0 * np.pi))
        if self._config.use_priors:
            ll += np.log(max(prior, 1e-12))
        return ll

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        """Detect novel samples via per-class log-likelihood.

        When ``calibration_mode="conformal"``, flagging uses p-values on
        negated log-likelihoods (higher = more OOD).
        """
        flags = set()
        metrics = {}

        if (
            self._config.calibration_mode == "conformal"
            and self._calibrator is not None
            and self._calibrator.is_calibrated
        ):
            neg_lls = np.array(
                [
                    -self._log_likelihood(embeddings[i], predicted_classes[i])
                    for i in range(len(embeddings))
                ]
            )
            if self._config.calibration_method == "mondrian":
                p_values = self._calibrator.predict_pvalues_for_class(
                    neg_lls, predicted_classes
                )
            else:
                p_values = self._calibrator.predict_pvalues(neg_lls)

            for idx in range(len(embeddings)):
                ll = -float(neg_lls[idx])
                metrics[idx] = {
                    "log_likelihood": ll,
                    "log_likelihood_threshold": self._threshold,
                    "predicted_class": predicted_classes[idx],
                    "p_value": float(p_values[idx]),
                    "calibration_mode": "conformal",
                    "mixture_gaussian_is_novel": p_values[idx]
                    < self._config.calibration_alpha,
                }
                if p_values[idx] < self._config.calibration_alpha:
                    flags.add(idx)
        else:
            for idx in range(len(embeddings)):
                pred_class = predicted_classes[idx]
                ll = self._log_likelihood(embeddings[idx], pred_class)
                metrics[idx] = {
                    "log_likelihood": ll,
                    "log_likelihood_threshold": self._threshold,
                    "predicted_class": pred_class,
                    "mixture_gaussian_is_novel": ll < self._threshold,
                }
                if ll < self._threshold:
                    flags.add(idx)

        return flags, metrics

    @property
    def config_schema(self) -> type:
        return MixtureGaussianConfig

    def get_weight(self) -> float:
        return 0.35
