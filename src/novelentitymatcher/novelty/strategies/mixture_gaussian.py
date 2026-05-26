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
from .base import NoveltyStrategy, SignalInfo
from .conformal_mixin import ConformalMixin

logger = get_logger(__name__)


@StrategyRegistry.register
class MixtureGaussianStrategy(ConformalMixin, NoveltyStrategy):
    strategy_id = "mixture_gaussian"
    maturity = "experimental"
    score_keys = ("log_likelihood",)
    signal_info = SignalInfo(
        score_key="log_likelihood",
        flag_key="mixture_gaussian_is_novel",
        weight_name="mixture_gaussian",
        kind="special",
    )
    default_weight = 0.35

    def __init__(self):
        self._config: MixtureGaussianConfig | None = None
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
        self._run_conformal_calibration(
            reference_embeddings,
            reference_labels,
            lambda embs, labels: np.array(
                [-self._log_likelihood(embs[i], labels[i]) for i in range(len(embs))]
            ),
        )

    def _initialize_core(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        """Core initialization: fit per-class Gaussians and threshold."""
        unique_labels = set(reference_labels)
        n_total = len(reference_embeddings)
        means = self.compute_class_means(reference_embeddings, reference_labels)

        for label in unique_labels:
            mask = np.array(reference_labels) == label
            class_embs = reference_embeddings[mask]
            if len(class_embs) < 2:
                cov = np.eye(self._dim) * self._config.regularization
            else:
                cov = np.cov(class_embs, rowvar=False)
                if cov.ndim < 2:
                    cov = np.array([[cov]])
                cov += self._config.regularization * np.eye(self._dim)

            self._class_models[label] = {
                "mean": means[label],
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
        model = self._class_models.get(label)
        means = {k: v["mean"] for k, v in self._class_models.items()}
        global_mean = (
            np.mean(list(means.values()), axis=0) if means else np.zeros(self._dim)
        )
        mean = self._resolve_class_mean(label, means, global_mean)

        if model is None:
            cov = np.eye(self._dim) * self._config.regularization
            cov_inv = np.eye(self._dim) / self._config.regularization
            prior = 1.0 / max(len(self._class_models), 1)
        else:
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

    def _detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        flags = set()
        metrics = {}

        if self._is_conformal_active():
            neg_lls = np.array(
                [
                    -self._log_likelihood(embeddings[i], predicted_classes[i])
                    for i in range(len(embeddings))
                ]
            )
            return self._conformal_detect_loop(
                neg_lls,
                predicted_classes,
                lambda idx, neg_ll, pv: {
                    "log_likelihood": -neg_ll,
                    "log_likelihood_threshold": self._threshold,
                    "predicted_class": predicted_classes[idx],
                    "mixture_gaussian_is_novel": pv < self._config.calibration_alpha,
                },
            )

        for idx in range(len(embeddings)):
            pred_class = predicted_classes[idx]
            ll = self._log_likelihood(embeddings[idx], pred_class)
            is_novel = ll < self._threshold
            metrics[idx] = {
                "log_likelihood": ll,
                "log_likelihood_threshold": self._threshold,
                "predicted_class": pred_class,
                "mixture_gaussian_is_novel": is_novel,
            }
            if is_novel:
                flags.add(idx)

        return flags, metrics

    @property
    def config_schema(self) -> type:
        return MixtureGaussianConfig
