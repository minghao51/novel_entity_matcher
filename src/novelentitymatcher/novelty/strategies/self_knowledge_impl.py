"""
Self-knowledge detection via sparse autoencoder for novelty detection.

Detects when query embeddings fall into "unknown" regions of the
representation space using sparse autoencoder-learned directions.
"""

from __future__ import annotations

import numpy as np

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class SparseAutoencoder:
    """
    Sparse Autoencoder for learning "known" embedding space.

    Trains on known entity embeddings to learn which directions
    correspond to "valid" entity space. Queries that don't project
    well onto these directions are flagged as novel.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        sparsity_weight: float = 1e-4,
        l2_weight: float = 1e-4,
        encoder_activation: str = "relu",
        decoder_activation: str = "sigmoid",
        random_state: int = 42,
    ):
        """
        Initialize sparse autoencoder.

        Args:
            hidden_dim: Dimension of hidden layer (bottleneck).
            sparsity_weight: Weight for sparsity penalty (KL divergence).
            l2_weight: Weight for L2 regularization on weights.
            encoder_activation: Activation function for encoder ('relu' or 'leaky_relu').
            decoder_activation: Activation function for decoder ('sigmoid' or 'relu').
            random_state: Random seed for reproducibility.
        """
        self.hidden_dim = hidden_dim
        self.sparsity_weight = sparsity_weight
        self.l2_weight = l2_weight
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.random_state = random_state

        self.encoder_weights: np.ndarray | None = None
        self.encoder_bias: np.ndarray | None = None
        self.decoder_weights: np.ndarray | None = None
        self.decoder_bias: np.ndarray | None = None
        self._is_fitted = False
        self._input_dim: int | None = None
        self._sparsity_target = 0.1

    def _init_weights(self, input_dim: int) -> None:
        """Initialize encoder/decoder weights using Xavier initialization."""
        np.random.seed(self.random_state)
        scale_encoder = np.sqrt(2.0 / (input_dim + self.hidden_dim))
        scale_decoder = np.sqrt(2.0 / (self.hidden_dim + input_dim))

        self.encoder_weights = (
            np.random.randn(input_dim, self.hidden_dim).astype(np.float32)
            * scale_encoder
        )
        self.encoder_bias = np.zeros(self.hidden_dim, dtype=np.float32)
        self.decoder_weights = (
            np.random.randn(self.hidden_dim, input_dim).astype(np.float32)
            * scale_decoder
        )
        self.decoder_bias = np.zeros(input_dim, dtype=np.float32)
        self._input_dim = input_dim

    @staticmethod
    def _activate(x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation == "relu":
            return np.maximum(0, x)
        elif activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        elif activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif activation == "tanh":
            return np.tanh(x)
        else:
            return x

    @staticmethod
    def _activate_derivative(x: np.ndarray, activation: str) -> np.ndarray:
        """Compute derivative of activation function."""
        if activation == "relu":
            return (x > 0).astype(np.float32)
        elif activation == "leaky_relu":
            return np.where(x > 0, 1.0, 0.01)
        elif activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        elif activation == "tanh":
            return 1.0 - np.tanh(x) ** 2
        else:
            return np.ones_like(x)

    def _encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass through encoder."""
        h_pre = np.dot(x, self.encoder_weights) + self.encoder_bias
        h = self._activate(h_pre, self.encoder_activation)
        return h, h_pre

    def _decode(self, h: np.ndarray) -> np.ndarray:
        """Forward pass through decoder."""
        x_recon = np.dot(h, self.decoder_weights) + self.decoder_bias
        return x_recon

    def fit(
        self,
        known_embeddings: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        verbose: bool = False,
    ) -> SparseAutoencoder:
        """
        Train autoencoder on known entity embeddings.

        Args:
            known_embeddings: Array of shape (n_samples, embedding_dim)
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Learning rate for Adam optimizer.
            verbose: Whether to log progress.

        Returns:
            self
        """
        X = np.asarray(known_embeddings, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        n_samples, input_dim = X.shape
        self._init_weights(input_dim)

        if batch_size > n_samples:
            batch_size = n_samples

        lr = learning_rate
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            total_recon_loss = 0.0
            total_sparsity_loss = 0.0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]

                h, h_pre = self._encode(X_batch)
                x_recon = self._decode(h)

                recon_loss = np.mean((X_batch - x_recon) ** 2)

                rho_hat = np.mean(h, axis=0)
                rho = self._sparsity_target
                kl_div = rho * np.log(rho / (rho_hat + 1e-8)) + (1 - rho) * np.log(
                    (1 - rho) / (1 - rho_hat + 1e-8)
                )
                sparsity_loss = np.sum(kl_div)

                l2_loss = np.sum(self.encoder_weights**2) + np.sum(
                    self.decoder_weights**2
                )

                loss = (
                    recon_loss
                    + self.sparsity_weight * sparsity_loss
                    + self.l2_weight * l2_loss
                )
                total_loss += loss * (end - start)
                total_recon_loss += recon_loss * (end - start)
                total_sparsity_loss += sparsity_loss * (end - start)

                grad_h = 2 * (x_recon - X_batch) @ self.decoder_weights.T
                kl_grad = (
                    rho / (rho_hat + 1e-8) - (1 - rho) / (1 - rho_hat + 1e-8)
                ) / (end - start)
                grad_h += self.sparsity_weight * kl_grad
                grad_h_pre = grad_h * self._activate_derivative(
                    h_pre, self.encoder_activation
                )

                grad_w_enc = X_batch.T @ grad_h_pre / (end - start)
                grad_b_enc = np.mean(grad_h_pre, axis=0)
                grad_recon_dec = 2 * (x_recon - X_batch) / (end - start)
                grad_w_dec = h.T @ grad_recon_dec
                grad_b_dec = np.mean(grad_recon_dec, axis=0)

                self.encoder_weights -= lr * (
                    grad_w_enc + 2 * self.l2_weight * self.encoder_weights
                )
                self.encoder_bias -= lr * grad_b_enc
                self.decoder_weights -= lr * (
                    grad_w_dec + 2 * self.l2_weight * self.decoder_weights
                )
                self.decoder_bias -= lr * grad_b_dec

            if verbose and (epoch + 1) % 20 == 0:
                avg_loss = total_loss / n_samples
                avg_recon = total_recon_loss / n_samples
                avg_sparsity = total_sparsity_loss / n_samples
                logger.info(
                    f"SparseAE epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} "
                    f"(recon={avg_recon:.4f}, sparsity={avg_sparsity:.4f})"
                )

        self._is_fitted = True
        logger.info(
            f"SparseAE trained on {n_samples} embeddings, hidden_dim={self.hidden_dim}"
        )
        return self

    def compute_reconstruction_error(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute per-sample reconstruction error.

        Higher reconstruction error = more novel.

        Args:
            embeddings: Query embeddings of shape (n_samples, embedding_dim)

        Returns:
            Array of reconstruction errors (MSE per sample)
        """
        if not self._is_fitted:
            raise RuntimeError("SparseAutoencoder must be fitted before use")

        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        h, _ = self._encode(X)
        x_recon = self._decode(h)

        mse_per_sample = np.mean((X - x_recon) ** 2, axis=1)
        return mse_per_sample

    def compute_knowledge_score(
        self,
        embeddings: np.ndarray,
        normalizer: str | None = "minmax",
    ) -> np.ndarray:
        """
        Compute knowledge score (0-1, higher = more known).

        Uses reconstruction error converted to knowledge score.
        Lower reconstruction error = higher knowledge.

        Args:
            embeddings: Query embeddings of shape (n_samples, embedding_dim)
            normalizer: How to normalize scores ('minmax', 'zscore', or None)

        Returns:
            Knowledge scores in [0, 1] range
        """
        reconstruction_errors = self.compute_reconstruction_error(embeddings)

        if normalizer == "minmax":
            max_err = np.max(reconstruction_errors) + 1e-8
            min_err = np.min(reconstruction_errors)
            if max_err > min_err:
                knowledge = 1.0 - (reconstruction_errors - min_err) / (
                    max_err - min_err
                )
            else:
                knowledge = np.ones_like(reconstruction_errors)
            return np.clip(knowledge, 0.0, 1.0)

        elif normalizer == "zscore":
            mean_err = np.mean(reconstruction_errors)
            std_err = np.std(reconstruction_errors) + 1e-8
            z_scores = (reconstruction_errors - mean_err) / std_err
            knowledge = 1.0 / (1.0 + np.exp(z_scores))
            return np.clip(knowledge, 0.0, 1.0)

        else:
            return reconstruction_errors

    def compute_novelty_score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute novelty score (0-1, higher = more novel).

        This is simply 1 - knowledge_score.

        Args:
            embeddings: Query embeddings

        Returns:
            Novelty scores in [0, 1] range
        """
        return 1.0 - self.compute_knowledge_score(embeddings)

    def save(self, path: str) -> None:
        """Save trained autoencoder to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted autoencoder")

        save_dict = {
            "encoder_weights": self.encoder_weights,
            "encoder_bias": self.encoder_bias,
            "decoder_weights": self.decoder_weights,
            "decoder_bias": self.decoder_bias,
            "hidden_dim": self.hidden_dim,
            "sparsity_weight": self.sparsity_weight,
            "l2_weight": self.l2_weight,
            "encoder_activation": self.encoder_activation,
            "decoder_activation": self.decoder_activation,
            "input_dim": self._input_dim,
        }
        np.savez(path, **save_dict)
        logger.info(f"SparseAE saved to {path}")

    @classmethod
    def load(cls, path: str) -> SparseAutoencoder:
        """Load trained autoencoder from disk."""
        data = np.load(path, allow_pickle=False)
        instance = cls(
            hidden_dim=int(data["hidden_dim"]),
            sparsity_weight=float(data["sparsity_weight"]),
            l2_weight=float(data["l2_weight"]),
            encoder_activation=str(data["encoder_activation"]),
            decoder_activation=str(data["decoder_activation"]),
        )
        instance.encoder_weights = data["encoder_weights"]
        instance.encoder_bias = data["encoder_bias"]
        instance.decoder_weights = data["decoder_weights"]
        instance.decoder_bias = data["decoder_bias"]
        instance._input_dim = int(data["input_dim"])
        instance._is_fitted = True
        logger.info(f"SparseAE loaded from {path}")
        return instance


class SelfKnowledgeDetector:
    """
    Novelty detector using sparse autoencoder-learned knowledge scores.

    This detector identifies novel samples by measuring how well
    query embeddings reconstruct against an autoencoder trained
    solely on known entity embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        sparsity_weight: float = 1e-4,
        knowledge_threshold: float = 0.5,
        autoencoder_epochs: int = 100,
        autoencoder_batch_size: int = 256,
        normalization: str = "minmax",
    ):
        """
        Initialize self-knowledge detector.

        Args:
            hidden_dim: Bottleneck dimension for autoencoder.
            sparsity_weight: Sparsity penalty weight.
            knowledge_threshold: Threshold below which samples are novel (0-1).
            autoencoder_epochs: Epochs for autoencoder training.
            autoencoder_batch_size: Batch size for training.
            normalization: Score normalization method.
        """
        self.hidden_dim = hidden_dim
        self.sparsity_weight = sparsity_weight
        self.knowledge_threshold = knowledge_threshold
        self.autoencoder_epochs = autoencoder_epochs
        self.autoencoder_batch_size = autoencoder_batch_size
        self.normalization = normalization

        self.autoencoder: SparseAutoencoder | None = None
        self._is_fitted = False
        self._reference_embeddings: np.ndarray | None = None

    def fit(
        self,
        known_embeddings: np.ndarray,
        verbose: bool = False,
    ) -> SelfKnowledgeDetector:
        """
        Train autoencoder on known embeddings.

        Args:
            known_embeddings: Embeddings of known entities (n_samples, dim)
            verbose: Log training progress.

        Returns:
            self
        """
        self._reference_embeddings = np.asarray(known_embeddings, dtype=np.float32)
        self.autoencoder = SparseAutoencoder(
            hidden_dim=self.hidden_dim,
            sparsity_weight=self.sparsity_weight,
        )
        self.autoencoder.fit(
            self._reference_embeddings,
            epochs=self.autoencoder_epochs,
            batch_size=self.autoencoder_batch_size,
            verbose=verbose,
        )
        self._is_fitted = True
        return self

    def detect_novel_samples(
        self,
        embeddings: np.ndarray,
        threshold: float | None = None,
        return_scores: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect novel samples based on knowledge scores.

        Args:
            embeddings: Query embeddings (n_samples, dim)
            threshold: Override for knowledge_threshold. Samples with
                       knowledge below this are flagged as novel.
            return_scores: If True, return (novelty_scores, is_novel).
                          If False, return (is_novel,) only.

        Returns:
            Tuple of (novelty_scores, is_novel_mask) or just is_novel
        """
        if not self._is_fitted:
            raise RuntimeError("SelfKnowledgeDetector must be fitted before detection")

        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        novelty_scores = self.autoencoder.compute_novelty_score(X)
        thresh = threshold if threshold is not None else self.knowledge_threshold
        is_novel = novelty_scores >= thresh

        if return_scores:
            return novelty_scores, is_novel
        return is_novel

    def compute_knowledge_scores(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute knowledge scores for embeddings.

        Args:
            embeddings: Query embeddings

        Returns:
            Knowledge scores (0-1, higher = more known)
        """
        if not self._is_fitted:
            raise RuntimeError("SelfKnowledgeDetector must be fitted before scoring")

        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self.autoencoder.compute_knowledge_score(
            X, normalizer=self.normalization
        )

    def compute_novelty_scores(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute novelty scores for embeddings.

        Args:
            embeddings: Query embeddings

        Returns:
            Novelty scores (0-1, higher = more novel)
        """
        return 1.0 - self.compute_knowledge_scores(embeddings)

    def fit_transform(
        self,
        known_embeddings: np.ndarray,
        query_embeddings: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Fit on known embeddings and optionally score query embeddings.

        Args:
            known_embeddings: Embeddings to train on
            query_embeddings: Optional query embeddings to score

        Returns:
            Tuple of (known_knowledge_scores, query_novelty_scores or None)
        """
        self.fit(known_embeddings)
        known_scores = self.compute_knowledge_scores(known_embeddings)

        query_scores = None
        if query_embeddings is not None:
            query_scores = self.compute_novelty_scores(query_embeddings)

        return known_scores, query_scores
