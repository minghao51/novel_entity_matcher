"""
One-Class SVM novelty detection strategy implementation.

Uses sklearn's OneClassSVM with sentence transformer embeddings
to detect novel entities without requiring negative examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.svm import OneClassSVM

from novelentitymatcher.utils.logging_config import get_logger
from novelentitymatcher.utils.embeddings import get_cached_sentence_transformer

logger = get_logger(__name__)


class OneClassSVMDetector:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale",
    ):
        self.model_name = model_name
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

        self.model: Optional[SentenceTransformer] = None
        self.oc_svm: Optional[OneClassSVM] = None
        self.known_embeddings: Optional[np.ndarray] = None
        self.is_trained = False

    def train(
        self,
        known_entities: List[str],
        show_progress: bool = False,
    ) -> None:
        if not known_entities:
            raise ValueError("known_entities cannot be empty")

        if show_progress:
            logger.info(f"Loading sentence transformer model: {self.model_name}")

        self.model = get_cached_sentence_transformer(self.model_name)

        if show_progress:
            logger.info(f"Encoding {len(known_entities)} known entities...")

        self.known_embeddings = self.model.encode(
            known_entities,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        if show_progress:
            logger.info(f"Training One-Class SVM (nu={self.nu}, kernel={self.kernel})...")

        self.oc_svm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.oc_svm.fit(self.known_embeddings)

        self.is_trained = True

        if show_progress:
            logger.info("Training complete!")

    def is_novel(self, text: str) -> Tuple[bool, float]:
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling is_novel()")

        if self.model is None or self.oc_svm is None:
            raise RuntimeError("Model or SVM not initialized")

        embedding = self.model.encode([text], convert_to_numpy=True)

        prediction = self.oc_svm.predict(embedding)[0]
        is_novel = bool(prediction == -1)

        distance = self.oc_svm.decision_function(embedding)[0]
        confidence = float(np.clip(-distance, 0.0, 1.0))

        return is_novel, confidence

    def score_batch(self, texts: List[str]) -> List[Tuple[bool, float]]:
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling score_batch()")

        if self.model is None or self.oc_svm is None:
            raise RuntimeError("Model or SVM not initialized")

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        predictions = self.oc_svm.predict(embeddings)
        distances = self.oc_svm.decision_function(embeddings)

        results = []
        for pred, dist in zip(predictions, distances):
            is_novel = bool(pred == -1)
            confidence = float(np.clip(-dist, 0.0, 1.0))
            results.append((is_novel, confidence))

        return results

    def save(self, path: str) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained detector")

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        import joblib

        joblib.dump(self.oc_svm, p / "oc_svm_model.pkl")
        np.save(p / "known_embeddings.npy", self.known_embeddings)

        metadata = {
            "model_name": self.model_name,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "is_trained": self.is_trained,
            "num_known_entities": len(self.known_embeddings)
            if self.known_embeddings is not None
            else 0,
        }

        import json

        with open(p / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OneClassSVMDetector":
        p = Path(path)

        import json

        with open(p / "metadata.json", "r") as f:
            metadata = json.load(f)

        detector = cls(
            model_name=metadata["model_name"],
            nu=metadata["nu"],
            kernel=metadata["kernel"],
            gamma=metadata["gamma"],
        )

        import joblib

        detector.oc_svm = joblib.load(p / "oc_svm_model.pkl")
        detector.known_embeddings = np.load(p / "known_embeddings.npy")
        detector.is_trained = metadata["is_trained"]
        detector.model = get_cached_sentence_transformer(metadata["model_name"])

        return detector

    def get_support_vectors_info(self) -> Dict[str, Any]:
        if not self.is_trained or self.oc_svm is None:
            return {}

        n_support = (
            self.oc_svm.n_support_[0] if hasattr(self.oc_svm, "n_support_") else 0
        )

        return {
            "n_support_vectors": int(n_support),
            "dual_coef_mean": float(np.mean(np.abs(self.oc_svm.dual_coef_)))
            if hasattr(self.oc_svm, "dual_coef_")
            else 0.0,
            "intercept": float(self.oc_svm.intercept_[0])
            if hasattr(self.oc_svm, "intercept_")
            else 0.0,
        }
