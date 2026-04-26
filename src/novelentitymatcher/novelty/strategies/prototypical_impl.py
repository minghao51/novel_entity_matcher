"""
Prototypical network novelty detection strategy implementation.

Computes prototype (mean embedding) for each class and detects
novelty by distance to nearest prototype.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from novelentitymatcher.utils.embeddings import get_cached_sentence_transformer
from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)


class PrototypicalDetector:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_threshold: float = 0.5,
        distance_metric: str = "cosine",
    ):
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.distance_metric = distance_metric

        self.model: SentenceTransformer | None = None
        self.prototypes: dict[str, np.ndarray] = {}
        self.class_covariances: dict[str, np.ndarray] = {}
        self.is_trained = False

    def train(
        self,
        training_data: list[dict[str, str]],
        show_progress: bool = False,
    ) -> None:
        if not training_data:
            raise ValueError("training_data cannot be empty")

        for item in training_data:
            if "text" not in item or "label" not in item:
                raise ValueError(
                    "Each item in training_data must have 'text' and 'label' keys"
                )

        if show_progress:
            logger.info(f"Loading sentence transformer model: {self.model_name}")

        self.model = get_cached_sentence_transformer(self.model_name)

        class_texts: dict[str, list[str]] = {}
        for item in training_data:
            label = item["label"]
            text = item["text"]
            if label not in class_texts:
                class_texts[label] = []
            class_texts[label].append(text)

        if show_progress:
            logger.info(f"Computing prototypes for {len(class_texts)} classes...")

        for label, texts in class_texts.items():
            if show_progress:
                logger.info(f"  Encoding {len(texts)} examples for class '{label}'...")

            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            prototype = np.mean(embeddings, axis=0)
            self.prototypes[label] = prototype

            if self.distance_metric == "mahalanobis":
                centered = embeddings - prototype
                cov = np.cov(centered.T)
                cov += np.eye(cov.shape[0]) * 1e-6
                self.class_covariances[label] = cov

        self.is_trained = True

        if show_progress:
            logger.info(
                f"Training complete! Computed {len(self.prototypes)} prototypes."
            )

    def is_novel(self, text: str) -> tuple[bool, float, str | None]:
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling is_novel()")

        if self.model is None or not self.prototypes:
            raise RuntimeError("Model or prototypes not initialized")

        embedding = self.model.encode([text], convert_to_numpy=True)[0]

        nearest_label = None
        min_distance = float("inf")

        for label, prototype in self.prototypes.items():
            distance = self._compute_distance(embedding, prototype, label)

            if distance < min_distance:
                min_distance = distance
                nearest_label = label

        is_novel = min_distance > self.distance_threshold

        return is_novel, float(min_distance), nearest_label

    def score_batch(
        self,
        texts: list[str],
    ) -> list[tuple[bool, float, str | None]]:
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling score_batch()")

        if self.model is None or not self.prototypes:
            raise RuntimeError("Model or prototypes not initialized")

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        results = []
        for embedding in embeddings:
            nearest_label = None
            min_distance = float("inf")

            for label, prototype in self.prototypes.items():
                distance = self._compute_distance(embedding, prototype, label)

                if distance < min_distance:
                    min_distance = distance
                    nearest_label = label

            is_novel = min_distance > self.distance_threshold
            results.append((is_novel, float(min_distance), nearest_label))

        return results

    def _compute_distance(
        self,
        embedding: np.ndarray,
        prototype: np.ndarray,
        label: str,
    ) -> float:
        emb_reshaped = embedding.reshape(1, -1)
        proto_reshaped = prototype.reshape(1, -1)

        if self.distance_metric == "cosine":
            distances = cosine_distances(emb_reshaped, proto_reshaped)
            return float(distances[0, 0])

        elif self.distance_metric == "euclidean":
            distances = euclidean_distances(emb_reshaped, proto_reshaped)
            return float(distances[0, 0])

        elif self.distance_metric == "mahalanobis":
            from scipy.spatial.distance import mahalanobis

            cov = self.class_covariances.get(label)
            if cov is None:
                distances = euclidean_distances(emb_reshaped, proto_reshaped)
                return float(distances[0, 0])

            try:
                inv_cov = np.linalg.inv(cov)
                distance = mahalanobis(embedding, prototype, inv_cov)
                return float(distance)
            except np.linalg.LinAlgError:
                distances = euclidean_distances(emb_reshaped, proto_reshaped)
                return float(distances[0, 0])

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def get_prototype_info(self) -> dict[str, dict[str, Any]]:
        info = {}

        for label, prototype in self.prototypes.items():
            info[label] = {
                "prototype_norm": float(np.linalg.norm(prototype)),
                "prototype_mean": float(np.mean(prototype)),
                "prototype_std": float(np.std(prototype)),
                "has_covariance": label in self.class_covariances,
            }

        return info

    def save(self, path: str) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained detector")

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        import json

        prototypes_serializable = {
            label: arr.tolist() for label, arr in self.prototypes.items()
        }
        with open(p / "prototypes.json", "w") as f:
            json.dump(prototypes_serializable, f)

        if self.class_covariances:
            cov_serializable = {
                label: arr.tolist() for label, arr in self.class_covariances.items()
            }
            with open(p / "covariances.json", "w") as f:
                json.dump(cov_serializable, f)

        metadata = {
            "model_name": self.model_name,
            "distance_threshold": self.distance_threshold,
            "distance_metric": self.distance_metric,
            "is_trained": self.is_trained,
            "num_classes": len(self.prototypes),
            "class_names": list(self.prototypes.keys()),
        }

        with open(p / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> PrototypicalDetector:
        p = Path(path)

        import json

        with open(p / "metadata.json") as f:
            metadata = json.load(f)

        detector = cls(
            model_name=metadata["model_name"],
            distance_threshold=metadata["distance_threshold"],
            distance_metric=metadata["distance_metric"],
        )

        with open(p / "prototypes.json") as f:
            proto_data = json.load(f)
        detector.prototypes = {
            label: np.array(arr) for label, arr in proto_data.items()
        }

        cov_path = p / "covariances.json"
        if cov_path.exists():
            with open(cov_path) as f:
                cov_data = json.load(f)
            detector.class_covariances = {
                label: np.array(arr) for label, arr in cov_data.items()
            }

        detector.is_trained = metadata["is_trained"]
        detector.model = get_cached_sentence_transformer(metadata["model_name"])

        return detector
