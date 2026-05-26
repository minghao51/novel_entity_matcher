"""
Prototypical network novelty detection strategy implementation.

Computes prototype (mean embedding) for each class and detects
novelty by distance to nearest prototype.
"""

from __future__ import annotations

import json
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

        all_texts = []
        class_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for label, texts in class_texts.items():
            all_texts.extend(texts)
            class_slices[label] = (offset, offset + len(texts))
            offset += len(texts)

        all_embeddings = self.model.encode(
            all_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        for label, (start, end) in class_slices.items():
            embeddings = all_embeddings[start:end]

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

        label_list = list(self.prototypes.keys())
        proto_matrix = np.array([self.prototypes[lbl] for lbl in label_list])

        if self.distance_metric == "cosine":
            dist_matrix = cosine_distances(embeddings, proto_matrix)
        elif self.distance_metric == "euclidean":
            dist_matrix = euclidean_distances(embeddings, proto_matrix)
        elif self.distance_metric == "mahalanobis":
            dist_matrix = self._mahalanobis_batch(embeddings, proto_matrix, label_list)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        nearest_idx = np.argmin(dist_matrix, axis=1)
        min_distances = dist_matrix[np.arange(len(embeddings)), nearest_idx]

        results = []
        for i in range(len(embeddings)):
            is_novel = bool(min_distances[i] > self.distance_threshold)
            results.append(
                (is_novel, float(min_distances[i]), label_list[nearest_idx[i]])
            )

        return results

    def _mahalanobis_batch(
        self,
        embeddings: np.ndarray,
        proto_matrix: np.ndarray,
        label_list: list[str],
    ) -> np.ndarray:
        n = len(embeddings)
        m = len(label_list)
        dist_matrix = np.empty((n, m), dtype=np.float64)
        for j, label in enumerate(label_list):
            cov = self.class_covariances.get(label)
            if cov is None:
                dist_matrix[:, j] = euclidean_distances(
                    embeddings, proto_matrix[j : j + 1]
                ).ravel()
                continue
            try:
                inv_cov = np.linalg.inv(cov)
                diff = embeddings - proto_matrix[j]
                dist_matrix[:, j] = np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1))
            except np.linalg.LinAlgError:
                dist_matrix[:, j] = euclidean_distances(
                    embeddings, proto_matrix[j : j + 1]
                ).ravel()
        return dist_matrix

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

        labels = list(self.prototypes.keys())
        proto_arrays = np.stack([self.prototypes[label] for label in labels])

        save_dict: dict[str, Any] = {
            "labels": np.array(labels),
            "prototypes": proto_arrays,
            "model_name": np.array(self.model_name),
            "distance_threshold": np.array(self.distance_threshold),
            "distance_metric": np.array(self.distance_metric),
        }

        if self.class_covariances:
            cov_labels = list(self.class_covariances.keys())
            cov_arrays = np.stack(
                [self.class_covariances[label] for label in cov_labels]
            )
            save_dict["cov_labels"] = np.array(cov_labels)
            save_dict["covariances"] = cov_arrays

        np.savez(p / "model.npz", **save_dict)

    @classmethod
    def load(cls, path: str) -> PrototypicalDetector:
        p = Path(path)

        npz_path = p / "model.npz"
        json_path = p / "model.json"

        if npz_path.exists():
            return cls._load_npz(npz_path)
        if json_path.exists():
            return cls._load_json(json_path)

        raise FileNotFoundError(
            f"No model file found at {path}/ (expected model.npz or model.json)"
        )

    @classmethod
    def _load_npz(cls, npz_path: Path) -> PrototypicalDetector:
        data = np.load(npz_path, allow_pickle=False)

        labels = list(data["labels"])
        proto_arrays = data["prototypes"]

        metadata = {
            "model_name": str(data["model_name"]),
            "distance_threshold": float(data["distance_threshold"]),
            "distance_metric": str(data["distance_metric"]),
        }

        detector = cls(
            model_name=metadata["model_name"],
            distance_threshold=metadata["distance_threshold"],
            distance_metric=metadata["distance_metric"],
        )

        detector.prototypes = dict(zip(labels, proto_arrays, strict=True))

        if "covariances" in data:
            cov_labels = list(data["cov_labels"])
            cov_arrays = data["covariances"]
            detector.class_covariances = dict(zip(cov_labels, cov_arrays, strict=True))

        detector.is_trained = True
        detector.model = get_cached_sentence_transformer(metadata["model_name"])

        return detector

    @classmethod
    def _load_json(cls, json_path: Path) -> PrototypicalDetector:
        with open(json_path) as f:
            data = json.load(f)

        metadata = data.get("metadata", data)
        detector = cls(
            model_name=metadata.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            distance_threshold=metadata.get("distance_threshold", 0.5),
            distance_metric=metadata.get("distance_metric", "cosine"),
        )

        for label, proto_list in data.get("prototypes", {}).items():
            detector.prototypes[label] = np.array(proto_list, dtype=np.float64)

        for label, cov_list in data.get("covariances", {}).items():
            detector.class_covariances[label] = np.array(cov_list, dtype=np.float64)

        detector.is_trained = True
        detector.model = get_cached_sentence_transformer(detector.model_name)

        return detector
