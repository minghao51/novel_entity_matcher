"""
SetFit contrastive novelty detection strategy implementation.

Uses SetFit with contrastive loss for novelty detection with
few-shot learning capability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
import numpy as np

from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)


class SetFitDetector:
    def __init__(
        self,
        known_entities: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_epochs: int = 4,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        margin: float = 0.5,
        _allow_empty_known_entities: bool = False,
    ):
        if not known_entities and not _allow_empty_known_entities:
            raise ValueError("known_entities cannot be empty")

        self.known_entities = known_entities
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.margin = margin

        self.model: Optional[Any] = None
        self.novelty_threshold: Optional[float] = None
        self.known_embeddings: Optional[np.ndarray] = None
        self.is_trained = False

    def train(
        self,
        synthetic_novels: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> None:
        try:
            from setfit import SetFitModel, SetFitTrainer
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "SetFit is not installed. Install with: pip install setfit"
            )

        if show_progress:
            logger.info(f"Loading SetFit model: {self.model_name}")

        self.model = SetFitModel.from_pretrained(self.model_name)

        train_texts = []
        train_labels = []

        for entity in self.known_entities:
            train_texts.append(entity)
            train_labels.append(0)

        synthetic_novels = self._prepare_synthetic_novels(synthetic_novels)

        for novel in synthetic_novels:
            train_texts.append(novel)
            train_labels.append(1)

        if show_progress:
            logger.info(
                f"Training with {len(train_texts)} examples ({len(self.known_entities)} known, {len(synthetic_novels)} novel)..."
            )

        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=train_dataset,
            num_iterations=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

        trainer.train()

        if show_progress:
            logger.info("Calibrating novelty threshold...")

        self.known_embeddings = self._encode_texts(
            self.known_entities, show_progress_bar=False
        )

        from sklearn.metrics.pairwise import cosine_distances

        distances = cosine_distances(self.known_embeddings)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]

        if len(upper_tri) > 0:
            self.novelty_threshold = float(np.percentile(upper_tri, 95))
        else:
            self.novelty_threshold = 0.5

        self.is_trained = True

        if show_progress:
            logger.info(f"Training complete! Threshold: {self.novelty_threshold:.4f}")

    def is_novel(self, text: str):
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling is_novel()")

        if (
            self.model is None
            or self.known_embeddings is None
            or self.novelty_threshold is None
        ):
            raise RuntimeError("Model, embeddings, or threshold not initialized")

        embedding = self._encode_texts([text])[0]

        from sklearn.metrics.pairwise import cosine_distances

        distances = cosine_distances([embedding], self.known_embeddings)[0]
        min_distance = float(np.min(distances))

        is_novel = bool(min_distance > self.novelty_threshold)

        confidence = 1.0 / (
            1.0 + np.exp(-5.0 * (min_distance - self.novelty_threshold))
        )

        return is_novel, float(confidence)

    def score_batch(self, texts: List[str]):
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before calling score_batch()")

        if (
            self.model is None
            or self.known_embeddings is None
            or self.novelty_threshold is None
        ):
            raise RuntimeError("Model, embeddings, or threshold not initialized")

        embeddings = self._encode_texts(texts)

        from sklearn.metrics.pairwise import cosine_distances

        distances = cosine_distances(embeddings, self.known_embeddings)
        min_distances = np.min(distances, axis=1)

        results = []
        for min_distance in min_distances:
            is_novel = bool(min_distance > self.novelty_threshold)
            confidence = 1.0 / (
                1.0 + np.exp(-5.0 * (float(min_distance) - self.novelty_threshold))
            )
            results.append((is_novel, float(confidence)))

        return results

    def save(self, path: str) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained detector")

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            model_path = p / "setfit_model"
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(model_path)
            else:
                self.model.save_model(str(model_path))

        if self.known_embeddings is not None:
            np.save(p / "known_embeddings.npy", self.known_embeddings)

        metadata = {
            "model_name": self.model_name,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "margin": self.margin,
            "novelty_threshold": self.novelty_threshold,
            "is_trained": self.is_trained,
            "num_known_entities": len(self.known_entities),
        }

        import json

        with open(p / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SetFitDetector":
        p = Path(path)

        import json

        with open(p / "metadata.json", "r") as f:
            metadata = json.load(f)

        detector = cls(
            known_entities=[],
            model_name=metadata["model_name"],
            num_epochs=metadata["num_epochs"],
            batch_size=metadata["batch_size"],
            learning_rate=metadata["learning_rate"],
            margin=metadata["margin"],
            _allow_empty_known_entities=True,
        )

        try:
            from setfit import SetFitModel

            detector.model = SetFitModel.from_pretrained(str(p / "setfit_model"))
        except ImportError:
            raise ImportError(
                "SetFit is not installed. Install with: pip install setfit"
            )

        detector.known_embeddings = np.load(p / "known_embeddings.npy")
        detector.novelty_threshold = metadata["novelty_threshold"]
        detector.is_trained = metadata["is_trained"]
        detector.known_entities = [""] * (metadata.get("num_known_entities", 0))

        return detector

    def generate_synthetic_novels(
        self,
        num_samples: int = 100,
        augmentation_methods: Optional[List[str]] = None,
    ) -> List[str]:
        if augmentation_methods is None:
            augmentation_methods = ["typos", "case_change", "spacing", "substring"]

        synthetic = []

        import random

        for _ in range(num_samples):
            base = random.choice(self.known_entities)
            method = random.choice(augmentation_methods)

            if method == "typos":
                synthetic.append(self._add_typos(base))
            elif method == "case_change":
                synthetic.append(self._change_case(base))
            elif method == "spacing":
                synthetic.append(self._modify_spacing(base))
            elif method == "substring":
                synthetic.append(self._create_substring_variant(base))
            else:
                synthetic.append(base)

        return synthetic

    def _prepare_synthetic_novels(
        self, synthetic_novels: Optional[List[str]]
    ) -> List[str]:
        novels = [text for text in (synthetic_novels or []) if text]
        if novels:
            return novels

        target_count = max(2, min(len(self.known_entities), 8))
        known_set = set(self.known_entities)
        generated: List[str] = []
        attempts = 0
        max_attempts = target_count * 10

        while len(generated) < target_count and attempts < max_attempts:
            attempts += 1
            candidate = self.generate_synthetic_novels(num_samples=1)[0]
            if candidate in known_set or candidate in generated:
                continue
            generated.append(candidate)

        while len(generated) < target_count:
            base = self.known_entities[len(generated) % len(self.known_entities)]
            candidate = f"{base} novel {len(generated) + 1}"
            if candidate not in generated:
                generated.append(candidate)

        return generated

    def _encode_texts(
        self, texts: List[str], show_progress_bar: Optional[bool] = None
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not initialized")

        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=show_progress_bar,
            )
        except TypeError:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            )

        return np.asarray(embeddings)

    def _add_typos(self, text: str, num_typos: int = 1) -> str:
        import random

        chars = list(text)
        for _ in range(num_typos):
            if chars:
                idx = random.randint(0, len(chars) - 1)
                replacements = {
                    "a": "s",
                    "s": "a",
                    "d": "s",
                    "e": "r",
                    "r": "e",
                    "t": "y",
                    "y": "t",
                    "i": "o",
                    "o": "i",
                    "n": "m",
                }
                if chars[idx] in replacements:
                    chars[idx] = replacements[chars[idx]]
        return "".join(chars)

    def _change_case(self, text: str) -> str:
        import random

        methods = [str.upper, str.lower, str.title]
        return random.choice(methods)(text)

    def _modify_spacing(self, text: str) -> str:
        import random

        if " " in text:
            parts = text.split(" ")
            if random.random() < 0.5 and len(parts) > 1:
                idx = random.randint(0, len(parts) - 2)
                parts[idx] = parts[idx] + parts[idx + 1]
                parts.pop(idx + 1)
            else:
                idx = random.randint(0, len(parts) - 1)
                if len(parts[idx]) > 1:
                    split_idx = random.randint(1, len(parts[idx]) - 1)
                    parts.insert(idx + 1, parts[idx][split_idx:])
                    parts[idx] = parts[idx][:split_idx]

            return " ".join(parts)

        return text

    def _create_substring_variant(self, text: str) -> str:
        import random

        if len(text) > 3:
            start = random.randint(0, len(text) // 2)
            end = random.randint(len(text) // 2 + 1, len(text))
            return text[start:end]

        return text
