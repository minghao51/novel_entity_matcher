from typing import Optional, Union, List, Any
import tempfile

import numpy as np
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity

from ..exceptions import TrainingError
from ..utils.logging_config import get_logger, suppress_third_party_loggers
from ..utils.embeddings import (
    get_cached_sentence_transformer,
    get_cached_setfit_model,
)

from sklearn.linear_model import LogisticRegression

def _load_setfit_model_class():
    try:
        from setfit import SetFitModel
    except ImportError as exc:
        raise ImportError("setfit is required. Install with: pip install setfit") from exc
    return SetFitModel


def _load_setfit_trainer_classes():
    try:
        from setfit import Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError("setfit is required. Install with: pip install setfit") from exc
    return Trainer, TrainingArguments


class SetFitClassifier:
    """Wrapper for SetFit training and prediction."""

    def __init__(
        self,
        labels: List[str],
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        num_epochs: int = 4,
        batch_size: int = 16,
        weight_decay: float = 0.01,
        head_c: float = 1.0,
        num_iterations: int = 5,
        pca_dims: Optional[int] = None,
        skip_body_training: bool = False,
    ):
        self.labels = labels
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.head_c = head_c
        self.num_iterations = num_iterations
        self.pca_dims = pca_dims
        self.skip_body_training = skip_body_training
        self.model: Optional[Any] = None
        self.model_head: Optional[LogisticRegression] = None
        self.class_centroids: dict[str, np.ndarray] = {}
        self.pca: Optional[Any] = None
        self.is_trained = False
        self.logger = get_logger(__name__)
        self._use_sentence_transformer_fallback = False

    def train(
        self,
        training_data: List[dict],
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ):
        """Train the classifier.

        Args:
            training_data: List of training examples with 'text' and 'label' keys
            num_epochs: Number of training epochs (overrides default)
            batch_size: Batch size for training (overrides default)
            show_progress: Whether to show progress bar during training
        """
        suppress_third_party_loggers()

        epochs = num_epochs or self.num_epochs
        batch = batch_size or self.batch_size

        texts = [item["text"] for item in training_data]
        labels_arr = np.array([item["label"] for item in training_data])

        if self.skip_body_training:
            self.model = get_cached_sentence_transformer(self.model_name)
            self._use_sentence_transformer_fallback = True
            embeddings = self.model.encode(texts, show_progress_bar=False)
            self._train_fallback_head(embeddings, labels_arr, training_data)
        else:
            try:
                self.model = get_cached_setfit_model(self.model_name, labels=self.labels)
                self._use_sentence_transformer_fallback = False
                Trainer, TrainingArguments = _load_setfit_trainer_classes()
            except ImportError as exc:
                self.logger.warning(
                    "SetFit training unavailable for %s; falling back to "
                    "sentence-transformer embeddings + logistic head: %s",
                    self.model_name,
                    exc,
                )
                self.model = get_cached_sentence_transformer(self.model_name)
                self._use_sentence_transformer_fallback = True
                embeddings = self.model.encode(texts, show_progress_bar=False)
                self._train_fallback_head(embeddings, labels_arr, training_data)
            else:
                dataset = Dataset.from_list(training_data)

                args = TrainingArguments(
                    output_dir=tempfile.mkdtemp(prefix="novelentitymatcher-setfit-"),
                    num_epochs=epochs,
                    batch_size=batch,
                    body_learning_rate=2e-5,
                    head_learning_rate=1e-3,
                    save_strategy="no",
                    report_to="none",
                    logging_dir=None,
                    l2_weight=self.weight_decay,
                    num_iterations=self.num_iterations,
                )

                trainer = Trainer(
                    model=self.model,
                    args=args,
                    train_dataset=dataset,
                )

                if show_progress:
                    try:
                        from tqdm.auto import tqdm

                        with tqdm(total=epochs, desc="Training", unit="epoch"):
                            trainer.train()
                    except ImportError:
                        trainer.train()
                else:
                    trainer.train()

                embeddings = self.model.model_body.encode(texts, show_progress_bar=False)
                self._train_logistic_head(embeddings, labels_arr, training_data)

        self.is_trained = True

    def _train_fallback_head(
        self,
        embeddings: np.ndarray,
        labels_arr: np.ndarray,
        training_data: List[dict],
    ) -> None:
        embeddings = np.asarray(embeddings)
        if self.pca_dims:
            from sklearn.decomposition import PCA

            n_components = min(
                self.pca_dims,
                len(training_data) - 1,
                embeddings.shape[1],
            )
            if n_components >= len(self.labels):
                self.pca = PCA(n_components=n_components)
                embeddings = self.pca.fit_transform(embeddings)

        self.class_centroids = {}
        for label in self.labels:
            label_embeddings = embeddings[labels_arr == label]
            if len(label_embeddings) == 0:
                continue
            self.class_centroids[label] = label_embeddings.mean(axis=0)

    def _train_logistic_head(
        self,
        embeddings: np.ndarray,
        labels_arr: np.ndarray,
        training_data: List[dict],
    ) -> None:
        embeddings = np.asarray(embeddings)
        if self.pca_dims:
            from sklearn.decomposition import PCA

            n_components = min(self.pca_dims, len(training_data) - 1, embeddings.shape[1])
            if n_components >= len(self.labels):
                self.pca = PCA(n_components=n_components)
                embeddings = self.pca.fit_transform(embeddings)

        model_head = LogisticRegression(
            C=self.head_c, max_iter=10000, solver="lbfgs", class_weight="balanced"
        )
        model_head.fit(embeddings, labels_arr)
        if self._use_sentence_transformer_fallback:
            self.model_head = model_head
        else:
            self.model.model_head = model_head

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        if not self.is_trained or self.model is None:
            raise TrainingError(
                "Model not trained. Call train() first.",
                details={"model_name": self.model_name},
            )

        if isinstance(texts, str):
            texts = [texts]

        if self._use_sentence_transformer_fallback:
            probs = np.asarray([self.predict_proba(text) for text in texts])
            predictions = np.asarray(self.labels)[np.argmax(probs, axis=1)]
        elif self.pca is not None:
            embeddings = self.model.model_body.encode(texts, show_progress_bar=False)
            embeddings = self.pca.transform(embeddings)
            predictions = self.model.model_head.predict(embeddings)
        else:
            predictions = self.model.predict(texts)

        if len(predictions) == 1:
            return predictions[0]
        return predictions.tolist()

    def predict_proba(self, text: str) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise TrainingError(
                "Model not trained. Call train() first.",
                details={"model_name": self.model_name},
            )
        if self._use_sentence_transformer_fallback:
            embeddings = self.model.encode([text], show_progress_bar=False)
            if self.pca is not None:
                embeddings = self.pca.transform(embeddings)
            centroid_matrix = np.asarray(
                [self.class_centroids[label] for label in self.labels]
            )
            similarities = cosine_similarity(np.asarray(embeddings), centroid_matrix)[0]
            stabilized = (similarities - np.max(similarities)) * 10.0
            probs = np.exp(stabilized)
            probs /= probs.sum()
            return probs
        if self.pca is not None:
            embeddings = self.model.model_body.encode([text], show_progress_bar=False)
            embeddings = self.pca.transform(embeddings)
            probs = self.model.model_head.predict_proba(embeddings)
            # Reorder probabilities to match self.labels order
            head_classes = list(self.model.model_head.classes_)
            probs = np.asarray(probs)[0]
            reordered = np.zeros(len(self.labels))
            for i, label in enumerate(self.labels):
                if label in head_classes:
                    reordered[i] = probs[head_classes.index(label)]
            return reordered
        else:
            probs = self.model.predict_proba([text])
            probs = np.asarray(probs)[0]
            # Reorder probabilities to match self.labels order
            head_classes = list(self.model.model_head.classes_)
            if head_classes != self.labels:
                reordered = np.zeros(len(self.labels))
                for i, label in enumerate(self.labels):
                    if label in head_classes:
                        reordered[i] = probs[head_classes.index(label)]
                return reordered
            return probs

    def save(self, path: str):
        if not self.is_trained or self.model is None:
            raise TrainingError(
                "Model not trained. Call train() first.",
                details={"model_name": self.model_name},
            )
        if self._use_sentence_transformer_fallback:
            from pathlib import Path
            import json
            import joblib

            output_dir = Path(path)
            output_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(
                {
                    "model_head": self.model_head,
                    "pca": self.pca,
                    "class_centroids": self.class_centroids,
                },
                output_dir / "fallback_artifacts.pkl",
            )
            with open(output_dir / "fallback_metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_name": self.model_name,
                        "labels": self.labels,
                        "num_epochs": self.num_epochs,
                        "batch_size": self.batch_size,
                        "weight_decay": self.weight_decay,
                        "head_c": self.head_c,
                        "num_iterations": self.num_iterations,
                        "pca_dims": self.pca_dims,
                        "skip_body_training": self.skip_body_training,
                        "use_sentence_transformer_fallback": True,
                    },
                    f,
                    indent=2,
                )
            return
        self.model.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "SetFitClassifier":
        from pathlib import Path
        import json

        model_path = Path(path)
        fallback_metadata = model_path / "fallback_metadata.json"
        if fallback_metadata.exists():
            import joblib

            with open(fallback_metadata, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            clf = cls(
                labels=list(metadata["labels"]),
                model_name=metadata["model_name"],
                num_epochs=metadata["num_epochs"],
                batch_size=metadata["batch_size"],
                weight_decay=metadata["weight_decay"],
                head_c=metadata["head_c"],
                num_iterations=metadata["num_iterations"],
                pca_dims=metadata["pca_dims"],
                skip_body_training=metadata["skip_body_training"],
            )
            clf.model = get_cached_sentence_transformer(clf.model_name)
            clf._use_sentence_transformer_fallback = True
            artifacts = joblib.load(model_path / "fallback_artifacts.pkl")
            clf.model_head = artifacts.get("model_head")
            clf.pca = artifacts.get("pca")
            clf.class_centroids = {
                str(label): np.asarray(centroid)
                for label, centroid in artifacts.get("class_centroids", {}).items()
            }
            clf.is_trained = True
            return clf

        SetFitModel = _load_setfit_model_class()
        model = SetFitModel.from_pretrained(path)
        clf = cls(labels=model.labels)
        clf.model = model
        clf.is_trained = True
        return clf
