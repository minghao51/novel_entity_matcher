from typing import Optional, Union, List, Any
import tempfile
import numpy as np
from datasets import Dataset

from ..exceptions import TrainingError
from ..utils.logging_config import get_logger, suppress_third_party_loggers

try:
    from setfit import SetFitModel, Trainer, TrainingArguments
    from sklearn.linear_model import LogisticRegression

    SETFIT_AVAILABLE = True
except ImportError:
    SETFIT_AVAILABLE = False


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
        if not SETFIT_AVAILABLE:
            raise ImportError("setfit is required. Install with: pip install setfit")

        self.labels = labels
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.head_c = head_c
        self.num_iterations = num_iterations
        self.pca_dims = pca_dims
        self.skip_body_training = skip_body_training
        self.model: Optional[SetFitModel] = None
        self.pca: Optional[Any] = None
        self.is_trained = False
        self.logger = get_logger(__name__)

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

        self.model = SetFitModel.from_pretrained(self.model_name, labels=self.labels)

        texts = [item["text"] for item in training_data]
        labels_arr = np.array([item["label"] for item in training_data])

        if self.skip_body_training:
            # Skip contrastive fine-tuning; train head directly on frozen embeddings
            embeddings = self.model.model_body.encode(texts, show_progress_bar=False)

            if self.pca_dims:
                from sklearn.decomposition import PCA

                n_components = min(self.pca_dims, len(training_data) - 1, 384)
                if n_components >= len(self.labels):
                    self.pca = PCA(n_components=n_components)
                    embeddings = self.pca.fit_transform(embeddings)

            self.model.model_head = LogisticRegression(
                C=self.head_c, max_iter=10000, solver="lbfgs", class_weight="balanced"
            )
            self.model.model_head.fit(embeddings, labels_arr)
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

            # Retrain head with configurable regularization
            embeddings = self.model.model_body.encode(texts, show_progress_bar=False)

            if self.pca_dims:
                from sklearn.decomposition import PCA

                n_components = min(self.pca_dims, len(training_data) - 1, 384)
                if n_components >= len(self.labels):
                    self.pca = PCA(n_components=n_components)
                    embeddings = self.pca.fit_transform(embeddings)

            self.model.model_head = LogisticRegression(
                C=self.head_c, max_iter=10000, solver="lbfgs", class_weight="balanced"
            )
            self.model.model_head.fit(embeddings, labels_arr)

        self.is_trained = True

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        if not self.is_trained or self.model is None:
            raise TrainingError(
                "Model not trained. Call train() first.",
                details={"model_name": self.model_name},
            )

        if isinstance(texts, str):
            texts = [texts]

        if self.pca is not None:
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
        self.model.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> "SetFitClassifier":
        model = SetFitModel.from_pretrained(path)
        clf = cls(labels=model.labels)
        clf.model = model
        clf.is_trained = True
        return clf
