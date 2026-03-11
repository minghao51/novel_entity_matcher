from typing import Optional, Union, List
import numpy as np
from datasets import Dataset

from semanticmatcher.exceptions import TrainingError
from ..utils.logging_config import get_logger, suppress_third_party_loggers

try:
    from setfit import SetFitModel, Trainer, TrainingArguments

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
    ):
        if not SETFIT_AVAILABLE:
            raise ImportError("setfit is required. Install with: pip install setfit")

        self.labels = labels
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model: Optional[SetFitModel] = None
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
        # Suppress third-party library logs
        suppress_third_party_loggers()

        epochs = num_epochs or self.num_epochs
        batch = batch_size or self.batch_size

        self.model = SetFitModel.from_pretrained(self.model_name, labels=self.labels)

        dataset = Dataset.from_list(training_data)

        args = TrainingArguments(
            num_epochs=epochs,
            batch_size=batch,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
            logging_dir=None,  # Suppress transformer logs
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
        )

        # Try to use tqdm for progress tracking
        use_tqdm = False
        if show_progress:
            try:
                from tqdm.auto import tqdm

                use_tqdm = True
            except ImportError:
                # tqdm not available, training will be silent
                pass

        if use_tqdm:
            # Wrap training with tqdm progress bar
            with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
                # Store original train method
                original_train = trainer.train

                # Wrap train method to update progress bar
                def train_with_progress(*args_train, **kwargs_train):
                    result = original_train(*args_train, **kwargs_train)
                    # SetFit trains for num_epochs, so we can update after training
                    pbar.update(epochs)
                    return result

                trainer.train = train_with_progress
                trainer.train()
        else:
            # Silent training
            trainer.train()

        self.is_trained = True

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        if not self.is_trained or self.model is None:
            raise TrainingError(
                "Model not trained. Call train() first.",
                details={"model_name": self.model_name},
            )

        if isinstance(texts, str):
            texts = [texts]

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
        probs = self.model.predict_proba([text])
        return np.asarray(probs)[0]

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
