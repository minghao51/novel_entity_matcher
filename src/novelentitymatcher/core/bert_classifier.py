"""BERT-based classifier using transformers library.

This module provides BERTClassifier, a drop-in alternative to SetFitClassifier
that uses fine-tuned BERT models for text classification. BERT classifiers provide
superior accuracy for complex pattern-driven tasks but with higher computational cost.
"""

from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import TrainingError
from ..utils.logging_config import get_logger, suppress_third_party_loggers

try:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BERTClassifier:
    """BERT-based text classifier using transformers library.

    This classifier provides a drop-in alternative to SetFitClassifier with
    identical interface. It uses fine-tuned BERT models for classification,
    offering superior accuracy for complex pattern-driven tasks.

    Example:
        >>> from novelentitymatcher.core.bert_classifier import BERTClassifier
        >>> labels = ["DE", "FR", "US"]
        >>> clf = BERTClassifier(labels=labels, model_name="distilbert-base-uncased")
        >>> training_data = [
        ...     {"text": "Germany", "label": "DE"},
        ...     {"text": "France", "label": "FR"},
        ...     {"text": "USA", "label": "US"},
        ... ]
        >>> clf.train(training_data, num_epochs=3)
        >>> prediction = clf.predict("Deutschland")  # "DE"
        >>> proba = clf.predict_proba("Deutschland")  # [0.02, 0.01, 0.97]
    """

    def __init__(
        self,
        labels: list[str],
        model_name: str = "distilbert-base-uncased",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 128,
        use_fp16: bool = True,
    ):
        """Initialize BERTClassifier.

        Args:
            labels: List of class labels for classification.
            model_name: HuggingFace model name or path. Default: "distilbert-base-uncased".
            num_epochs: Number of training epochs. Default: 3.
            batch_size: Training batch size. Default: 16.
            learning_rate: Learning rate for training. Default: 2e-5.
            max_length: Maximum sequence length for tokenization. Default: 128.
            use_fp16: Whether to use mixed precision training (faster, less memory).
                Only works on GPU. Default: True.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for BERTClassifier. "
                "Install with: pip install transformers torch"
            )

        self.labels = labels
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.is_trained = False
        self.logger = get_logger(__name__)

    def train(
        self,
        training_data: list[dict],
        num_epochs: int | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
    ):
        """Train the BERT classifier.

        Args:
            training_data: List of training examples with 'text' and 'label' keys.
            num_epochs: Number of training epochs (overrides default).
            batch_size: Batch size for training (overrides default).
            show_progress: Whether to show progress bar during training.

        Raises:
            TrainingError: If training fails or data is invalid.
        """
        # Suppress third-party library logs
        suppress_third_party_loggers()

        epochs = num_epochs or self.num_epochs
        batch = batch_size or self.batch_size

        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            raise TrainingError(
                f"Failed to load model/tokenizer: {e}",
                details={"model_name": self.model_name},
            ) from e

        # Prepare dataset
        try:
            dataset = Dataset.from_list(training_data)

            # Tokenize data
            tokenizer = self.tokenizer

            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )

            tokenized_dataset = dataset.map(tokenize_function, batched=True)

            # Convert string labels to numeric IDs
            def format_labels(example):
                example["label"] = self.label2id[example["label"]]
                return example

            tokenized_dataset = tokenized_dataset.map(format_labels)

            # Remove text column as it's not needed for training
            tokenized_dataset = tokenized_dataset.remove_columns(["text"])
            tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

            # Set format for PyTorch
            tokenized_dataset.set_format("torch")

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            raise TrainingError(
                f"Failed to prepare training data: {e}",
                details={"num_examples": len(training_data)},
            ) from e

        # Determine if we should use fp16 (disable for MPS due to compatibility)
        use_fp16 = self.use_fp16
        if use_fp16:
            try:
                import torch

                # Disable fp16 on MPS (Apple Silicon) due to PyTorch version requirements
                if torch.backends.mps.is_available():
                    import warnings

                    warnings.warn(
                        "Disabling fp16 on MPS (Apple Silicon) due to compatibility. "
                        "This may slightly slow down training but will not affect accuracy.",
                        stacklevel=2,
                    )
                    use_fp16 = False
            except ImportError:
                use_fp16 = False

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f".tmp/bert_classifier_{id(self)}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_dir=None,  # Suppress transformer logs
            logging_steps=50,
            save_strategy="no",  # Don't save checkpoints during training
            report_to="none",  # Disable wandb/tensorboard
            fp16=use_fp16,
            load_best_model_at_end=False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        # Train with optional progress tracking
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
            with tqdm(total=epochs, desc="Training BERT", unit="epoch") as pbar:
                # Store original train method
                original_train = trainer.train

                # Wrap train method to update progress bar
                def train_with_progress(*args_train, **kwargs_train):
                    result = original_train(*args_train, **kwargs_train)
                    pbar.update(epochs)
                    return result

                trainer.train = train_with_progress
                trainer.train()
        else:
            # Silent training
            trainer.train()

        self.is_trained = True

    def predict(self, texts: str | list[str]) -> str | list[str]:
        """Predict labels for input text(s).

        Args:
            texts: Single text string or list of text strings.

        Returns:
            Predicted label(s). If input is single string, returns single label.
            If input is list, returns list of labels.

        Raises:
            TrainingError: If model is not trained yet.
        """
        if not self.is_trained or self.model is None or self.tokenizer is None:
            raise TrainingError(
                "Model not trained. Call train() first.",
                details={"model_name": self.model_name},
            )

        single_input = isinstance(texts, str)
        if single_input:
            texts_list: list[str] = [texts]  # type: ignore[list-item]
        else:
            texts_list = texts  # type: ignore[assignment]

        # Tokenize
        tokenizer = self.tokenizer
        inputs = tokenizer(
            texts_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)

        # Convert to labels
        predicted_labels = [self.id2label[pred.item()] for pred in predictions]

        if single_input:
            return predicted_labels[0]
        return predicted_labels

    def predict_proba(self, text: str) -> np.ndarray:
        """Get prediction probabilities for all labels.

        Args:
            text: Input text string.

        Returns:
            NumPy array of probabilities for each label, in same order as self.labels.

        Raises:
            TrainingError: If model is not trained yet.
        """
        if not self.is_trained or self.model is None or self.tokenizer is None:
            raise TrainingError(
                "Model not trained. Call train() first.",
                details={"model_name": self.model_name},
            )

        # Tokenize
        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict with probabilities
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()[0]

    def save(self, path: str):
        """Save the trained model and tokenizer.

        Args:
            path: Directory path to save the model.

        Raises:
            TrainingError: If model is not trained yet.
        """
        if not self.is_trained or self.model is None or self.tokenizer is None:
            raise TrainingError(
                "Model not trained. Call train() first.",
                details={"model_name": self.model_name},
            )

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save labels
        labels_path = save_path / "labels.txt"
        with open(labels_path, "w") as f:
            f.write("\n".join(self.labels))

    @classmethod
    def load(cls, path: str) -> "BERTClassifier":
        """Load a trained BERTClassifier from disk.

        Args:
            path: Directory path containing the saved model.

        Returns:
            Loaded BERTClassifier instance.
        """
        load_path = Path(path)

        # Load labels
        labels_path = load_path / "labels.txt"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found at {labels_path}")

        with open(labels_path) as f:
            labels = f.read().splitlines()

        # Initialize classifier
        clf = cls(labels=labels)

        # Load model and tokenizer
        clf.tokenizer = AutoTokenizer.from_pretrained(load_path)
        clf.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        clf.is_trained = True

        return clf


# Import torch at the end to avoid issues if not available
try:
    import torch
except ImportError:
    pass
