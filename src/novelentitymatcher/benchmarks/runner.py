"""Benchmark runner orchestrator for HuggingFace benchmarks."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ..novelty.entity_matcher import NovelEntityMatcher
from .base import EvaluationResult
from .classification import ClassificationEvaluator
from .entity_resolution import EntityResolutionEvaluator
from .loader import DatasetLoader
from .novelty import NoveltyEvaluator
from .registry import (
    DATASET_REGISTRY,
    get_dataset_config,
    get_datasets_by_task,
)

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(
        self,
        output_dir: Path | None = None,
        cache_dir: Path | None = None,
    ):
        self.output_dir = output_dir or Path("data/hf_benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DatasetLoader(cache_dir=cache_dir)

        self.er_evaluator = EntityResolutionEvaluator()
        self.clf_evaluator = ClassificationEvaluator()
        self.novelty_evaluator = NoveltyEvaluator()

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    def _write_artifact_json(
        self,
        artifact_type: str,
        section_name: str,
        threshold: float,
        payload: dict[str, Any],
    ) -> Path:
        artifact_dir = self.output_dir / "artifacts" / artifact_type
        artifact_dir.mkdir(parents=True, exist_ok=True)
        section_slug = section_name.replace("/", "__")
        threshold_slug = str(threshold).replace(".", "_")
        output_path = artifact_dir / f"{section_slug}_thr_{threshold_slug}.json"
        output_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return output_path

    def _summarize_processed_ood_results(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, float]:
        tp = sum(
            1
            for item in results
            if item["true_is_novel"] and item["predicted_is_novel"]
        )
        fp = sum(
            1
            for item in results
            if (not item["true_is_novel"]) and item["predicted_is_novel"]
        )
        tn = sum(
            1
            for item in results
            if (not item["true_is_novel"]) and (not item["predicted_is_novel"])
        )
        fn = sum(
            1
            for item in results
            if item["true_is_novel"] and (not item["predicted_is_novel"])
        )
        known_correct = sum(1 for item in results if item.get("correct_known_match"))
        known_total = sum(1 for item in results if not item["true_is_novel"])
        novel_total = sum(1 for item in results if item["true_is_novel"])
        precision = self._safe_divide(tp, tp + fp)
        recall = self._safe_divide(tp, tp + fn)
        f1 = self._safe_divide(2 * precision * recall, precision + recall)

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "novel_precision": precision,
            "novel_recall": recall,
            "novel_f1": f1,
            "known_accuracy": self._safe_divide(known_correct, known_total),
            "novel_detection_rate": self._safe_divide(tp, novel_total),
            "false_positive_novel_rate": self._safe_divide(fp, known_total),
            "overall_accuracy": self._safe_divide(tp + known_correct, len(results)),
        }

    @staticmethod
    def _pick_calibrated_threshold(
        threshold_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return max(
            threshold_summaries,
            key=lambda item: (
                item["validation"]["novel_f1"],
                item["validation"]["known_accuracy"],
                -item["validation"]["false_positive_novel_rate"],
                item["validation"]["overall_accuracy"],
                -item["threshold"],
            ),
        )

    @staticmethod
    def _is_multi_label_value(value: Any) -> bool:
        return isinstance(value, (list, tuple, set)) or (
            hasattr(value, "tolist") and not isinstance(value, (str, bytes))
        )

    @staticmethod
    def _has_label_values(value: Any) -> bool:
        if value is None:
            return False
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            return len(value) > 0
        return True

    @staticmethod
    def _extract_primary_label(value: Any) -> Any:
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return value[0]
        if isinstance(value, set):
            return sorted(value)[0]
        return value

    def _prepare_single_label_frame(
        self,
        df: pd.DataFrame,
        label_col: str,
    ) -> pd.DataFrame:
        prepared = df.copy()

        if prepared.empty:
            return prepared

        if prepared[label_col].apply(self._is_multi_label_value).any():
            prepared = prepared[
                prepared[label_col].apply(self._has_label_values)
            ].copy()
            prepared[label_col] = prepared[label_col].apply(self._extract_primary_label)

        return prepared

    def _build_label_entities(
        self,
        labels: list[Any],
        class_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        entities = []
        for index, label in enumerate(labels):
            label_id = str(label)
            display_name = (
                class_names[index]
                if class_names is not None and index < len(class_names)
                else label_id
            )
            aliases = [label_id] if display_name != label_id else []
            entities.append({"id": label_id, "name": display_name, "aliases": aliases})
        return entities

    def _create_matcher_wrapper(
        self,
        entities: list[dict],
        model: str = "potion-32m",
        mode: str = "zero-shot",
        threshold: float = 0.7,
        training_data: list[dict[str, str]] | None = None,
        regularize: bool = True,
        **fit_kwargs,
    ) -> Callable[[str], tuple[str, float]]:
        from ..core.matcher import Matcher

        matcher = Matcher(
            entities=entities,
            model=model,
            mode=mode,
            threshold=threshold,
        )
        if mode != "zero-shot":
            fit_params = {"show_progress": False, **fit_kwargs}
            if regularize and training_data:
                n = len(training_data)
                fit_params.setdefault("skip_body_training", True)
                fit_params.setdefault(
                    "head_c", 0.001 if n <= 100 else (0.01 if n <= 200 else 0.1)
                )
                pca_dims = 5 if n <= 100 else (10 if n <= 200 else 20)
                fit_params.setdefault("pca_dims", pca_dims)
            matcher.fit(training_data=training_data, mode=mode, **fit_params)

        def wrapper(text: str) -> tuple[str, float]:
            result = matcher.match(text)
            if isinstance(result, dict):
                return result.get("id", ""), result.get("score", 0.0)
            elif isinstance(result, list) and result:
                return result[0].get("id", ""), result[0].get("score", 0.0)
            return ("", 0.0)

        return wrapper

    def _create_er_matcher_wrapper(
        self,
        entities: list[dict],
        model: str = "potion-32m",
        threshold: float = 0.7,
    ) -> Callable[[str, str], float]:
        from ..core.embedding_matcher import EmbeddingMatcher

        matcher = EmbeddingMatcher(
            entities,
            model_name=model,
            threshold=threshold,
        )
        matcher.build_index()

        def wrapper(left: str, right: str) -> float:
            left_result = matcher.match(left)
            right_result = matcher.match(right)
            if left_result and right_result:
                left_id = left_result.get("id", "")
                right_id = right_result.get("id", "")
                if left_id and left_id == right_id:
                    return max(
                        left_result.get("score", 0.0), right_result.get("score", 0.0)
                    )
            return 0.0

        return wrapper

    def _create_er_embedding_similarity_fn(
        self,
        pairs_df: pd.DataFrame,
        model: str = "potion-32m",
        left_col: str = "left",
        right_col: str = "right",
    ) -> Callable[[str, str], float]:
        from sklearn.metrics.pairwise import cosine_similarity

        from ..utils.embeddings import get_cached_sentence_transformer

        all_texts = list(
            set(pairs_df[left_col].tolist() + pairs_df[right_col].tolist())
        )
        st_model = get_cached_sentence_transformer(model)
        embeddings = st_model.encode(all_texts, show_progress_bar=False)
        text_to_emb = {t: embeddings[i] for i, t in enumerate(all_texts)}

        def wrapper(left: str, right: str) -> float:
            left_emb = text_to_emb.get(left)
            right_emb = text_to_emb.get(right)
            if left_emb is not None and right_emb is not None:
                return float(cosine_similarity([left_emb], [right_emb])[0][0])
            return 0.0

        return wrapper

    def _evaluate_with_threshold(
        self,
        data: pd.DataFrame,
        matcher_fn: Callable[[str], tuple[str, float]],
        threshold: float,
        label_col: str = "label",
        text_col: str = "text",
        classes: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate classification with a fixed threshold."""
        from sklearn.metrics import accuracy_score, f1_score

        predictions = []
        confidences = []
        true_labels_raw = data[label_col].tolist()

        for _, row in data.iterrows():
            result = matcher_fn(str(row[text_col]))
            if isinstance(result, tuple):
                predictions.append(result[0])
                confidences.append(result[1])
            else:
                predictions.append(None)
                confidences.append(0.0)

        true_labels = [str(label) for label in true_labels_raw]
        pred_labels = [
            "__no_match__" if (pred is None or conf < threshold) else str(pred)
            for pred, conf in zip(predictions, confidences, strict=False)
        ]

        macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(
            true_labels, pred_labels, average="weighted", zero_division=0
        )
        accuracy = accuracy_score(true_labels, pred_labels)

        unique_labels = sorted(set(true_labels + pred_labels))
        class_names = (
            classes
            if classes and len(classes) == len(unique_labels)
            else [str(label) for label in unique_labels]
        )

        per_sample_results = pd.DataFrame(
            {
                "text": data[text_col],
                "true_label": true_labels,
                "pred_label": pred_labels,
                "confidence": confidences,
            }
        )

        return EvaluationResult(
            metrics={
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            },
            details={
                "num_samples": len(data),
                "num_classes": len(unique_labels),
                "classes": class_names,
                "threshold": threshold,
            },
            dataframe=per_sample_results,
        )

    def _auto_tune_threshold(
        self,
        data: pd.DataFrame,
        matcher_fn: Callable[[str], tuple[str, float]],
        label_col: str = "label",
        text_col: str = "text",
        classes: list[str] | None = None,
        threshold_candidates: list[float] | None = None,
    ) -> EvaluationResult:
        """Find optimal threshold by sweeping candidates and picking best accuracy."""
        if threshold_candidates is None:
            threshold_candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        best_result: EvaluationResult | None = None
        best_threshold = 0.5
        best_accuracy = 0.0

        for thresh in threshold_candidates:
            result = self._evaluate_with_threshold(
                data,
                matcher_fn,
                thresh,
                label_col=label_col,
                text_col=text_col,
                classes=classes,
            )
            if result.metrics.get("accuracy", 0.0) > best_accuracy:
                best_accuracy = result.metrics["accuracy"]
                best_threshold = thresh
                best_result = result

        if best_result is None:
            best_result = self._evaluate_with_threshold(
                data,
                matcher_fn,
                0.5,
                label_col=label_col,
                text_col=text_col,
                classes=classes,
            )

        best_result.details["optimal_threshold"] = best_threshold
        return best_result

    def _build_entities_from_pairs(
        self,
        pairs_df: pd.DataFrame,
        left_col: str = "left",
        right_col: str = "right",
        label_col: str = "label",
    ) -> list[dict]:
        entity_texts = set()
        for _, row in pairs_df.iterrows():
            entity_texts.add(str(row[left_col]))
            entity_texts.add(str(row[right_col]))

        entities = []
        for i, text in enumerate(sorted(entity_texts)):
            entities.append(
                {
                    "id": f"entity_{i}",
                    "name": text,
                    "aliases": [],
                }
            )
        return entities

    def run_entity_resolution(
        self,
        datasets: list[str] | None = None,
        model: str = "potion-32m",
        threshold: float = 0.7,
        thresholds_to_sweep: list[float] | None = None,
    ) -> pd.DataFrame:
        thresholds = thresholds_to_sweep or [threshold]
        return self.run_entity_resolution_benchmark(
            datasets=datasets,
            model=model,
            thresholds=thresholds,
        )

    def run_classification(
        self,
        datasets: list[str] | None = None,
        model: str = "potion-32m",
        mode: str = "zero-shot",
        threshold: float = 0.7,
        class_counts: list[int] | None = None,
        max_train_samples: int | None = None,
        max_test_samples: int | None = None,
        per_split: bool = True,
        regularize: bool = True,
        **fit_kwargs,
    ) -> pd.DataFrame:
        if datasets is None:
            datasets = list(get_datasets_by_task("classification").keys())

        records = []
        for name in datasets:
            if name not in DATASET_REGISTRY:
                logger.warning(f"Unknown dataset: {name}")
                continue

            config = get_dataset_config(name)
            if not config:
                continue

            try:
                data = self.loader.load_dataset(name)
                if mode != "zero-shot" and "train" not in data:
                    data = self.loader.load_dataset(name, force_redownload=True)
                test_key = "test" if "test" in data else config.split
                if test_key not in data:
                    logger.warning(f"No {test_key} split for {name}")
                    continue
                train_key = "train" if "train" in data else None
                val_key = "validation" if "validation" in data else None

                clf_df = self._prepare_single_label_frame(
                    data[test_key], config.label_column
                )
                if clf_df.empty:
                    raise ValueError(
                        "No evaluable rows remain after label normalization"
                    )
                if max_test_samples and len(clf_df) > max_test_samples:
                    clf_df = clf_df.sample(n=max_test_samples, random_state=42)

                raw_labels = sorted(clf_df[config.label_column].unique().tolist())
                class_names = None
                if config.classes:
                    class_names = [
                        config.classes[int(label)]
                        if isinstance(label, (int, str))
                        and str(label).isdigit()
                        and int(label) < len(config.classes)
                        else str(label)
                        for label in raw_labels
                    ]
                entities = self._build_label_entities(raw_labels, class_names)
                training_data = None
                if mode != "zero-shot":
                    if train_key is None:
                        raise ValueError(
                            f"Dataset {name} does not provide a train split for mode={mode}"
                        )
                    train_df = self._prepare_single_label_frame(
                        data[train_key], config.label_column
                    )
                    train_df = train_df[train_df[config.label_column].isin(raw_labels)]
                    if max_train_samples and len(train_df) > max_train_samples:
                        train_df = train_df.sample(n=max_train_samples, random_state=42)
                    training_data = [
                        {
                            "text": str(row[config.text_column]),
                            "label": str(row[config.label_column]),
                        }
                        for _, row in train_df.iterrows()
                    ]
                    if not training_data:
                        raise ValueError(
                            f"Dataset {name} yielded no training rows after label normalization"
                        )

                matcher_fn = self._create_matcher_wrapper(
                    entities,
                    model=model,
                    mode=mode,
                    threshold=0.0 if mode == "zero-shot" else threshold,
                    training_data=training_data,
                    regularize=regularize,
                    **fit_kwargs,
                )

                # Determine splits to evaluate
                splits_to_eval = {"test": (clf_df, test_key)}
                if per_split:
                    if train_key and train_key in data:
                        train_df_eval = self._prepare_single_label_frame(
                            data[train_key], config.label_column
                        )
                        train_df_eval = train_df_eval[
                            train_df_eval[config.label_column].isin(raw_labels)
                        ]
                        if not train_df_eval.empty:
                            splits_to_eval["train"] = (train_df_eval, train_key)
                    if val_key and val_key in data:
                        val_df_eval = self._prepare_single_label_frame(
                            data[val_key], config.label_column
                        )
                        val_df_eval = val_df_eval[
                            val_df_eval[config.label_column].isin(raw_labels)
                        ]
                        if not val_df_eval.empty:
                            splits_to_eval["validation"] = (val_df_eval, val_key)

                # For zero-shot mode, auto-tune threshold once on test split
                # then apply to all splits for fair comparison
                zero_shot_threshold = None
                if mode == "zero-shot":
                    tuning_df = splits_to_eval.get("test", (clf_df,))[0]
                    threshold_result = self._auto_tune_threshold(
                        tuning_df,
                        matcher_fn,
                        label_col=config.label_column,
                        text_col=config.text_column,
                        classes=class_names,
                    )
                    zero_shot_threshold = threshold_result.details.get(
                        "optimal_threshold", 0.0
                    )

                for split_name, (split_df, _split_key) in splits_to_eval.items():
                    if mode == "zero-shot" and zero_shot_threshold is not None:
                        result = self._evaluate_with_threshold(
                            split_df,
                            matcher_fn,
                            zero_shot_threshold,
                            label_col=config.label_column,
                            text_col=config.text_column,
                            classes=class_names,
                        )
                    else:
                        result = self.clf_evaluator.evaluate(
                            split_df,
                            matcher_fn=matcher_fn,
                            label_col=config.label_column,
                            text_col=config.text_column,
                            classes=class_names,
                        )

                    record = {
                        "dataset": name,
                        "model": model,
                        "mode": mode,
                        "split": split_name,
                        "num_classes": len(raw_labels),
                        "num_samples": len(split_df),
                        "accuracy": result.metrics.get("accuracy", 0.0),
                        "macro_f1": result.metrics.get("macro_f1", 0.0),
                        "weighted_f1": result.metrics.get("weighted_f1", 0.0),
                    }
                    records.append(record)

                    if (
                        class_counts
                        and len(raw_labels) >= max(class_counts)
                        and split_name == "test"
                    ):
                        from .classification import evaluate_by_class_count

                        subset_classes = raw_labels[: max(class_counts)]
                        eval_df = split_df[
                            split_df[config.label_column].isin(subset_classes)
                        ]
                        count_results = evaluate_by_class_count(
                            matcher_fn,
                            eval_df,
                            label_col=config.label_column,
                            text_col=config.text_column,
                            class_counts=class_counts,
                        )
                        for count, metrics in count_results.items():
                            record[f"classes_{count}_accuracy"] = metrics.get(
                                "accuracy", 0.0
                            )
                            record[f"classes_{count}_macro_f1"] = metrics.get(
                                "macro_f1", 0.0
                            )

            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Error evaluating %s: %s", name, e)
                records.append(
                    {
                        "dataset": name,
                        "model": model,
                        "mode": mode,
                        "split": "test",
                        "error": str(e),
                    }
                )

        return pd.DataFrame(records)

    def run_novelty(
        self,
        datasets: list[str] | None = None,
        model: str = "potion-32m",
        ood_ratio: float = 0.2,
        k_values: list[int] | None = None,
        distance_thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        if datasets is None:
            datasets = list(get_datasets_by_task("novelty").keys())

        if k_values is None:
            k_values = [3, 5, 10, 20]
        if distance_thresholds is None:
            distance_thresholds = [0.4, 0.5, 0.6, 0.7]

        records = []
        for name in datasets:
            if name not in DATASET_REGISTRY:
                logger.warning(f"Unknown dataset: {name}")
                continue

            config = get_dataset_config(name)
            if not config:
                continue

            try:
                data = self.loader.load_dataset(name)
                test_key = "test" if "test" in data else config.split
                if test_key not in data:
                    logger.warning(f"No {test_key} split for {name}")
                    continue

                clf_df = self._prepare_single_label_frame(
                    data[test_key], config.label_column
                )
                if clf_df.empty:
                    raise ValueError(
                        "No evaluable rows remain after label normalization"
                    )

                known_data, ood_data = self.novelty_evaluator.create_ood_split(
                    clf_df,
                    label_col=config.label_column,
                    ood_ratio=ood_ratio,
                )

                known_classes = sorted(
                    known_data[config.label_column].unique().tolist()
                )
                known_class_names = None
                if config.classes:
                    known_class_names = [
                        config.classes[int(label)]
                        if isinstance(label, (int, str))
                        and str(label).isdigit()
                        and int(label) < len(config.classes)
                        else str(label)
                        for label in known_classes
                    ]

                entities = self._build_label_entities(known_classes, known_class_names)

                matcher_fn = self._create_matcher_wrapper(
                    entities, model=model, mode="zero-shot", threshold=0.5
                )

                def novelty_fn(text: str, _matcher_fn=matcher_fn) -> float:
                    return 1.0 - _matcher_fn(text)[1]

                result = self.novelty_evaluator.evaluate(
                    (known_data, ood_data),
                    novelty_score_fn=novelty_fn,
                    text_col=config.text_column,
                )

                record = {
                    "dataset": name,
                    "model": model,
                    "ood_ratio": ood_ratio,
                    "num_known": len(known_data),
                    "num_ood": len(ood_data),
                    "auroc": result.metrics.get("auroc", 0.0),
                    "auprc": result.metrics.get("auprc", 0.0),
                    "detection_rate_at_1fp": result.metrics.get(
                        "detection_rate_at_1fp", 0.0
                    ),
                    "detection_rate_at_5fp": result.metrics.get(
                        "detection_rate_at_5fp", 0.0
                    ),
                    "detection_rate_at_10fp": result.metrics.get(
                        "detection_rate_at_10fp", 0.0
                    ),
                }
                records.append(record)

            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Error evaluating %s: %s", name, e)
                records.append(
                    {
                        "dataset": name,
                        "model": model,
                        "error": str(e),
                    }
                )

        return pd.DataFrame(records)

    def run_entity_resolution_benchmark(
        self,
        datasets: list[str] | None = None,
        model: str = "all-MiniLM-L6-v2",
        thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

        if datasets is None:
            datasets = list(get_datasets_by_task("entity_matching").keys())

        records = []
        for name in datasets:
            if name not in DATASET_REGISTRY:
                continue

            config = get_dataset_config(name)
            if not config:
                continue

            try:
                data = self.loader.load_dataset(name)
                test_key = "test" if "test" in data else config.split
                if test_key not in data:
                    continue

                pairs_df = data[test_key]
                matcher_fn = self._create_er_embedding_similarity_fn(
                    pairs_df, model=model
                )

                base_record = {
                    "dataset": name,
                    "model": model,
                    "num_pairs": len(pairs_df),
                    "match_rate": pairs_df["label"].mean(),
                }

                for thresh in thresholds:
                    result = self.er_evaluator.evaluate(
                        pairs_df,
                        matcher_fn=matcher_fn,
                        match_threshold=thresh,
                    )
                    record = {
                        **base_record,
                        "threshold": thresh,
                        "accuracy": result.metrics.get("accuracy", 0.0),
                        "precision": result.metrics.get("precision", 0.0),
                        "recall": result.metrics.get("recall", 0.0),
                        "f1": result.metrics.get("f1", 0.0),
                    }
                    records.append(record)

            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning(
                    "Error evaluating %s with threshold %.2f: %s", name, thresh, e
                )

        return pd.DataFrame(records)

    def run_novelty_on_processed(
        self,
        datasets: list[str] | None = None,
        model: str = "potion-32m",
        confidence_thresholds: list[float] | None = None,
        ood_ratio: float = 0.2,
        calibrate_thresholds: bool = True,
    ) -> pd.DataFrame:
        if confidence_thresholds is None:
            confidence_thresholds = [0.2, 0.3, 0.4, 0.5]

        if datasets is None:
            from ..utils.benchmark_dataset import build_processed_ood_sections

            sections = build_processed_ood_sections(ood_ratio=ood_ratio)
            datasets = [s["section"] for s in sections]

        records = []
        for section_name in datasets:
            try:
                from ..utils.benchmark_dataset import build_processed_ood_sections

                sections = build_processed_ood_sections(
                    sections=[section_name],
                    ood_ratio=ood_ratio,
                )
                if not sections:
                    continue

                section = sections[0]
                known_entities = section["known_entities"]
                training_data = section["training_data"]
                known_class_ids = section["known_class_ids"]
                val_pairs = section["val_known_pairs"] + section["val_novel_pairs"]
                test_pairs = section["test_known_pairs"] + section["test_novel_pairs"]

                if (
                    not known_entities
                    or not training_data
                    or not val_pairs
                    or not test_pairs
                ):
                    continue

                threshold_summaries = []
                for thresh in confidence_thresholds:
                    novelty_matcher = NovelEntityMatcher(
                        entities=known_entities,
                        model=model,
                        mode="zero-shot",
                        match_threshold=0.5,
                        confidence_threshold=thresh,
                        use_novelty_detector=True,
                    )
                    novelty_matcher.fit(
                        training_data=training_data,
                        mode="head-only",
                        show_progress=False,
                        num_epochs=1,
                    )

                    val_results = []
                    for pair in val_pairs:
                        result = novelty_matcher.match(
                            pair["query"],
                            existing_classes=known_class_ids,
                        )
                        is_correct_known = (
                            not pair["is_novel"]
                            and not result.is_novel
                            and result.id == pair["expected_id"]
                        )
                        val_results.append(
                            {
                                "query": pair["query"],
                                "expected_id": pair["expected_id"],
                                "true_is_novel": pair["is_novel"],
                                "predicted_id": result.predicted_id,
                                "matched_id": result.id,
                                "predicted_is_novel": result.is_novel,
                                "score": result.score,
                                "novel_score": result.novel_score,
                                "match_method": result.match_method,
                                "signals": result.signals,
                                "split": pair["split"],
                                "correct_known_match": is_correct_known,
                            }
                        )

                    test_results = []
                    for pair in test_pairs:
                        result = novelty_matcher.match(
                            pair["query"],
                            existing_classes=known_class_ids,
                        )
                        is_correct_known = (
                            not pair["is_novel"]
                            and not result.is_novel
                            and result.id == pair["expected_id"]
                        )
                        test_results.append(
                            {
                                "query": pair["query"],
                                "expected_id": pair["expected_id"],
                                "true_is_novel": pair["is_novel"],
                                "predicted_id": result.predicted_id,
                                "matched_id": result.id,
                                "predicted_is_novel": result.is_novel,
                                "score": result.score,
                                "novel_score": result.novel_score,
                                "match_method": result.match_method,
                                "signals": result.signals,
                                "split": pair["split"],
                                "correct_known_match": is_correct_known,
                            }
                        )

                    threshold_summaries.append(
                        {
                            "threshold": thresh,
                            "validation_results": val_results,
                            "test_results": test_results,
                            "validation": self._summarize_processed_ood_results(
                                val_results
                            ),
                            "test": self._summarize_processed_ood_results(test_results),
                        }
                    )

                selected = (
                    self._pick_calibrated_threshold(threshold_summaries)
                    if calibrate_thresholds and len(threshold_summaries) > 1
                    else threshold_summaries[0]
                )

                artifact_payload = {
                    "dataset": section_name,
                    "model": model,
                    "track": "ood_novelty",
                    "selected_threshold": selected["threshold"],
                    "known_class_ids": known_class_ids,
                    "heldout_class_ids": section["heldout_class_ids"],
                    "threshold_candidates": [
                        {
                            "threshold": item["threshold"],
                            "validation": item["validation"],
                            "test": item["test"],
                        }
                        for item in threshold_summaries
                    ],
                    "validation_results": selected["validation_results"],
                    "test_results": selected["test_results"],
                }
                selected_threshold = cast(float, selected["threshold"])
                selected_validation = cast(list[Any], selected["validation_results"])
                selected_test = cast(list[Any], selected["test_results"])
                artifact_path = self._write_artifact_json(
                    artifact_type="processed_ood_novelty",
                    section_name=section_name,
                    threshold=selected_threshold,
                    payload=artifact_payload,
                )

                records.append(
                    {
                        "track": "ood_novelty",
                        "dataset": section_name,
                        "section": section_name,
                        "model": model,
                        "selected_threshold": selected_threshold,
                        "num_threshold_candidates": len(threshold_summaries),
                        "num_known_classes": len(known_class_ids),
                        "num_heldout_classes": len(section["heldout_class_ids"]),
                        "num_validation_pairs": len(selected_validation),
                        "num_test_pairs": len(selected_test),
                        "validation_novel_f1": selected["validation"]["novel_f1"],  # type: ignore[index]
                        "validation_known_accuracy": selected["validation"][
                            "known_accuracy"
                        ],  # type: ignore[index]
                        "validation_false_positive_novel_rate": selected["validation"][
                            "false_positive_novel_rate"
                        ],  # type: ignore[index]
                        **selected["test"],  # type: ignore[dict-item]
                        "artifact_path": str(artifact_path),
                    }
                )

            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning(
                    "Error evaluating %s: %s (traceback logged above)",
                    section_name,
                    e,
                    exc_info=True,
                )

        return pd.DataFrame(records)

    def run_all(
        self,
        embedding_models: list[str] | None = None,
        modes: list[str] | None = None,
        thresholds: list[float] | None = None,
        class_counts: list[int] | None = None,
        ood_ratio: float = 0.2,
    ) -> dict[str, Any]:
        if embedding_models is None:
            embedding_models = ["potion-32m", "bge-base"]
        if modes is None:
            modes = ["zero-shot", "head-only"]
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

        results: dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "embedding_models": embedding_models,
                "modes": modes,
                "thresholds": thresholds,
                "class_counts": class_counts,
                "ood_ratio": ood_ratio,
            },
            "entity_resolution": [],
            "classification": [],
            "novelty": [],
        }

        for model in embedding_models:
            er_df = self.run_entity_resolution(
                model=model,
                threshold=0.7,
                thresholds_to_sweep=thresholds,
            )
            results["entity_resolution"].append(er_df.to_dict(orient="records"))

            clf_df = self.run_classification(
                model=model,
                mode="zero-shot",
                class_counts=class_counts,
            )
            results["classification"].append(clf_df.to_dict(orient="records"))

            novelty_df = self.run_novelty(
                model=model,
                ood_ratio=ood_ratio,
            )
            results["novelty"].append(novelty_df.to_dict(orient="records"))

        self._save_results(results)
        return results

    def _save_results(self, results: dict[str, Any], suffix: str = "") -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results{suffix}_{timestamp}.json"
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved results to {output_path}")
        return output_path

    def save_results(
        self,
        results: dict[str, Any],
        filename: str | None = None,
    ) -> Path:
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        else:
            output_path = Path(filename)
            if not output_path.is_absolute():
                output_path = output_path.resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return output_path

    def load_all(
        self, datasets: list[str] | None = None, force_redownload: bool = False
    ) -> dict[str, Any]:
        return self.loader.load_all(
            datasets=datasets, force_redownload=force_redownload
        )

    async def aload_all(
        self, datasets: list[str] | None = None, force_redownload: bool = False
    ) -> dict[str, Any]:
        return await self.loader.aload_all(datasets, force_redownload)
