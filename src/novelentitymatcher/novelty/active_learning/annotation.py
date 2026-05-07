from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.classifier import SetFitClassifier
from ...core.matcher import Matcher


@dataclass
class AnnotationResult:
    text: str
    assigned_label: str
    annotator: str = "human"
    metadata: dict[str, Any] = field(default_factory=dict)


class AnnotationCollector:
    def __init__(self, matcher: Matcher):
        self._matcher = matcher

    def apply_annotations(
        self,
        annotations: list[AnnotationResult],
        retrain: bool = True,
    ) -> dict[str, Any]:
        new_entities: list[dict[str, Any]] = []
        new_training_data: list[dict[str, Any]] = []
        novel_classes: set[str] = set()
        stats: dict[str, Any] = {
            "total": len(annotations),
            "novel": 0,
            "existing": 0,
            "new_classes": [],
        }

        entity_lookup = {e["id"]: e for e in getattr(self._matcher, "entities", [])}

        for ann in annotations:
            label = ann.assigned_label
            if label.startswith("novel:"):
                class_name = label[len("novel:") :]
                if class_name not in entity_lookup:
                    entry = {"id": class_name, "name": class_name}
                    new_entities.append(entry)
                    entity_lookup[class_name] = entry
                    novel_classes.add(class_name)
                new_training_data.append({"text": ann.text, "label": class_name})
                stats["novel"] += 1
            else:
                if label in entity_lookup:
                    new_training_data.append({"text": ann.text, "label": label})
                stats["existing"] += 1

        if new_entities:
            self._matcher.add_entities(new_entities)
            stats["new_classes"] = list(novel_classes)

        if retrain and new_training_data and hasattr(self._matcher, "_active_matcher"):
            active = self._matcher._active_matcher
            classifier = getattr(active, "classifier", None)
            if classifier is not None and isinstance(classifier, SetFitClassifier):
                for label in novel_classes:
                    examples = [
                        item["text"]
                        for item in new_training_data
                        if item["label"] == label
                    ]
                    if examples:
                        classifier.add_class(label, examples)
                classifier.retrain_head()

        return stats
