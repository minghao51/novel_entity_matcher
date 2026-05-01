from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.matcher import Matcher
from .schemas import NovelClassDiscoveryReport, NovelSampleMetadata, NovelSampleReport
from .storage.persistence import export_summary, save_proposals


class DiscoveryBase:
    """Shared base for novelty-aware matching and discovery orchestration."""

    matcher: Matcher
    entities: list[dict[str, Any]]
    acceptance_threshold: float
    use_novelty_detector: bool
    output_dir: str
    auto_save: bool
    review_manager: Any

    def get_reference_corpus(self) -> dict[str, Any]:
        return self.matcher.get_reference_corpus()

    def _derive_existing_classes(
        self, existing_classes: list[str] | None = None
    ) -> list[str]:
        from ..pipeline.discovery_support import derive_existing_classes

        return derive_existing_classes(
            entities=self.entities,
            get_reference_corpus=self.get_reference_corpus,
            existing_classes=existing_classes,
        )

    def _coerce_novel_sample_report(self, report: Any) -> NovelSampleReport:
        if isinstance(report, NovelSampleReport):
            return report

        samples = [
            sample
            if isinstance(sample, NovelSampleMetadata)
            else NovelSampleMetadata(
                text=str(getattr(sample, "text", "")),
                index=int(getattr(sample, "index", 0)),
                confidence=float(getattr(sample, "confidence", 0.0)),
                predicted_class=str(getattr(sample, "predicted_class", "unknown")),
                novelty_score=getattr(sample, "novelty_score", None),
                cluster_id=getattr(sample, "cluster_id", None),
                signals=dict(getattr(sample, "signals", {})),
            )
            for sample in getattr(report, "novel_samples", [])
        ]
        return NovelSampleReport(
            novel_samples=samples,
            detection_strategies=list(getattr(report, "detection_strategies", [])),
            config=dict(getattr(report, "config", {})),
            signal_counts=dict(getattr(report, "signal_counts", {})),
        )

    def _get_matcher_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "matcher_type": self.matcher.__class__.__name__,
        }
        if hasattr(self.matcher, "model_name"):
            config["model"] = str(self.matcher.model_name)
        if hasattr(self.matcher, "threshold"):
            config["threshold"] = str(self.matcher.threshold)
        if hasattr(self.matcher, "_training_mode"):
            config["mode"] = self.matcher._training_mode
        return config

    def _build_discovery_report(
        self,
        *,
        pipeline_result: Any,
        detection_config_dump: dict[str, Any],
        existing_classes: list[str] | None,
        context: str | None,
    ) -> NovelClassDiscoveryReport:
        discovery_id = str(uuid.uuid4())[:8]
        known_classes = self._derive_existing_classes(existing_classes)
        novel_sample_report = self._coerce_novel_sample_report(
            pipeline_result.context.artifacts["novel_sample_report"]
        )
        discovery_clusters = pipeline_result.context.artifacts.get(
            "discovery_clusters", []
        )
        class_proposals = pipeline_result.context.artifacts.get("class_proposals")

        report = NovelClassDiscoveryReport(
            discovery_id=discovery_id,
            timestamp=datetime.now(timezone.utc),
            matcher_config=self._get_matcher_config(),
            detection_config=detection_config_dump,
            novel_sample_report=novel_sample_report,
            discovery_clusters=discovery_clusters,
            class_proposals=class_proposals,
            diagnostics={
                "stage_metadata": pipeline_result.context.metadata,
            },
            metadata={
                "num_queries": len(pipeline_result.context.inputs),
                "num_existing_classes": len(known_classes),
                "num_novel_samples": len(novel_sample_report.novel_samples),
                "num_discovery_clusters": len(discovery_clusters),
                "context": context,
                "pipeline_stage_metadata": pipeline_result.context.metadata,
            },
        )
        return report

    def _finalize_report(
        self, report: NovelClassDiscoveryReport
    ) -> NovelClassDiscoveryReport:
        if self.auto_save:
            output_file = save_proposals(report, output_dir=self.output_dir)
            report.output_file = output_file
            summary_path = output_file.replace(
                f".{output_file.split('.')[-1]}",
                "_summary.md",
            )
            export_summary(report, summary_path)
            report.metadata["summary_file"] = summary_path
        return report

    def export_metrics(
        self,
        format: str = "json",
        path: str | None = None,
    ) -> Path:
        from ..pipeline.discovery_support import export_pipeline_metrics

        metrics: dict[str, Any] = {
            "num_entities": len(self.entities),
            "acceptance_threshold": self.acceptance_threshold,
            "use_novelty_detector": bool(self.use_novelty_detector),
        }
        if hasattr(self.matcher, "model_name"):
            metrics["model_name"] = str(self.matcher.model_name)
        if hasattr(self.matcher, "_training_mode"):
            metrics["training_mode"] = self.matcher._training_mode

        return export_pipeline_metrics(metrics=metrics, format=format, path=path)
