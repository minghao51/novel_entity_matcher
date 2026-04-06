"""Tests for persistence layer (save_proposals, list_proposals, export_summary)."""

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from novelentitymatcher.novelty.schemas import (
    ClassProposal,
    DiscoveryCluster,
    NovelClassAnalysis,
    NovelClassDiscoveryReport,
    NovelSampleMetadata,
    NovelSampleReport,
    ProposalReviewRecord,
)
from novelentitymatcher.novelty.storage import persistence


def _make_discovery_report(with_proposals: bool = True) -> NovelClassDiscoveryReport:
    """Create a valid discovery report for testing."""
    novel_samples = [
        NovelSampleMetadata(
            text="quantum biology pathway",
            index=0,
            confidence=0.4,
            predicted_class="physics",
            novelty_score=0.9,
            signals={"confidence": True},
        ),
        NovelSampleMetadata(
            text="quantum biology proteins",
            index=1,
            confidence=0.35,
            predicted_class="biology",
            novelty_score=0.92,
            signals={"confidence": True},
        ),
    ]

    clusters = [
        DiscoveryCluster(
            cluster_id=0,
            sample_indices=[0, 1],
            sample_count=2,
            example_texts=["quantum biology pathway", "quantum biology proteins"],
            keywords=["quantum", "biology"],
            mean_novelty_score=0.91,
            mean_confidence=0.375,
        )
    ]

    review_records = [
        ProposalReviewRecord(
            review_id="abc123",
            discovery_id="disc123",
            proposal_index=0,
            proposal_name="Quantum Biology",
            proposal=ClassProposal(
                name="Quantum Biology",
                description="Quantum effects in biological systems",
                confidence=0.91,
                sample_count=2,
                example_samples=["quantum biology pathway"],
                justification="coherent concept",
            ),
        )
    ]

    class_proposals = None
    if with_proposals:
        class_proposals = NovelClassAnalysis(
            proposed_classes=[
                ClassProposal(
                    name="Quantum Biology",
                    description="Quantum effects in biological systems",
                    confidence=0.91,
                    sample_count=2,
                    example_samples=[
                        "quantum biology pathway",
                        "quantum biology proteins",
                    ],
                    justification="Both samples describe the same emerging concept",
                    source_cluster_ids=[0],
                )
            ],
            rejected_as_noise=[],
            analysis_summary="One coherent cluster.",
            cluster_count=1,
            model_used="test-model",
        )

    return NovelClassDiscoveryReport(
        discovery_id="disc123",
        timestamp=datetime(2026, 3, 25, 10, 30, 0),
        matcher_config={"model": "minilm", "threshold": 0.6},
        detection_config={"strategies": ["confidence"], "threshold": 0.5},
        novel_sample_report=NovelSampleReport(
            novel_samples=novel_samples,
            detection_strategies=["confidence"],
            config={},
            signal_counts={"confidence": 2},
        ),
        discovery_clusters=clusters,
        class_proposals=class_proposals,
        review_records=review_records,
        diagnostics={"total_samples": 2},
    )


def test_save_proposals_yaml_format(tmp_path: Path):
    """save_proposals should write YAML file with correct structure."""
    report = _make_discovery_report()
    output_path = tmp_path / "proposals"

    saved_path = persistence.save_proposals(
        report, output_dir=output_path, format="yaml"
    )

    assert Path(saved_path).exists()
    assert Path(saved_path).suffix == ".yaml"

    with open(saved_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["discovery_id"] == "disc123"
    assert len(data["novel_sample_report"]["novel_samples"]) == 2
    assert len(data["discovery_clusters"]) == 1


def test_save_proposals_json_format(tmp_path: Path):
    """save_proposals should write JSON file when format=json."""
    report = _make_discovery_report()
    output_path = tmp_path / "proposals"

    saved_path = persistence.save_proposals(
        report, output_dir=output_path, format="json"
    )

    assert Path(saved_path).exists()
    assert Path(saved_path).suffix == ".json"

    with open(saved_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["discovery_id"] == "disc123"


def test_save_proposals_creates_directory(tmp_path: Path):
    """save_proposals should create output directory if it doesn't exist."""
    report = _make_discovery_report()
    output_path = tmp_path / "nested" / "proposals" / "dir"

    persistence.save_proposals(report, output_dir=output_path)

    assert output_path.exists()


def test_save_proposals_unsupported_format_raises_error(tmp_path: Path):
    """save_proposals should raise ValueError for unsupported format."""
    report = _make_discovery_report()
    output_path = tmp_path / "proposals"

    with pytest.raises(ValueError, match="Unsupported format"):
        persistence.save_proposals(report, output_dir=output_path, format="xml")


def test_save_proposals_without_class_proposals(tmp_path: Path):
    """save_proposals should handle reports without class proposals."""
    report = _make_discovery_report(with_proposals=False)
    output_path = tmp_path / "proposals"

    saved_path = persistence.save_proposals(
        report, output_dir=output_path, format="yaml"
    )

    with open(saved_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["discovery_id"] == "disc123"
    assert data.get("class_proposals") is None


def test_load_proposals_yaml(tmp_path: Path):
    """load_proposals should correctly load YAML files."""
    report = _make_discovery_report()
    output_path = tmp_path / "proposals"
    saved_path = persistence.save_proposals(
        report, output_dir=output_path, format="yaml"
    )

    loaded = persistence.load_proposals(saved_path)

    assert loaded.discovery_id == "disc123"
    assert len(loaded.novel_sample_report.novel_samples) == 2
    assert len(loaded.discovery_clusters) == 1


def test_load_proposals_json(tmp_path: Path):
    """load_proposals should correctly load JSON files."""
    report = _make_discovery_report()
    output_path = tmp_path / "proposals"
    saved_path = persistence.save_proposals(
        report, output_dir=output_path, format="json"
    )

    loaded = persistence.load_proposals(saved_path)

    assert loaded.discovery_id == "disc123"
    assert loaded.class_proposals is not None
    assert len(loaded.class_proposals.proposed_classes) == 1


def test_load_proposals_nonexistent_raises_file_not_found(tmp_path: Path):
    """load_proposals should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        persistence.load_proposals(tmp_path / "nonexistent.yaml")


def test_load_proposals_unsupported_format_raises_error(tmp_path: Path):
    """load_proposals should raise ValueError for unsupported file formats."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        persistence.load_proposals(file_path)


def test_list_proposals_empty_directory(tmp_path: Path):
    """list_proposals should return empty list for empty directory."""
    proposals = persistence.list_proposals(tmp_path / "nonexistent")

    assert proposals == []


def test_list_proposals_discovers_yaml_and_json(tmp_path: Path):
    """list_proposals should find both YAML and JSON discovery files."""
    report = _make_discovery_report()
    output_dir = tmp_path / "proposals"
    output_dir.mkdir(parents=True)

    persistence.save_proposals(report, output_dir=output_dir, format="yaml")
    persistence.save_proposals(report, output_dir=output_dir, format="json")

    proposals = persistence.list_proposals(output_dir)

    assert len(proposals) == 2
    formats = {p["format"] for p in proposals}
    assert formats == {"yaml", "json"}


def test_list_proposals_sort_newest(tmp_path: Path):
    """list_proposals should sort by newest first by default."""
    output_dir = tmp_path / "proposals"
    output_dir.mkdir(parents=True)

    report1 = _make_discovery_report()
    report1.timestamp = datetime(2026, 1, 1)
    report1.discovery_id = "old"
    persistence.save_proposals(report1, output_dir=output_dir, format="yaml")

    report2 = _make_discovery_report()
    report2.timestamp = datetime(2026, 3, 1)
    report2.discovery_id = "new"
    persistence.save_proposals(report2, output_dir=output_dir, format="yaml")

    proposals = persistence.list_proposals(output_dir, sort="newest")

    assert proposals[0]["filename"].startswith("discovery_20260301")


def test_list_proposals_sort_oldest(tmp_path: Path):
    """list_proposals should sort by oldest first when specified."""
    output_dir = tmp_path / "proposals"
    output_dir.mkdir(parents=True)

    report1 = _make_discovery_report()
    report1.timestamp = datetime(2026, 1, 1)
    persistence.save_proposals(report1, output_dir=output_dir, format="yaml")

    report2 = _make_discovery_report()
    report2.timestamp = datetime(2026, 3, 1)
    persistence.save_proposals(report2, output_dir=output_dir, format="yaml")

    proposals = persistence.list_proposals(output_dir, sort="oldest")

    assert proposals[0]["filename"].startswith("discovery_20260101")


def test_list_proposals_sort_by_name(tmp_path: Path):
    """list_proposals should sort alphabetically by filename when specified."""
    output_dir = tmp_path / "proposals"
    output_dir.mkdir(parents=True)

    report1 = _make_discovery_report()
    report1.discovery_id = "aaa"
    persistence.save_proposals(report1, output_dir=output_dir, format="yaml")

    report2 = _make_discovery_report()
    report2.discovery_id = "zzz"
    persistence.save_proposals(report2, output_dir=output_dir, format="yaml")

    proposals = persistence.list_proposals(output_dir, sort="name")

    assert len(proposals) == 2
    assert proposals[0]["filename"] <= proposals[1]["filename"]


def test_export_summary_markdown_format(tmp_path: Path):
    """export_summary should write markdown file with correct structure."""
    report = _make_discovery_report()
    output_path = tmp_path / "summary.md"

    persistence.export_summary(report, output_path, format="markdown")

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "# Novel Class Discovery Report" in content
    assert "**Discovery ID:** disc123" in content
    assert "**Novel Samples Detected:** 2" in content
    assert "**Discovery Clusters:** 1" in content


def test_export_summary_with_proposals(tmp_path: Path):
    """export_summary should include proposal details when available."""
    report = _make_discovery_report(with_proposals=True)
    output_path = tmp_path / "summary.md"

    persistence.export_summary(report, output_path, format="markdown")

    content = output_path.read_text(encoding="utf-8")
    assert "## Proposed Classes" in content
    assert "Quantum Biology" in content
    assert "**Confidence:** 91.00%" in content


def test_export_summary_review_lifecycle(tmp_path: Path):
    """export_summary should include review lifecycle stats when records exist."""
    report = _make_discovery_report()
    output_path = tmp_path / "summary.md"

    persistence.export_summary(report, output_path, format="markdown")

    content = output_path.read_text(encoding="utf-8")
    assert "## Review Lifecycle" in content
    assert "**Pending Review:** 1" in content


def test_export_summary_creates_parent_directory(tmp_path: Path):
    """export_summary should create parent directories if needed."""
    report = _make_discovery_report()
    output_path = tmp_path / "nested" / "dir" / "summary.md"

    persistence.export_summary(report, output_path, format="markdown")

    assert output_path.exists()


def test_report_to_dict_converts_numpy_types():
    """_report_to_dict should convert numpy types to Python native types."""
    from novelentitymatcher.novelty.storage import persistence

    report = _make_discovery_report()
    data = persistence._report_to_dict(report)

    assert isinstance(data["matcher_config"]["threshold"], float)
    assert isinstance(data["discovery_clusters"][0]["sample_count"], int)


def test_report_to_dict_handles_none_class_proposals():
    """_report_to_dict should handle None class_proposals gracefully."""
    report = _make_discovery_report(with_proposals=False)
    data = persistence._report_to_dict(report)

    assert data.get("class_proposals") is None
