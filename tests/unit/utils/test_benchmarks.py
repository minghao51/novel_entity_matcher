import json
from types import SimpleNamespace

import pandas as pd

from novelentitymatcher.benchmarks.runner import BenchmarkRunner
from novelentitymatcher.utils.benchmark_dataset import build_processed_ood_sections
from novelentitymatcher.utils import benchmarks


def test_load_processed_sections_reads_csv_sections(tmp_path):
    section_dir = tmp_path / "languages"
    section_dir.mkdir()
    csv_path = section_dir / "languages.csv"
    csv_path.write_text(
        "id,name,aliases,type\n"
        "en,English,en|anglais,language\n"
        "fr,French,fr|francais,language\n",
        encoding="utf-8",
    )

    sections = benchmarks.load_processed_sections(
        processed_dir=tmp_path,
        max_entities_per_section=10,
        max_queries_per_section=5,
    )

    assert len(sections) == 1
    assert sections[0]["section"] == "languages/languages"
    assert sections[0]["entities"][0]["aliases"] == ["anglais"]
    assert sections[0]["queries"]
    assert sections[0]["training_data"]
    assert sections[0]["base_pairs"]
    assert sections[0]["train_pairs"]
    assert sections[0]["base_pairs"][0]["query"] not in {
        pair["query"] for pair in sections[0]["train_pairs"]
    }
    train_texts = {item["text"] for item in sections[0]["training_data"]}
    eval_texts = {
        item["query"] for item in sections[0]["val_pairs"] + sections[0]["test_pairs"]
    }
    assert train_texts.isdisjoint(eval_texts)


def test_load_processed_sections_skips_eval_for_single_text_entities(tmp_path):
    section_dir = tmp_path / "products"
    section_dir.mkdir()
    csv_path = section_dir / "products.csv"
    csv_path.write_text(
        "id,name,aliases,type\nsku-1,Widget,,product\n",
        encoding="utf-8",
    )

    sections = benchmarks.load_processed_sections(
        processed_dir=tmp_path,
        max_entities_per_section=10,
        max_queries_per_section=5,
    )

    assert len(sections) == 1
    assert sections[0]["training_data"] == [{"text": "Widget", "label": "sku-1"}]
    assert sections[0]["evaluation_pairs"]
    assert sections[0]["val_pairs"]
    assert sections[0]["test_pairs"] == []
    assert sections[0]["evaluation_pairs"][0]["query"] != "Widget"


def test_load_processed_sections_drops_shared_aliases(tmp_path):
    section_dir = tmp_path / "products"
    section_dir.mkdir()
    csv_path = section_dir / "products.csv"
    csv_path.write_text(
        "id,name,aliases,type\n"
        "sku-1,Widget,common|alpha,product\n"
        "sku-2,Gadget,common|beta,product\n",
        encoding="utf-8",
    )

    sections = benchmarks.load_processed_sections(
        processed_dir=tmp_path,
        max_entities_per_section=10,
        max_queries_per_section=5,
    )

    assert len(sections) == 1
    assert sections[0]["entities"][0]["aliases"] == ["alpha"]
    assert sections[0]["entities"][1]["aliases"] == ["beta"]
    assert all("common" not in pair["query"] for pair in sections[0]["base_pairs"])


def test_benchmark_embedding_models_expands_registry(monkeypatch):
    called = []

    class FakeEmbeddingMatcher:
        def __init__(self, entities, model_name, threshold):
            called.append(model_name)
            self.model_name = model_name

        def build_index(self, batch_size=None):
            return None

        def match(self, query, batch_size=None):
            if isinstance(query, list):
                return [{"id": "ID-0", "score": 0.9} for _ in query]
            return {"id": "ID-0", "score": 0.9}

    monkeypatch.setattr(benchmarks, "EmbeddingMatcher", FakeEmbeddingMatcher)

    results = benchmarks.benchmark_embedding_models(
        model_names=["potion-8m", "mpnet"],
        iterations=1,
        sections_data=[
            {
                "section": "custom/test",
                "entities": [
                    {"id": "ID-0", "name": "Alpha"},
                    {"id": "ID-1", "name": "Beta"},
                ],
                "queries": ["Alpha", "Beta"],
                "accuracy_pairs": [
                    {"query": "Alpha", "expected_id": "ID-0"},
                    {"query": "Beta", "expected_id": "ID-1"},
                ],
            }
        ],
    )

    assert called == ["potion-8m", "mpnet"]
    assert set(results["model"]) == {"potion-8m", "mpnet"}
    assert set(results["section"]) == {"custom/test"}


def test_benchmark_embedding_models_records_skips(monkeypatch):
    class FailingEmbeddingMatcher:
        def __init__(self, entities, model_name, threshold):
            self.model_name = model_name

        def build_index(self, batch_size=None):
            raise RuntimeError("missing model")

    monkeypatch.setattr(benchmarks, "EmbeddingMatcher", FailingEmbeddingMatcher)

    results = benchmarks.benchmark_embedding_models(
        model_names=["potion-8m"],
        iterations=1,
        sections_data=[
            {
                "section": "custom/test",
                "entities": [{"id": "ID-0", "name": "Alpha"}],
                "queries": ["Alpha"],
                "accuracy_pairs": [{"query": "Alpha", "expected_id": "ID-0"}],
            }
        ],
    )

    row = results.iloc[0]
    assert row["status"] == "skipped"
    assert "missing model" in row["skip_reason"]


def test_save_benchmark_report_writes_json(tmp_path):
    results = pd.DataFrame(
        [
            {
                "track": "embedding",
                "model": "potion-8m",
                "status": "ok",
                "throughput_qps": 10.0,
            }
        ]
    )

    output_path = tmp_path / "benchmarks.json"
    benchmarks.save_benchmark_report(results, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["model"] == "potion-8m"


def test_format_benchmark_summary_includes_skip_reason():
    results = pd.DataFrame(
        [
            {
                "track": "embedding",
                "section": "languages/languages",
                "model": "potion-8m",
                "backend": "static",
                "status": "skipped",
                "throughput_qps": None,
                "accuracy": None,
                "speedup_vs_minilm": None,
                "skip_reason": "unavailable",
            }
        ]
    )

    summary = benchmarks.format_benchmark_summary(results)

    assert "BENCHMARK RESULTS" in summary
    assert "languages/languages" in summary
    assert "unavailable" in summary


def test_build_processed_ood_sections_creates_known_and_novel_pairs(tmp_path):
    section_dir = tmp_path / "languages"
    section_dir.mkdir()
    csv_path = section_dir / "languages.csv"
    csv_path.write_text(
        "id,name,aliases,type\n"
        "en,English,anglais|eng,language\n"
        "fr,French,francais|fra,language\n"
        "de,German,allemand|deu,language\n"
        "es,Spanish,espanol|spa,language\n"
        "it,Italian,italiano|ita,language\n",
        encoding="utf-8",
    )

    sections = build_processed_ood_sections(
        processed_dir=tmp_path,
        max_entities_per_section=10,
        max_queries_per_section=5,
        ood_ratio=0.2,
    )

    assert len(sections) == 1
    section = sections[0]
    assert section["track"] == "ood_novelty"
    assert len(section["known_entities"]) == 4
    assert len(section["heldout_class_ids"]) == 1
    assert section["training_data"]
    assert section["val_known_pairs"]
    assert section["val_novel_pairs"]
    assert section["test_known_pairs"]
    assert section["test_novel_pairs"]
    assert all(not pair["is_novel"] for pair in section["test_known_pairs"])
    assert all(pair["is_novel"] for pair in section["test_novel_pairs"])


def test_run_novelty_on_processed_uses_detector_and_writes_artifact(
    tmp_path,
    monkeypatch,
):
    captured = {}

    monkeypatch.setattr(
        "novelentitymatcher.utils.benchmark_dataset.build_processed_ood_sections",
        lambda **kwargs: [
            {
                "section": "custom/test",
                "known_entities": [
                    {"id": "known-a", "name": "Known A"},
                    {"id": "known-b", "name": "Known B"},
                    {"id": "known-c", "name": "Known C"},
                ],
                "training_data": [
                    {"text": "Known A", "label": "known-a"},
                    {"text": "Known B", "label": "known-b"},
                    {"text": "Known C", "label": "known-c"},
                ],
                "known_class_ids": ["known-a", "known-b", "known-c"],
                "heldout_class_ids": ["novel-z"],
                "val_known_pairs": [
                    {
                        "query": "Known A",
                        "expected_id": "known-a",
                        "label": "known-a",
                        "is_novel": False,
                        "split": "val_known",
                    }
                ],
                "val_novel_pairs": [
                    {
                        "query": "Novel Z",
                        "expected_id": "novel-z",
                        "label": "novel-z",
                        "is_novel": True,
                        "split": "val_novel",
                    }
                ],
                "test_known_pairs": [
                    {
                        "query": "Known B",
                        "expected_id": "known-b",
                        "label": "known-b",
                        "is_novel": False,
                        "split": "test_known",
                    }
                ],
                "test_novel_pairs": [
                    {
                        "query": "Novel Z",
                        "expected_id": "novel-z",
                        "label": "novel-z",
                        "is_novel": True,
                        "split": "test_novel",
                    }
                ],
            }
        ],
    )

    class FakeNovelEntityMatcher:
        def __init__(self, **kwargs):
            captured["use_novelty_detector"] = kwargs["use_novelty_detector"]

        def fit(self, training_data=None, mode=None, show_progress=True, **kwargs):
            captured["training_data"] = training_data
            captured["mode"] = mode
            return self

        def match(self, query, existing_classes=None):
            if query == "Novel Z":
                return SimpleNamespace(
                    predicted_id="known-a",
                    id=None,
                    is_novel=True,
                    score=0.2,
                    novel_score=0.8,
                    match_method="novelty_detector",
                    signals={"confidence": True},
                )
            return SimpleNamespace(
                predicted_id="known-b",
                id="known-b",
                is_novel=False,
                score=0.9,
                novel_score=0.1,
                match_method="accepted_known",
                signals={},
            )

    monkeypatch.setattr(
        "novelentitymatcher.benchmarks.runner.NovelEntityMatcher",
        FakeNovelEntityMatcher,
    )

    runner = BenchmarkRunner(output_dir=tmp_path / "bench")
    results = runner.run_novelty_on_processed(
        datasets=["custom/test"],
        model="potion-8m",
        confidence_thresholds=[0.3],
        ood_ratio=0.2,
    )

    assert captured["use_novelty_detector"] is True
    assert captured["mode"] == "head-only"
    row = results.iloc[0]
    assert row["track"] == "ood_novelty"
    assert row["selected_threshold"] == 0.3
    assert row["num_threshold_candidates"] == 1
    assert row["novel_precision"] == 1.0
    assert row["novel_recall"] == 1.0
    artifact_path = (
        tmp_path
        / "bench"
        / "artifacts"
        / "processed_ood_novelty"
        / "custom__test_thr_0_3.json"
    )
    assert artifact_path.exists()


def test_format_benchmark_summary_handles_ood_novelty_track():
    results = pd.DataFrame(
        [
            {
                "track": "ood_novelty",
                "section": "languages/languages",
                "model": "potion-8m",
                "selected_threshold": 0.3,
                "num_threshold_candidates": 3,
                "num_known_classes": 4,
                "num_heldout_classes": 1,
                "validation_novel_f1": 0.72,
                "validation_known_accuracy": 0.88,
                "validation_false_positive_novel_rate": 0.12,
                "novel_precision": 0.75,
                "novel_recall": 0.60,
                "novel_f1": 0.67,
                "known_accuracy": 0.90,
                "false_positive_novel_rate": 0.10,
                "overall_accuracy": 0.80,
                "artifact_path": "/tmp/artifact.json",
            }
        ]
    )

    summary = benchmarks.format_benchmark_summary(results)

    assert "ood_novelty" in summary
    assert "novel_precision" in summary
    assert "/tmp/artifact.json" in summary


def test_run_novelty_on_processed_calibrates_on_validation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "novelentitymatcher.utils.benchmark_dataset.build_processed_ood_sections",
        lambda **kwargs: [
            {
                "section": "custom/calibrated",
                "known_entities": [
                    {"id": "known-a", "name": "Known A"},
                    {"id": "known-b", "name": "Known B"},
                    {"id": "known-c", "name": "Known C"},
                ],
                "training_data": [
                    {"text": "Known A", "label": "known-a"},
                    {"text": "Known B", "label": "known-b"},
                    {"text": "Known C", "label": "known-c"},
                ],
                "known_class_ids": ["known-a", "known-b", "known-c"],
                "heldout_class_ids": ["novel-z"],
                "val_known_pairs": [
                    {
                        "query": "VAL_KNOWN",
                        "expected_id": "known-a",
                        "label": "known-a",
                        "is_novel": False,
                        "split": "val_known",
                    }
                ],
                "val_novel_pairs": [
                    {
                        "query": "VAL_NOVEL",
                        "expected_id": "novel-z",
                        "label": "novel-z",
                        "is_novel": True,
                        "split": "val_novel",
                    }
                ],
                "test_known_pairs": [
                    {
                        "query": "TEST_KNOWN",
                        "expected_id": "known-a",
                        "label": "known-a",
                        "is_novel": False,
                        "split": "test_known",
                    }
                ],
                "test_novel_pairs": [
                    {
                        "query": "TEST_NOVEL",
                        "expected_id": "novel-z",
                        "label": "novel-z",
                        "is_novel": True,
                        "split": "test_novel",
                    }
                ],
            }
        ],
    )

    class FakeNovelEntityMatcher:
        def __init__(self, **kwargs):
            self.threshold = kwargs["confidence_threshold"]

        def fit(self, training_data=None, mode=None, show_progress=True, **kwargs):
            return self

        def match(self, query, existing_classes=None):
            if self.threshold == 0.2:
                predictions = {
                    "VAL_KNOWN": (True, None, "known-a"),
                    "VAL_NOVEL": (True, None, "known-a"),
                    "TEST_KNOWN": (True, None, "known-a"),
                    "TEST_NOVEL": (True, None, "known-a"),
                }
            else:
                predictions = {
                    "VAL_KNOWN": (False, "known-a", "known-a"),
                    "VAL_NOVEL": (True, None, "known-a"),
                    "TEST_KNOWN": (False, "known-a", "known-a"),
                    "TEST_NOVEL": (True, None, "known-a"),
                }
            is_novel, matched_id, predicted_id = predictions[query]
            return SimpleNamespace(
                predicted_id=predicted_id,
                id=matched_id,
                is_novel=is_novel,
                score=0.9 if not is_novel else 0.2,
                novel_score=0.8 if is_novel else 0.1,
                match_method="novelty_detector" if is_novel else "accepted_known",
                signals={"confidence": is_novel},
            )

    monkeypatch.setattr(
        "novelentitymatcher.benchmarks.runner.NovelEntityMatcher",
        FakeNovelEntityMatcher,
    )

    runner = BenchmarkRunner(output_dir=tmp_path / "bench")
    results = runner.run_novelty_on_processed(
        datasets=["custom/calibrated"],
        model="potion-8m",
        confidence_thresholds=[0.2, 0.4],
        ood_ratio=0.2,
        calibrate_thresholds=True,
    )

    row = results.iloc[0]
    assert row["selected_threshold"] == 0.4
    assert row["validation_novel_f1"] == 1.0
    assert row["validation_known_accuracy"] == 1.0
