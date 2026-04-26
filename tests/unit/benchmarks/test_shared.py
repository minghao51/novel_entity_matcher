import numpy as np

from novelentitymatcher.benchmarks.shared import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OOD_RATIO,
    DEFAULT_RANDOM_SEED,
    SplitData,
    compute_ood_metrics,
    generate_synthetic_data,
    prepare_binary_labels,
    timer,
)


class TestSplitData:
    def test_defaults(self):
        sd = SplitData(
            train_texts=["a"],
            train_labels=["x"],
            val_texts=["b"],
            val_labels=["x"],
            test_texts=["c"],
            test_labels=["x"],
        )
        assert sd.entities == []
        assert sd.known_classes == []
        assert sd.ood_classes == []

    def test_custom_fields(self):
        sd = SplitData(
            train_texts=["a"],
            train_labels=["x"],
            val_texts=[],
            val_labels=[],
            test_texts=[],
            test_labels=[],
            entities=[{"id": "1"}],
            known_classes=["x"],
            ood_classes=["y"],
        )
        assert len(sd.entities) == 1
        assert sd.known_classes == ["x"]
        assert sd.ood_classes == ["y"]


class TestTimer:
    def test_records_elapsed(self):
        import time

        with timer() as t:
            time.sleep(0.05)
        assert t["elapsed"] >= 0.04

    def test_zero_elapsed(self):
        with timer() as t:
            pass
        assert t["elapsed"] >= 0.0


class TestComputeOodMetrics:
    def test_perfect_separation(self):
        labels = ["known", "known", "__NOVEL__", "__NOVEL__"]
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        result = compute_ood_metrics(labels, scores)
        assert result["auroc"] == 1.0
        assert result["auprc"] == 1.0

    def test_random_scores(self):
        labels = ["known"] * 50 + ["__NOVEL__"] * 50
        rng = np.random.default_rng(42)
        scores = rng.random(100)
        result = compute_ood_metrics(labels, scores)
        assert 0.3 < result["auroc"] < 0.7

    def test_single_class_returns_half(self):
        labels = ["known"] * 10
        scores = np.random.random(10)
        result = compute_ood_metrics(labels, scores)
        assert result["auroc"] == 0.5
        assert result["auprc"] == 0.5

    def test_numpy_binary_input(self):
        true_binary = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        result = compute_ood_metrics(true_binary, scores)
        assert result["auroc"] == 1.0

    def test_custom_novel_marker(self):
        labels = ["ok", "ok", "ANOMALY", "ANOMALY"]
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        result = compute_ood_metrics(labels, scores, novel_marker="ANOMALY")
        assert result["auroc"] == 1.0

    def test_detection_rates(self):
        labels = ["known"] * 100 + ["__NOVEL__"] * 10
        scores = np.concatenate(
            [
                np.random.uniform(0, 0.3, 100),
                np.random.uniform(0.7, 1.0, 10),
            ]
        )
        result = compute_ood_metrics(labels, scores)
        assert 0.0 <= result["dr_1fp"] <= 1.0
        assert 0.0 <= result["dr_5fp"] <= 1.0
        assert 0.0 <= result["dr_10fp"] <= 1.0


class TestPrepareBinaryLabels:
    def test_basic(self):
        labels = ["cat", "dog", "__OOD__", "cat"]
        result = prepare_binary_labels(labels)
        np.testing.assert_array_equal(result, [0, 0, 1, 0])

    def test_custom_marker(self):
        labels = ["a", "UNKNOWN", "b"]
        result = prepare_binary_labels(labels, novel_marker="UNKNOWN")
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_all_known(self):
        labels = ["a", "b", "c"]
        result = prepare_binary_labels(labels)
        assert result.sum() == 0

    def test_all_novel(self):
        labels = ["__OOD__"] * 5
        result = prepare_binary_labels(labels)
        assert result.sum() == 5


class TestGenerateSyntheticData:
    def test_default_counts(self):
        train, test = generate_synthetic_data()
        assert len(train) == 10 * 50
        assert len(test) == 10 * 5

    def test_custom_counts(self):
        train, test = generate_synthetic_data(num_entities=3, samples_per_entity=20)
        assert len(train) == 3 * 20
        assert len(test) == 3 * 2

    def test_data_structure(self):
        train, test = generate_synthetic_data(num_entities=2, samples_per_entity=10)
        assert train[0]["label"] == "ENTITY_0"
        assert "ENTITY_0" in train[0]["text"]
        assert test[0]["label"] in ["ENTITY_0", "ENTITY_1"]

    def test_labels_match_entities(self):
        train, _ = generate_synthetic_data(num_entities=5, samples_per_entity=4)
        unique_labels = {item["label"] for item in train}
        assert unique_labels == {
            "ENTITY_0",
            "ENTITY_1",
            "ENTITY_2",
            "ENTITY_3",
            "ENTITY_4",
        }


class TestDefaults:
    def test_default_model_name(self):
        assert "MiniLM" in DEFAULT_MODEL_NAME

    def test_default_ood_ratio(self):
        assert DEFAULT_OOD_RATIO == 0.2

    def test_default_seed(self):
        assert DEFAULT_RANDOM_SEED == 42
