"""Tests for Phase 2 features: SetFit centroid, meta-learner, adaptive weights."""

import json
import tempfile
from pathlib import Path

import numpy as np

from novelentitymatcher.novelty.config.strategies import SetFitCentroidConfig
from novelentitymatcher.novelty.core.adaptive_weights import (
    DatasetCharacteristics,
    adaptive_weights,
    compute_characteristics,
)
from novelentitymatcher.novelty.core.signal_combiner import SignalCombiner
from novelentitymatcher.novelty.strategies.setfit_centroid import SetFitCentroidStrategy


class TestSetFitCentroidStrategy:
    """Tests for the SetFit centroid distance strategy."""

    def _make_embeddings(self, n, dim=384, seed=42):
        rng = np.random.RandomState(seed)
        emb = rng.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms

    def test_initialization_computes_centroids(self):
        strategy = SetFitCentroidStrategy()
        embeddings = self._make_embeddings(20, dim=64)
        labels = ["A"] * 10 + ["B"] * 10

        strategy.initialize(embeddings, labels, SetFitCentroidConfig())

        assert strategy._centroids is not None
        assert strategy._centroids.shape == (2, 64)
        assert set(strategy._class_labels) == {"A", "B"}
        assert strategy._threshold is not None

    def test_detect_known_samples_not_flagged(self):
        strategy = SetFitCentroidStrategy()
        rng = np.random.RandomState(42)

        centroid_a = rng.randn(64).astype(np.float32)
        centroid_a /= np.linalg.norm(centroid_a)
        centroid_b = rng.randn(64).astype(np.float32)
        centroid_b /= np.linalg.norm(centroid_b)

        known_embeddings = np.vstack(
            [
                centroid_a + rng.randn(20, 64).astype(np.float32) * 0.05,
                centroid_b + rng.randn(20, 64).astype(np.float32) * 0.05,
            ]
        )
        known_embeddings = known_embeddings / np.linalg.norm(
            known_embeddings, axis=1, keepdims=True
        )
        labels = ["A"] * 20 + ["B"] * 20

        strategy.initialize(known_embeddings, labels, SetFitCentroidConfig())

        query_embeddings = np.vstack(
            [
                centroid_a + rng.randn(2, 64).astype(np.float32) * 0.02,
                centroid_b + rng.randn(2, 64).astype(np.float32) * 0.02,
            ]
        )
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )

        flags, metrics = strategy.detect(
            texts=[""] * 4,
            embeddings=query_embeddings,
            predicted_classes=["A", "A", "B", "B"],
            confidences=np.array([0.9, 0.85, 0.9, 0.88]),
        )

        assert len(flags) == 0
        for idx in range(4):
            assert "setfit_centroid_min_distance" in metrics[idx]
            assert "setfit_centroid_novelty_score" in metrics[idx]
            assert "setfit_centroid_nearest_class" in metrics[idx]

    def test_detect_novel_samples_flagged(self):
        strategy = SetFitCentroidStrategy()
        rng = np.random.RandomState(42)

        centroid_a = rng.randn(64).astype(np.float32)
        centroid_a /= np.linalg.norm(centroid_a)
        centroid_b = rng.randn(64).astype(np.float32)
        centroid_b /= np.linalg.norm(centroid_b)

        known_embeddings = np.vstack(
            [
                centroid_a + rng.randn(10, 64).astype(np.float32) * 0.01,
                centroid_b + rng.randn(10, 64).astype(np.float32) * 0.01,
            ]
        )
        known_embeddings = known_embeddings / np.linalg.norm(
            known_embeddings, axis=1, keepdims=True
        )
        labels = ["A"] * 10 + ["B"] * 10

        strategy.initialize(known_embeddings, labels, SetFitCentroidConfig())

        novel_embedding = rng.randn(1, 64).astype(np.float32)
        novel_embedding = novel_embedding / np.linalg.norm(novel_embedding)

        flags, metrics = strategy.detect(
            texts=["novel sample"],
            embeddings=novel_embedding,
            predicted_classes=["A"],
            confidences=np.array([0.5]),
        )

        assert len(flags) == 1
        assert 0 in flags
        assert metrics[0]["setfit_centroid_is_novel"] is True

    def test_continuous_scores(self):
        strategy = SetFitCentroidStrategy()
        embeddings = self._make_embeddings(20, dim=64)
        labels = ["A"] * 10 + ["B"] * 10

        strategy.initialize(embeddings, labels, SetFitCentroidConfig())

        query = self._make_embeddings(1, dim=64, seed=99)
        _flags, metrics = strategy.detect(
            texts=[""],
            embeddings=query,
            predicted_classes=["A"],
            confidences=np.array([0.7]),
        )

        score = metrics[0]["setfit_centroid_novelty_score"]
        assert 0.0 <= score <= 1.0

    def test_get_weight(self):
        strategy = SetFitCentroidStrategy()
        assert strategy.get_weight() == 0.45

    def test_config_schema(self):
        strategy = SetFitCentroidStrategy()
        assert strategy.config_schema == SetFitCentroidConfig


class TestMetaLearnerSignalCombiner:
    """Tests for the meta-learner signal combination."""

    def _make_config(self, method="weighted"):
        from novelentitymatcher.novelty.config.base import DetectionConfig

        return DetectionConfig(
            strategies=["confidence", "knn_distance"],
            combine_method=method,
        )

    def test_weighted_combination_with_setfit_centroid(self):
        config = self._make_config("weighted")
        combiner = SignalCombiner(config)

        strategy_outputs = {
            "confidence": ({0, 1}, {}),
            "knn_distance": ({0}, {}),
            "setfit_centroid": ({0, 2}, {}),
        }
        all_metrics = {
            0: {
                "confidence_is_novel": True,
                "knn_novelty_score": 0.6,
                "setfit_centroid_novelty_score": 0.7,
            },
            1: {
                "confidence_is_novel": True,
                "knn_novelty_score": 0.2,
                "setfit_centroid_novelty_score": 0.3,
            },
            2: {
                "confidence_is_novel": False,
                "knn_novelty_score": 0.1,
                "setfit_centroid_novelty_score": 0.8,
            },
        }

        novel_indices, scores = combiner.combine(strategy_outputs, all_metrics)

        assert 0 in novel_indices or 2 in novel_indices
        assert 0 in scores
        assert 2 in scores

    def test_meta_learner_fallback_when_not_trained(self):
        config = self._make_config("meta_learner")
        combiner = SignalCombiner(config)

        strategy_outputs = {"confidence": ({0}, {})}
        all_metrics = {0: {"confidence_is_novel": True}}

        _novel_indices, scores = combiner.combine(strategy_outputs, all_metrics)

        assert 0 in scores

    def test_meta_learner_train_and_predict(self):
        config = self._make_config("meta_learner")
        combiner = SignalCombiner(config)

        features = np.array(
            [
                [1.0, 0.8, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.5, 0.3],
                [0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1],
                [1.0, 0.9, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.6, 0.4],
                [0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.05],
            ]
        )
        labels = np.array([1, 0, 1, 0])

        accuracy = combiner.train_meta_learner(features, labels)
        assert accuracy >= 0.5

        strategy_outputs = {"confidence": ({0, 1}, {}), "setfit_centroid": ({0}, {})}
        all_metrics = {
            0: {
                "confidence_is_novel": True,
                "uncertainty_score": 0.8,
                "knn_novelty_score": 0.7,
                "cluster_support_score": 0.2,
                "self_knowledge_is_novel": False,
                "pattern_is_novel": False,
                "oneclass_is_novel": False,
                "prototypical_is_novel": False,
                "setfit_is_novel": False,
                "setfit_centroid_novelty_score": 0.6,
                "mahalanobis_novelty_score": 0.5,
                "lof_novelty_score": 0.3,
            },
            1: {
                "confidence_is_novel": True,
                "uncertainty_score": 0.1,
                "knn_novelty_score": 0.1,
                "cluster_support_score": 0.0,
                "self_knowledge_is_novel": False,
                "pattern_is_novel": False,
                "oneclass_is_novel": False,
                "prototypical_is_novel": False,
                "setfit_is_novel": False,
                "setfit_centroid_novelty_score": 0.05,
                "mahalanobis_novelty_score": 0.2,
                "lof_novelty_score": 0.1,
            },
        }

        _novel_indices, scores = combiner.combine(strategy_outputs, all_metrics)
        assert 0 in scores
        assert 1 in scores

    def test_meta_learner_save_and_load(self):
        config = self._make_config("meta_learner")
        combiner = SignalCombiner(config)

        features = np.array(
            [
                [1.0, 0.8, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.5, 0.3],
                [0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.1],
            ]
        )
        labels = np.array([1, 0])
        combiner.train_meta_learner(features, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            combiner.save_meta_learner(tmpdir)

            assert (Path(tmpdir) / "meta_learner.pkl").exists()
            assert (Path(tmpdir) / "meta_learner_meta.json").exists()

            with open(Path(tmpdir) / "meta_learner_meta.json") as f:
                meta = json.load(f)
            assert "feature_names" in meta

            combiner2 = SignalCombiner(config)
            combiner2.load_meta_learner(tmpdir)
            assert combiner2._meta_model is not None


class TestAdaptiveWeights:
    """Tests for adaptive strategy weight computation."""

    def test_compute_characteristics(self):
        rng = np.random.RandomState(42)
        centroids = rng.randn(3, 64).astype(np.float32)
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        embeddings = np.vstack(
            [
                centroids[0] + rng.randn(10, 64).astype(np.float32) * 0.1,
                centroids[1] + rng.randn(10, 64).astype(np.float32) * 0.1,
                centroids[2] + rng.randn(10, 64).astype(np.float32) * 0.1,
            ]
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 10

        chars = compute_characteristics(embeddings, labels)

        assert chars.n_samples == 30
        assert chars.n_classes == 3
        assert chars.class_separability > 0
        assert 0.0 < chars.effective_dimensionality <= 1.0
        assert chars.mean_knn_distance >= 0

    def test_adaptive_weights_high_separability(self):
        chars = DatasetCharacteristics(
            n_samples=100,
            n_classes=5,
            samples_per_class={"A": 20, "B": 20, "C": 20, "D": 20, "E": 20},
            class_separability=5.0,
            mean_intra_class_distance=0.1,
            mean_inter_class_distance=0.5,
            effective_dimensionality=0.6,
            class_balance_entropy=1.0,
            mean_knn_distance=0.2,
        )

        weights = adaptive_weights(chars)
        assert weights.setfit_centroid > 0.45

    def test_adaptive_weights_low_samples(self):
        chars = DatasetCharacteristics(
            n_samples=10,
            n_classes=5,
            samples_per_class={"A": 2, "B": 2, "C": 2, "D": 2, "E": 2},
            class_separability=1.0,
            mean_intra_class_distance=0.3,
            mean_inter_class_distance=0.3,
            effective_dimensionality=0.8,
            class_balance_entropy=1.0,
            mean_knn_distance=0.5,
        )

        weights = adaptive_weights(chars)
        assert weights.oneclass < 0.1

    def test_adaptive_weights_normalization(self):
        chars = DatasetCharacteristics(
            n_samples=50,
            n_classes=3,
            samples_per_class={"A": 20, "B": 15, "C": 15},
            class_separability=2.0,
            mean_intra_class_distance=0.2,
            mean_inter_class_distance=0.4,
            effective_dimensionality=0.5,
            class_balance_entropy=0.9,
            mean_knn_distance=0.3,
        )

        weights = adaptive_weights(chars)
        normalized = weights.normalize_weights()

        total = (
            normalized.confidence
            + normalized.uncertainty
            + normalized.knn
            + normalized.cluster
            + normalized.self_knowledge
            + normalized.pattern
            + normalized.oneclass
            + normalized.prototypical
            + normalized.setfit
            + normalized.setfit_centroid
            + normalized.mahalanobis
            + normalized.lof
        )
        assert abs(total - 1.0) < 0.01
