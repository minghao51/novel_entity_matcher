"""Integration tests for novel class detection system."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
from datetime import datetime

from novelentitymatcher import Matcher, NovelEntityMatcher
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import ConfidenceConfig, KNNConfig
from novelentitymatcher.novelty.core.detector import NoveltyDetector
from novelentitymatcher.novelty.storage import load_proposals, save_proposals


class TestNovelClassDetectionIntegration:
    """Integration tests for the complete novel class detection pipeline."""

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            {"id": "physics", "name": "Quantum Physics"},
            {"id": "cs", "name": "Computer Science"},
            {"id": "biology", "name": "Molecular Biology"},
            {"id": "chemistry", "name": "Organic Chemistry"},
        ]

    @pytest.fixture
    def training_data(self):
        """Create training data for matcher."""
        return {
            "physics": [
                "quantum mechanics",
                "wave function",
                "Schrödinger equation",
                "quantum entanglement",
            ],
            "cs": [
                "machine learning",
                "neural networks",
                "algorithm design",
                "data structures",
            ],
            "biology": [
                "gene expression",
                "protein synthesis",
                "cell division",
                "DNA replication",
            ],
            "chemistry": [
                "chemical bonding",
                "molecular structure",
                "reaction kinetics",
                "organic synthesis",
            ],
        }

    @pytest.fixture
    def test_queries(self):
        """Create test queries with known and novel classes."""
        return [
            # Known classes
            "quantum superposition",
            "deep learning models",
            "CRISPR gene editing",
            "covalent bonding",
            # Novel/ambiguous classes
            "quantum biology applications",
            "bioinformatics algorithms",
            "computational chemistry methods",
            # Outliers/noise
            "random unrelated text here",
            "gibberish content test",
        ]

    @pytest.fixture
    def trained_matcher(self, sample_entities, training_data):
        """Create a trained matcher."""
        # Flatten training data
        texts = []
        labels = []
        for label, samples in training_data.items():
            texts.extend(samples)
            labels.extend([label] * len(samples))

        # Create and train matcher
        matcher = Matcher(
            entities=sample_entities,
            model="minilm",  # Use smaller model for faster testing
            threshold=0.6,
        )

        # Train the matcher
        matcher.fit(texts=texts, labels=labels)

        return matcher

    @pytest.mark.asyncio
    async def test_end_to_end_discovery(self, trained_matcher, test_queries):
        """Test end-to-end novel class discovery."""
        # Create detector
        detector = NovelEntityMatcher(
            matcher=trained_matcher,
            detection_config=DetectionConfig(
                strategies=["confidence", "knn_distance"],
                confidence=ConfidenceConfig(threshold=0.7),
                knn_distance=KNNConfig(distance_threshold=0.4),
            ),
            llm_provider=None,  # Skip LLM for this test
            auto_save=False,
        )

        # Run discovery
        report = await detector.discover_novel_classes(
            queries=test_queries,
            existing_classes=["physics", "cs", "biology", "chemistry"],
            run_llm_proposal=False,  # Skip LLM for faster testing
        )

        # Verify report structure
        assert report.discovery_id is not None
        assert report.novel_sample_report is not None
        assert len(report.novel_sample_report.novel_samples) >= 0

        novel_indices = [s.index for s in report.novel_sample_report.novel_samples]
        assert all(isinstance(idx, int) for idx in novel_indices)
        assert all(0 <= idx < len(test_queries) for idx in novel_indices)

    @pytest.mark.asyncio
    async def test_metadata_return(self, trained_matcher):
        """Test that match() returns metadata when requested."""
        # Test with return_metadata=True
        result = trained_matcher.match(
            ["test query", "another query"],
            return_metadata=True,
        )

        # Verify metadata structure
        assert hasattr(result, "predictions")
        assert hasattr(result, "confidences")
        assert hasattr(result, "embeddings")
        assert hasattr(result, "metadata")

        assert len(result.predictions) == 2
        assert len(result.confidences) == 2
        assert result.embeddings.shape[0] == 2

    @pytest.mark.asyncio
    async def test_metadata_return_single_query(self, trained_matcher):
        """Test metadata return with single query."""
        result = trained_matcher.match("test query", return_metadata=True)

        assert hasattr(result, "predictions")
        assert len(result.predictions) == 1
        assert result.embeddings.shape[0] == 1

    def test_metadata_return_sync(self, trained_matcher):
        """Test synchronous metadata return."""
        result = trained_matcher.match(
            ["test query 1", "test query 2"],
            return_metadata=True,
        )

        assert hasattr(result, "predictions")
        assert hasattr(result, "confidences")
        assert hasattr(result, "embeddings")
        assert hasattr(result, "candidate_results")
        assert hasattr(result, "records")

    def test_metadata_return_exposes_stable_candidate_records(self, trained_matcher):
        """Metadata mode should expose normalized candidates and per-query records."""
        result = trained_matcher.match(
            ["test query 1", "test query 2"],
            return_metadata=True,
            top_k=3,
        )

        assert len(result.candidate_results) == 2
        assert len(result.records) == 2
        assert result.records[0].text == "test query 1"
        assert result.records[1].text == "test query 2"
        assert result.records[0].confidence == pytest.approx(result.confidences[0])
        assert isinstance(result.candidate_results[0], list)
        assert result.metadata["candidate_results"] == result.candidate_results

    def test_metadata_return_preserves_top_prediction_below_threshold(self):
        """Metadata mode should keep the best candidate for novelty detection."""
        matcher = Matcher(
            entities=[
                {"id": "physics", "name": "Quantum Physics"},
                {"id": "cs", "name": "Computer Science"},
            ],
            model="minilm",
            threshold=0.99,
        )
        matcher.fit(
            texts=[
                "quantum mechanics",
                "wave function",
                "machine learning",
                "data structures",
            ],
            labels=["physics", "physics", "cs", "cs"],
        )

        standard = matcher.match(["quantum superposition"])
        metadata = matcher.match(["quantum superposition"], return_metadata=True)

        assert standard == [None]
        assert metadata.predictions[0] != "unknown"
        assert metadata.confidences[0] > 0.0
        assert metadata.metadata["threshold_override"] == 0.0
        assert metadata.metadata["evaluation_threshold"] == 0.99
        assert metadata.records[0].predicted_id == metadata.predictions[0]

    @pytest.mark.asyncio
    async def test_discovery_uses_async_matcher_path(self, trained_matcher):
        """Novel discovery should use the async matcher API when available."""
        detector = NovelEntityMatcher(
            matcher=trained_matcher,
            detection_config=DetectionConfig(
                strategies=["confidence"],
            ),
            auto_save=False,
        )

        original_match = trained_matcher.match
        original_match_async = trained_matcher.match_async
        calls = {"match_async": 0, "match": 0}

        async def tracked_match_async(*args, **kwargs):
            calls["match_async"] += 1
            return await Matcher.match_async(trained_matcher, *args, **kwargs)

        def forbidden_sync_match(*args, **kwargs):
            calls["match"] += 1
            raise AssertionError("discover_novel_classes should not call sync match()")

        trained_matcher.match_async = tracked_match_async
        trained_matcher.match = forbidden_sync_match
        try:
            report = await detector.discover_novel_classes(
                queries=["quantum superposition", "random unrelated text here"],
                existing_classes=["physics", "cs", "biology", "chemistry"],
                run_llm_proposal=False,
            )
        finally:
            trained_matcher.match = original_match
            trained_matcher.match_async = original_match_async

        assert calls["match_async"] >= 1
        assert calls["match"] == 0
        assert report.novel_sample_report is not None
        assert "match" in report.metadata["pipeline_stage_metadata"]
        assert "ood" in report.metadata["pipeline_stage_metadata"]
        assert "proposal" in report.metadata["pipeline_stage_metadata"]

    @pytest.mark.asyncio
    async def test_file_persistence(self, trained_matcher, test_queries):
        """Test saving and loading discovery reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = NovelEntityMatcher(
                matcher=trained_matcher,
                auto_save=True,
                output_dir=tmpdir,
            )

            # Run discovery (will save to file)
            report = await detector.discover_novel_classes(
                queries=test_queries,
                existing_classes=["physics", "cs", "biology"],
                run_llm_proposal=False,
            )

            # Check that file was created
            assert report.output_file is not None
            assert Path(report.output_file).exists()

            # Load the report
            loaded_report = load_proposals(report.output_file)

            # Verify loaded report matches original
            assert loaded_report.discovery_id == report.discovery_id
            assert len(loaded_report.novel_sample_report.novel_samples) == len(
                report.novel_sample_report.novel_samples
            )

    def test_save_proposals_uses_unique_filenames_per_discovery(self):
        """Reports saved in the same second should not overwrite each other."""
        from novelentitymatcher.novelty.schemas import (
            NovelClassDiscoveryReport,
            NovelSampleReport,
        )

        timestamp = datetime(2026, 3, 17, 16, 33, 28)

        report_one = NovelClassDiscoveryReport(
            discovery_id="first123",
            timestamp=timestamp,
            matcher_config={"matcher_type": "Matcher"},
            detection_config={"combine_method": "intersection"},
            novel_sample_report=NovelSampleReport(
                novel_samples=[],
                detection_strategies=["confidence"],
                config={"combine_method": "intersection"},
                signal_counts={"confidence": 0},
            ),
            metadata={"num_queries": 1},
        )
        report_two = NovelClassDiscoveryReport(
            discovery_id="second45",
            timestamp=timestamp,
            matcher_config={"matcher_type": "Matcher"},
            detection_config={"combine_method": "intersection"},
            novel_sample_report=NovelSampleReport(
                novel_samples=[],
                detection_strategies=["confidence"],
                config={"combine_method": "intersection"},
                signal_counts={"confidence": 0},
            ),
            metadata={"num_queries": 2},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path_one = save_proposals(report_one, output_dir=tmpdir)
            path_two = save_proposals(report_two, output_dir=tmpdir)

            assert path_one != path_two
            assert Path(path_one).exists()
            assert Path(path_two).exists()
            assert load_proposals(path_one).discovery_id == "first123"
            assert load_proposals(path_two).discovery_id == "second45"

    def test_detection_strategies(self, trained_matcher):
        """Test different detection strategies."""
        queries = [
            "quantum physics research",
            "machine learning model",
            "unknown topic here",
        ]

        config = DetectionConfig(
            strategies=["confidence"],
            confidence=ConfidenceConfig(threshold=0.7),
        )

        detector = NoveltyDetector(config=config)

        # Get embeddings and predictions
        embeddings = trained_matcher.model.encode(queries)
        predictions = trained_matcher.match(queries)

        # Convert predictions to list if needed
        if isinstance(predictions, list) and all(
            isinstance(p, dict) for p in predictions
        ):
            predicted_classes = [p.get("id", "unknown") for p in predictions]
            confidences = np.array([p.get("score", 0.5) for p in predictions])
        else:
            predicted_classes = [str(p) for p in predictions]
            confidences = np.ones(len(queries)) * 0.8

        report = detector.detect_novel_samples(
            texts=queries,
            confidences=confidences,
            embeddings=embeddings,
            predicted_classes=predicted_classes,
            reference_embeddings=trained_matcher.get_reference_corpus()["embeddings"],
            reference_labels=trained_matcher.get_reference_corpus()["labels"],
        )

        # Verify confidence strategy was used
        assert "confidence" in report.detection_strategies
        assert "confidence" in report.signal_counts

    @pytest.mark.asyncio
    async def test_batch_discovery(self, trained_matcher):
        """Test batch discovery with multiple query lists."""
        detector = NovelEntityMatcher(
            matcher=trained_matcher,
            auto_save=False,
        )

        query_batches = [
            ["query 1", "query 2"],
            ["query 3", "query 4"],
        ]

        reports = detector.batch_discover(
            queries_batch=query_batches,
            existing_classes=["physics", "cs"],
        )

        # Should return two reports
        assert len(reports) == 2
        assert all(hasattr(r, "discovery_id") for r in reports)


@pytest.mark.skipif(
    not pytest.importorskip("litellm", None),
    reason="litellm not available or no API key configured",
)
class TestLLMIntegration:
    """Tests that require LLM API access."""

    @pytest.fixture
    def detector_with_llm(self, sample_entities, training_data):
        """Create detector with LLM enabled."""
        # Train matcher
        texts = []
        labels = []
        for label, samples in training_data.items():
            texts.extend(samples)
            labels.extend([label] * len(samples))

        matcher = Matcher(
            entities=sample_entities,
            model="minilm",
            threshold=0.6,
        )
        matcher.fit(texts=texts, labels=labels)

        # Create detector with LLM
        detector = NovelEntityMatcher(
            matcher=matcher,
            llm_model="gpt-3.5-turbo",  # Use cheaper model for testing
            auto_save=False,
        )

        return detector

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires actual API key - set manually for testing")
    async def test_llm_proposal_generation(self, detector_with_llm):
        """Test actual LLM proposal generation."""
        queries = [
            "quantum biology research",
            "bioinformatics algorithms",
            "computational chemistry",
        ]

        report = await detector_with_llm.discover_novel_classes(
            queries=queries,
            existing_classes=["physics", "cs", "biology", "chemistry"],
            run_llm_proposal=True,
        )

        # Check that LLM proposals were generated
        assert report.class_proposals is not None
        assert len(report.class_proposals.proposed_classes) > 0

        # Verify proposal structure
        proposal = report.class_proposals.proposed_classes[0]
        assert proposal.name
        assert proposal.description
        assert 0 <= proposal.confidence <= 1
        assert len(proposal.example_samples) > 0
