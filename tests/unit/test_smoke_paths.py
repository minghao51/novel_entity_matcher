import asyncio

import numpy as np
import pytest

from novelentitymatcher import DiscoveryPipeline, Matcher
from novelentitymatcher.novelty.schemas import ClassProposal, NovelClassAnalysis
from novelentitymatcher.novelty.storage.review import ProposalReviewManager
from novelentitymatcher.pipeline.config import PipelineConfig
from novelentitymatcher.pipeline.match_result import (
    MatchRecord,
    MatchResultWithMetadata,
)


@pytest.mark.smoke
def test_smoke_core_matcher_zero_shot_path(monkeypatch):
    vectors = {
        "germany": [1.0, 0.0],
        "deutschland": [1.0, 0.0],
        "france": [0.0, 1.0],
    }

    class FakeModel:
        def __init__(self, _model_name):
            pass

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, batch_size=None):
            del batch_size
            if isinstance(texts, str):
                texts = [texts]
            return np.asarray([vectors.get(t.lower(), [0.0, 0.0]) for t in texts])

    monkeypatch.setattr("novelentitymatcher.core.matcher.SentenceTransformer", FakeModel)

    matcher = Matcher(
        entities=[
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "FR", "name": "France"},
        ],
        model="minilm",
        mode="zero-shot",
        threshold=0.0,
    )
    matcher.fit()
    result = matcher.match("Deutschland")

    assert result is not None
    assert result["id"] == "DE"


class _FakeMatcher:
    def __init__(self):
        self.entities = [
            {"id": "physics", "name": "Quantum Physics"},
            {"id": "biology", "name": "Molecular Biology"},
        ]
        self.model_name = "smoke-model"
        self.threshold = 0.6
        self._training_mode = "zero-shot"

    def fit(self, *args, **kwargs):
        del args, kwargs
        return self

    async def fit_async(self, *args, **kwargs):
        del args, kwargs
        return self

    def get_reference_corpus(self):
        return {
            "texts": ["quantum physics", "molecular biology"],
            "labels": ["physics", "biology"],
            "embeddings": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            "source": "smoke",
        }

    def match(self, queries, return_metadata=True, top_k=5):
        del return_metadata, top_k
        texts = list(queries)
        predictions = ["physics" for _ in texts]
        confidences = np.asarray([0.91 for _ in texts], dtype=float)
        embeddings = np.asarray([[1.0, 0.0] for _ in texts], dtype=float)
        candidate_results = [[{"id": "physics", "score": 0.91}] for _ in texts]
        records = [
            MatchRecord(
                text=text,
                predicted_id="physics",
                confidence=0.91,
                embedding=np.asarray([1.0, 0.0]),
                candidates=[{"id": "physics", "score": 0.91}],
            )
            for text in texts
        ]
        return MatchResultWithMetadata(
            predictions=predictions,
            confidences=confidences,
            embeddings=embeddings,
            metadata={"texts": texts, "top_k": 1},
            candidate_results=candidate_results,
            records=records,
        )

    async def match_async(self, queries, return_metadata=True, top_k=5):
        del return_metadata, top_k
        return self.match(queries)


@pytest.mark.smoke
def test_smoke_discovery_pipeline_happy_path_without_llm():
    pipeline = DiscoveryPipeline(
        matcher=_FakeMatcher(),
        auto_save=False,
        config=PipelineConfig(proposal_enabled=False),
    )
    pipeline.detector.detect_novel_samples = lambda **kwargs: type(
        "Report",
        (),
        {
            "novel_samples": [
                type(
                    "Sample",
                    (),
                    {
                        "text": kwargs["texts"][0],
                        "index": 0,
                        "confidence": 0.3,
                        "predicted_class": "physics",
                        "novelty_score": 0.95,
                        "cluster_id": None,
                        "signals": {"confidence": True},
                    },
                )()
            ],
            "detection_strategies": ["confidence"],
            "config": {},
            "signal_counts": {"confidence": 1},
        },
    )()

    report = asyncio.run(pipeline.discover(["quantum protein"], run_llm_proposal=False))

    assert report.discovery_id
    assert len(report.novel_sample_report.novel_samples) == 1
    assert report.class_proposals is None


@pytest.mark.smoke
def test_smoke_review_lifecycle(tmp_path):
    manager = ProposalReviewManager(tmp_path / "review_records.json")
    report = type(
        "Report",
        (),
        {
            "discovery_id": "smoke123",
            "timestamp": __import__("datetime").datetime.now(),
            "diagnostics": {},
            "class_proposals": NovelClassAnalysis(
                proposed_classes=[
                    ClassProposal(
                        name="Quantum Biology",
                        description="desc",
                        confidence=0.9,
                        sample_count=2,
                        example_samples=["a", "b"],
                        justification="coherent",
                    )
                ],
                rejected_as_noise=[],
                analysis_summary="one cluster",
                cluster_count=1,
                model_used="test",
            ),
        },
    )()

    records = manager.create_records(report)
    approved = manager.update_state(records[0].review_id, "approved")
    promoted = manager.promote(approved.review_id)

    assert approved.state == "approved"
    assert promoted.state == "promoted"
