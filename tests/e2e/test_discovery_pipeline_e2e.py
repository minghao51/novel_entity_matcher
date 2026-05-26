import asyncio

import numpy as np
import pytest

from novelentitymatcher import DiscoveryPipeline
from novelentitymatcher.pipeline.config import PipelineConfig
from novelentitymatcher.pipeline.match_result import (
    MatchRecord,
    MatchResultWithMetadata,
)


class _FakeMatcher:
    def __init__(self):
        self.entities = [
            {"id": "physics", "name": "Quantum Physics"},
            {"id": "biology", "name": "Molecular Biology"},
        ]
        self.model_name = "e2e-smoke-model"
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
            "source": "e2e",
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


@pytest.mark.e2e
def test_e2e_discovery_pipeline_happy_path_without_llm():
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
