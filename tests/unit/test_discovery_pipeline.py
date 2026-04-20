from pathlib import Path

from novelentitymatcher import DiscoveryPipeline, Matcher, NovelEntityMatcher
from novelentitymatcher.novelty.schemas import (
    ClassProposal,
    NovelClassAnalysis,
)


def _build_trained_matcher() -> Matcher:
    entities = [
        {"id": "physics", "name": "Quantum Physics"},
        {"id": "biology", "name": "Molecular Biology"},
    ]
    matcher = Matcher(entities=entities, model="minilm", threshold=0.6)
    matcher.fit(
        texts=[
            "quantum mechanics",
            "wave function",
            "gene expression",
            "DNA replication",
        ],
        labels=["physics", "physics", "biology", "biology"],
    )
    return matcher


def test_discovery_pipeline_is_exported():
    assert DiscoveryPipeline is not None


def test_discovery_pipeline_creates_review_records(tmp_path: Path):
    pipeline = DiscoveryPipeline(
        matcher=_build_trained_matcher(),
        auto_save=False,
        review_storage_path=tmp_path / "review_records.json",
    )
    pipeline.novel_entity_matcher.detector.detect_novel_samples = lambda **kwargs: type(
        "Report",
        (),
        {
            "novel_samples": [
                type(
                    "Sample",
                    (),
                    {
                        "text": "quantum biology proteins",
                        "index": 0,
                        "confidence": 0.3,
                        "predicted_class": "physics",
                        "novelty_score": 0.95,
                        "cluster_id": None,
                        "signals": {"confidence": True},
                    },
                )(),
                type(
                    "Sample",
                    (),
                    {
                        "text": "quantum biology enzymes",
                        "index": 1,
                        "confidence": 0.32,
                        "predicted_class": "biology",
                        "novelty_score": 0.94,
                        "cluster_id": None,
                        "signals": {"confidence": True},
                    },
                )(),
            ],
            "detection_strategies": ["confidence"],
        },
    )()
    pipeline.novel_entity_matcher.llm_proposer.propose_from_clusters = lambda **kwargs: (
        NovelClassAnalysis(
            proposed_classes=[
                ClassProposal(
                    name="Quantum Biology",
                    description="Quantum effects in biological systems",
                    confidence=0.91,
                    sample_count=2,
                    example_samples=[
                        "quantum biology proteins",
                        "quantum biology enzymes",
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
    )

    import asyncio

    report = asyncio.run(
        pipeline.discover(
            ["quantum biology proteins", "quantum biology enzymes"],
            run_llm_proposal=True,
        )
    )

    assert len(report.discovery_clusters) == 1
    assert len(report.review_records) == 1
    assert report.review_records[0].state == "pending_review"
    assert (
        pipeline.list_review_records(report.discovery_id)[0].proposal_name
        == "Quantum Biology"
    )


def test_discovery_pipeline_review_lifecycle(tmp_path: Path):
    pipeline = DiscoveryPipeline(
        matcher=_build_trained_matcher(),
        auto_save=False,
        review_storage_path=tmp_path / "review_records.json",
    )
    manager = pipeline.review_manager
    records = manager.create_records(
        type(
            "Report",
            (),
            {
                "discovery_id": "disc123",
                "timestamp": __import__("datetime").datetime.now(),
                "class_proposals": NovelClassAnalysis(
                    proposed_classes=[
                        ClassProposal(
                            name="Quantum Biology",
                            description="Quantum effects in biological systems",
                            confidence=0.91,
                            sample_count=2,
                            example_samples=["a", "b"],
                            justification="coherent",
                        )
                    ],
                    rejected_as_noise=[],
                    analysis_summary="One cluster",
                    cluster_count=1,
                    model_used="test-model",
                ),
                "diagnostics": {},
            },
        )()
    )

    approved = pipeline.approve_proposal(records[0].review_id, notes="looks good")
    promoted = pipeline.promote_proposal(approved.review_id)

    assert approved.state == "approved"
    assert promoted.state == "promoted"


def test_discovery_pipeline_and_novel_entity_matcher_share_discovery_outputs():
    matcher = _build_trained_matcher()
    pipeline = DiscoveryPipeline(matcher=matcher, auto_save=False)
    novelty_matcher = NovelEntityMatcher(matcher=matcher, auto_save=False)

    def detect_novel_samples(**kwargs):
        return type(
            "Report",
            (),
            {
                "novel_samples": [
                    type(
                        "Sample",
                        (),
                        {
                            "text": text,
                            "index": idx,
                            "confidence": 0.3 + idx * 0.01,
                            "predicted_class": "physics" if idx == 0 else "biology",
                            "novelty_score": 0.95,
                            "cluster_id": None,
                            "signals": {"confidence": True},
                        },
                    )()
                    for idx, text in enumerate(kwargs["texts"])
                ],
                "detection_strategies": ["confidence"],
            },
        )()

    def propose_from_clusters(**kwargs):
        return NovelClassAnalysis(
            proposed_classes=[
                ClassProposal(
                    name="Quantum Biology",
                    description="Quantum effects in biological systems",
                    confidence=0.91,
                    sample_count=2,
                    example_samples=[
                        "quantum biology proteins",
                        "quantum biology enzymes",
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

    pipeline.detector.detect_novel_samples = detect_novel_samples
    novelty_matcher.detector.detect_novel_samples = detect_novel_samples
    pipeline.llm_proposer.propose_from_clusters = propose_from_clusters
    novelty_matcher.llm_proposer.propose_from_clusters = propose_from_clusters

    import asyncio

    queries = ["quantum biology proteins", "quantum biology enzymes"]
    pipeline_report = asyncio.run(pipeline.discover(queries, run_llm_proposal=True))
    novelty_report = asyncio.run(
        novelty_matcher.discover_novel_classes(
            queries,
            run_llm_proposal=True,
        )
    )

    assert len(pipeline_report.discovery_clusters) == 1
    assert len(novelty_report.discovery_clusters) == 1
    assert (
        pipeline_report.class_proposals.proposed_classes[0].name
        == novelty_report.class_proposals.proposed_classes[0].name
    )
