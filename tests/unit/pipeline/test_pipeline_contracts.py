from types import SimpleNamespace

from novelentitymatcher.novelty.clustering.scalable import ScalableClusterer
from novelentitymatcher.novelty.schemas import (
    DiscoveryCluster,
    NovelSampleMetadata,
    NovelSampleReport,
)
from novelentitymatcher.pipeline import (
    ClusterEvidenceStage,
    CommunityDetectionStage,
    MatcherMetadataStage,
    OODDetectionStage,
    PipelineOrchestrator,
    ProposalStage,
    StageContext,
)


def test_internal_pipeline_runs_stages_in_order():
    match_result = SimpleNamespace(
        metadata={"top_k": 3},
        confidences=[0.8],
        embeddings=[[1.0, 0.0]],
        predictions=["physics"],
        candidate_results=[[{"id": "physics", "score": 0.8}]],
    )
    reference = {"embeddings": [[1.0, 0.0]], "labels": ["physics"]}
    detector = SimpleNamespace(
        detect_novel_samples=lambda **kwargs: SimpleNamespace(
            novel_samples=[],
            detection_strategies=["confidence"],
        )
    )
    proposer = SimpleNamespace(
        propose_classes=lambda **kwargs: {"status": "unused-for-empty-report"}
    )

    pipeline = PipelineOrchestrator(
        stages=[
            MatcherMetadataStage(
                collect_sync=lambda inputs: (match_result, reference),
                collect_async=lambda inputs: (match_result, reference),
            ),
            OODDetectionStage(detector=detector, enabled=True),
            CommunityDetectionStage(
                ScalableClusterer(), enabled=True, min_cluster_size=1
            ),
            ClusterEvidenceStage(enabled=True),
            ProposalStage(
                proposer=proposer,
                existing_classes_resolver=lambda: ["physics"],
                enabled=True,
            ),
        ]
    )

    result = pipeline.run(StageContext(inputs=["quantum mechanics"]))

    assert [stage.stage_name for stage in result.stage_results] == [
        "match",
        "ood",
        "cluster",
        "evidence",
        "proposal",
    ]
    assert result.context.artifacts["match_result"] is match_result
    assert result.context.artifacts["reference_corpus"] == reference
    assert result.context.artifacts["novel_sample_report"].detection_strategies == [
        "confidence"
    ]
    assert result.context.artifacts["class_proposals"] is None


def test_cluster_and_evidence_stages_enrich_novel_samples():
    report = NovelSampleReport(
        novel_samples=[
            NovelSampleMetadata(
                text="quantum biology pathway",
                index=0,
                confidence=0.4,
                predicted_class="physics",
                novelty_score=0.9,
            ),
            NovelSampleMetadata(
                text="quantum biology proteins",
                index=1,
                confidence=0.35,
                predicted_class="biology",
                novelty_score=0.92,
            ),
        ]
    )
    match_result = SimpleNamespace(
        embeddings=[[1.0, 0.0], [0.99, 0.01]],
    )

    context = StageContext(
        inputs=["a", "b"],
        artifacts={
            "novel_sample_report": report,
            "match_result": match_result,
        },
    )

    clustered = CommunityDetectionStage(
        ScalableClusterer(), enabled=True, min_cluster_size=1
    ).run(context)
    context.artifacts.update(clustered.artifacts)
    evidenced = ClusterEvidenceStage(enabled=True).run(context)

    clusters = evidenced.artifacts["discovery_clusters"]
    assert len(clusters) == 1
    assert clusters[0].sample_count == 2
    assert "quantum" in clusters[0].keywords
    assert clusters[0].evidence is not None


def test_community_detection_auto_backend_does_not_crash():
    report = NovelSampleReport(
        novel_samples=[
            NovelSampleMetadata(
                text="quantum biology pathway",
                index=0,
                confidence=0.4,
                predicted_class="physics",
                novelty_score=0.9,
            ),
            NovelSampleMetadata(
                text="quantum biology proteins",
                index=1,
                confidence=0.35,
                predicted_class="biology",
                novelty_score=0.92,
            ),
        ]
    )
    match_result = SimpleNamespace(
        embeddings=[[1.0, 0.0], [0.99, 0.01]],
    )
    context = StageContext(
        inputs=["a", "b"],
        artifacts={
            "novel_sample_report": report,
            "match_result": match_result,
        },
    )

    clustered = CommunityDetectionStage(
        ScalableClusterer(backend="auto"),
        enabled=True,
        min_cluster_size=1,
    ).run(context)

    assert clustered.metadata["backend"] in {
        "hdbscan",
        "soptics",
        "umap_hdbscan",
        "fallback_connected_components",
    }
    assert len(clustered.artifacts["discovery_clusters"]) == 1


def test_cluster_evidence_tfidf_is_batch_safe_across_calls():
    from novelentitymatcher.novelty.extraction import ClusterEvidenceExtractor

    extractor = ClusterEvidenceExtractor(method="tfidf", max_keywords=3)

    first_keywords = list(
        extractor.extract(
            cluster_texts=["quantum biology proteins", "quantum biology enzymes"]
        ).keywords
    )
    second_keywords = list(
        extractor.extract(
            cluster_texts=[
                "graph neural routing",
                "graph transformers routing",
                "routing with graph memory",
            ]
        ).keywords
    )

    assert "quantum" in first_keywords
    assert "graph" in second_keywords


def test_proposal_stage_sample_mode_uses_sample_level_proposals():
    proposer = SimpleNamespace()
    proposer.calls = []

    def propose_from_clusters(**kwargs):
        proposer.calls.append("cluster")
        return {"mode": "cluster"}

    def propose_classes(**kwargs):
        proposer.calls.append("sample")
        return {"mode": "sample"}

    proposer.propose_from_clusters = propose_from_clusters
    proposer.propose_classes = propose_classes

    report = NovelSampleReport(
        novel_samples=[
            NovelSampleMetadata(
                text="quantum biology pathway",
                index=0,
                confidence=0.4,
                predicted_class="physics",
                novelty_score=0.9,
            )
        ]
    )
    context = StageContext(
        inputs=["quantum biology pathway"],
        artifacts={
            "novel_sample_report": report,
            "discovery_clusters": [
                DiscoveryCluster(
                    cluster_id=0,
                    sample_indices=[0],
                    sample_count=1,
                    example_texts=["quantum biology pathway"],
                )
            ],
        },
    )

    result = ProposalStage(
        proposer=proposer,
        existing_classes_resolver=lambda: ["physics"],
        proposal_mode="sample",
    ).run(context)

    assert result.artifacts["class_proposals"] == {"mode": "sample"}
    assert proposer.calls == ["sample"]
