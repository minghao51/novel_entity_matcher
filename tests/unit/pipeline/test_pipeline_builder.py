from novelentitymatcher.pipeline.adapters import (
    ClusterEvidenceStage,
    CommunityDetectionStage,
    OODDetectionStage,
    ProposalStage,
)
from novelentitymatcher.pipeline.config import PipelineConfig
from novelentitymatcher.pipeline.pipeline_builder import PipelineBuilder


def test_pipeline_builder_propagates_pipeline_config_stage_options():
    config = PipelineConfig(
        clustering_backend="soptics",
        use_tfidf=False,
        prefer_cluster_level=False,
    )
    builder = PipelineBuilder.from_pipeline_config(
        config,
        collect_sync=lambda inputs: (None, {}),
        collect_async=lambda inputs: (None, {}),
        detector=object(),
        clusterer=None,
        llm_proposer=object(),
        existing_classes_resolver=lambda: ["known"],
    )

    orchestrator = builder.build()

    cluster_stage = next(
        stage
        for stage in orchestrator.stages
        if isinstance(stage, CommunityDetectionStage)
    )
    evidence_stage = next(
        stage
        for stage in orchestrator.stages
        if isinstance(stage, ClusterEvidenceStage)
    )
    proposal_stage = next(
        stage for stage in orchestrator.stages if isinstance(stage, ProposalStage)
    )

    # Verify clusterer was injected with correct backend
    assert cluster_stage.clusterer is not None
    assert cluster_stage.clusterer.backend == "soptics"
    assert evidence_stage.use_tfidf is False
    assert evidence_stage.evidence_method == "centroid"
    assert proposal_stage.force_cluster_level is False


def test_pipeline_builder_propagates_runtime_knobs():
    config = PipelineConfig(
        clustering_backend="hdbscan",
        clustering_metric="euclidean",
        clustering_min_samples=7,
        clustering_cluster_selection_epsilon=0.2,
        proposal_mode="sample",
        proposal_schema_discovery=True,
        proposal_schema_max_attributes=6,
        ood_calibration_mode="conformal",
        ood_calibration_alpha=0.2,
        ood_mahalanobis_mode="global",
    )
    builder = PipelineBuilder.from_pipeline_config(
        config,
        collect_sync=lambda inputs: (None, {}),
        collect_async=lambda inputs: (None, {}),
        detector=object(),
        clusterer=None,
        llm_proposer=object(),
        existing_classes_resolver=lambda: ["known"],
    )

    orchestrator = builder.build()

    ood_stage = next(
        stage for stage in orchestrator.stages if isinstance(stage, OODDetectionStage)
    )
    cluster_stage = next(
        stage
        for stage in orchestrator.stages
        if isinstance(stage, CommunityDetectionStage)
    )
    proposal_stage = next(
        stage for stage in orchestrator.stages if isinstance(stage, ProposalStage)
    )

    assert cluster_stage.clusterer.min_samples == 7
    assert cluster_stage.clusterer.cluster_selection_epsilon == 0.2
    assert cluster_stage.clustering_metric == "euclidean"
    assert proposal_stage.proposal_mode == "sample"
    assert proposal_stage.proposal_schema_discovery is True
    assert proposal_stage.proposal_schema_max_attributes == 6
    assert ood_stage.ood_calibration_mode == "conformal"
    assert ood_stage.ood_calibration_alpha == 0.2
    assert ood_stage.ood_mahalanobis_mode == "global"
