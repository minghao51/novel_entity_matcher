from novelentitymatcher.pipeline.adapters import (
    ClusterEvidenceStage,
    CommunityDetectionStage,
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
        stage for stage in orchestrator.stages if isinstance(stage, CommunityDetectionStage)
    )
    evidence_stage = next(
        stage for stage in orchestrator.stages if isinstance(stage, ClusterEvidenceStage)
    )
    proposal_stage = next(
        stage for stage in orchestrator.stages if isinstance(stage, ProposalStage)
    )

    assert cluster_stage.backend_name == "soptics"
    assert evidence_stage.use_tfidf is False
    assert proposal_stage.force_cluster_level is False
