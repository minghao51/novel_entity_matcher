"""Tests for backward-compatible DiscoveryPipeline aliases."""

import pytest

import novelentitymatcher.pipeline as pipeline_module
from novelentitymatcher.novelty.entity_matcher import NovelEntityMatcher
from novelentitymatcher.pipeline.discovery import DiscoveryPipeline


def test_pipeline_discovery_alias_points_to_novel_entity_matcher() -> None:
    assert DiscoveryPipeline is NovelEntityMatcher


def test_pipeline_module_getattr_discovery_pipeline() -> None:
    resolved = pipeline_module.__getattr__("DiscoveryPipeline")
    assert resolved is NovelEntityMatcher


def test_pipeline_module_getattr_unknown_raises() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        pipeline_module.__getattr__("does_not_exist")
