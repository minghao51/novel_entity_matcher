"""Backward-compatible alias for DiscoveryPipeline.

Use novelentitymatcher.NovelEntityMatcher directly for new code.
"""

from ..novelty.entity_matcher import NovelEntityMatcher as DiscoveryPipeline

__all__ = ["DiscoveryPipeline"]
