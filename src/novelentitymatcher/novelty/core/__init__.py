"""
Core novelty detection functionality.

This module contains the main detector orchestration, strategy registry,
and signal combination logic.
"""

from .detector import NoveltyDetector
from .metadata import MetadataBuilder
from .signal_combiner import SignalCombiner
from .strategies import StrategyRegistry

__all__ = [
    "MetadataBuilder",
    "NoveltyDetector",
    "SignalCombiner",
    "StrategyRegistry",
]
