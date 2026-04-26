"""
Storage functionality for novelty detection.

This module contains persistence and indexing utilities
for storing proposals and searching embeddings.
"""

from .index import ANNBackend, ANNIndex
from .persistence import list_proposals, load_proposals, save_proposals
from .review import ProposalReviewManager

__all__ = [
    "ANNBackend",
    "ANNIndex",
    "ProposalReviewManager",
    "list_proposals",
    "load_proposals",
    "save_proposals",
]
