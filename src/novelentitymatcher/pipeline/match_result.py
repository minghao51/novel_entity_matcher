"""
Backward-compatible re-exports from core.match_result.

The canonical location for these types is now core.match_result.
"""

from ..core.match_result import (  # noqa: F401
    MatchRecord,
    MatchResultWithMetadata,
    build_match_records,
    build_match_result_with_metadata,
    convert_match_result_to_metadata,
    normalize_candidate_results,
)
