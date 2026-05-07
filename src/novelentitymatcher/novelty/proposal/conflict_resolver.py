from __future__ import annotations

from dataclasses import dataclass

from ..schemas.models import ClassProposal


@dataclass
class ProposalConflict:
    proposal_a: ClassProposal
    proposal_b: ClassProposal
    overlap_type: str
    overlap_score: float
    shared_cluster_ids: list[int]
    recommendation: str


class ProposalConflictResolver:
    def __init__(
        self,
        cluster_overlap_threshold: float = 0.3,
        name_similarity_threshold: float = 0.7,
    ):
        self._cluster_overlap_threshold = cluster_overlap_threshold
        self._name_similarity_threshold = name_similarity_threshold

    def detect_conflicts(
        self, proposals: list[ClassProposal]
    ) -> list[ProposalConflict]:
        conflicts: list[ProposalConflict] = []
        for i in range(len(proposals)):
            for j in range(i + 1, len(proposals)):
                a, b = proposals[i], proposals[j]
                shared = list(set(a.source_cluster_ids) & set(b.source_cluster_ids))
                if not shared and not self._names_similar(a.name, b.name):
                    continue

                cluster_overlap = (
                    len(shared)
                    / max(len(set(a.source_cluster_ids) | set(b.source_cluster_ids)), 1)
                    if a.source_cluster_ids or b.source_cluster_ids
                    else 0.0
                )

                example_jaccard = self._jaccard(
                    set(a.example_samples), set(b.example_samples)
                )
                overlap_score = max(cluster_overlap, example_jaccard)

                if (
                    cluster_overlap < self._cluster_overlap_threshold
                    and example_jaccard < self._cluster_overlap_threshold
                ):
                    if not self._names_similar(a.name, b.name):
                        continue
                    overlap_type = "duplicate"
                    overlap_score = 1.0
                else:
                    overlap_type = self._classify_overlap(
                        a.source_cluster_ids, b.source_cluster_ids, shared
                    )

                recommendation = self._recommend(a, b, overlap_type)
                conflicts.append(
                    ProposalConflict(
                        proposal_a=a,
                        proposal_b=b,
                        overlap_type=overlap_type,
                        overlap_score=overlap_score,
                        shared_cluster_ids=shared,
                        recommendation=recommendation,
                    )
                )
        return conflicts

    def resolve(self, proposals: list[ClassProposal]) -> list[ClassProposal]:
        conflicts = self.detect_conflicts(proposals)
        if not conflicts:
            return list(proposals)

        to_remove: set[int] = set()
        for c in conflicts:
            if c.recommendation == "keep_a":
                to_remove.add(id(c.proposal_b))
            elif c.recommendation == "keep_b":
                to_remove.add(id(c.proposal_a))

        return [p for p in proposals if id(p) not in to_remove]

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    @staticmethod
    def _names_similar(name_a: str, name_b: str) -> bool:
        a_lower = name_a.lower().replace(" ", "")
        b_lower = name_b.lower().replace(" ", "")
        if a_lower == b_lower:
            return True
        if len(a_lower) > 2 and len(b_lower) > 2:
            if a_lower in b_lower or b_lower in a_lower:
                return True
        return False

    @staticmethod
    def _classify_overlap(ids_a: list[int], ids_b: list[int], shared: list[int]) -> str:
        set_a, set_b = set(ids_a), set(ids_b)
        if not set_a or not set_b:
            return "partial"
        if set_a == set_b:
            return "duplicate"
        shared_set = set(shared)
        if shared_set and (shared_set == set_a or shared_set == set_b):
            return "nested"
        return "partial"

    @staticmethod
    def _recommend(a: ClassProposal, b: ClassProposal, overlap_type: str) -> str:
        if overlap_type == "duplicate":
            return "keep_a" if a.confidence >= b.confidence else "keep_b"
        if overlap_type == "nested":
            set_a, set_b = set(a.source_cluster_ids), set(b.source_cluster_ids)
            if len(set_a) >= len(set_b):
                return "keep_a"
            return "keep_b"
        if overlap_type == "partial":
            if a.confidence >= b.confidence and a.sample_count >= b.sample_count:
                return "keep_a"
            return "keep_b"
        return "keep_b"
