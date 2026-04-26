"""Lifecycle-aware review and promotion storage for discovery proposals."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schemas import (
    NovelClassDiscoveryReport,
    PromotionResult,
    ProposalReviewRecord,
)


class ProposalReviewManager:
    """Persist and update proposal review records for HITL workflows."""

    _ALLOWED_TRANSITIONS = {
        "pending_review": {"approved", "rejected"},
        "approved": {"approved", "promoted"},
        "rejected": {"rejected"},
        "promoted": {"promoted"},
    }

    def __init__(self, storage_path: str | Path = "./proposals/review_records.json"):
        self.storage_path = Path(storage_path)

    def create_records(
        self,
        report: NovelClassDiscoveryReport,
    ) -> list[ProposalReviewRecord]:
        if not report.class_proposals:
            return []

        records = [
            ProposalReviewRecord(
                review_id=str(uuid.uuid4())[:8],
                discovery_id=report.discovery_id,
                proposal_index=index,
                proposal_name=proposal.name,
                proposal=proposal,
                provenance={
                    "discovery_timestamp": report.timestamp.isoformat(),
                    "cluster_ids": list(proposal.source_cluster_ids),
                    "diagnostics": report.diagnostics,
                },
            )
            for index, proposal in enumerate(report.class_proposals.proposed_classes)
        ]
        self.save_records(records)
        return records

    def save_records(self, records: Iterable[ProposalReviewRecord]) -> None:
        payload = [record.model_dump(mode="json") for record in records]
        existing = {record["review_id"]: record for record in self._read_storage()}
        for record in payload:
            existing[record["review_id"]] = record

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(list(existing.values()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(self.storage_path)

    def list_records(
        self, discovery_id: str | None = None
    ) -> list[ProposalReviewRecord]:
        records = [ProposalReviewRecord(**record) for record in self._read_storage()]
        if discovery_id is None:
            return records
        return [record for record in records if record.discovery_id == discovery_id]

    def update_state(
        self,
        review_id: str,
        state: str,
        *,
        notes: str | None = None,
    ) -> ProposalReviewRecord:
        records = self.list_records()
        updated: ProposalReviewRecord | None = None
        now = datetime.now()

        for index, record in enumerate(records):
            if record.review_id != review_id:
                continue
            self._validate_transition(record.state, state, review_id)
            record.state = state  # type: ignore[assignment]
            record.updated_at = now
            record.notes = notes if notes is not None else record.notes
            if state in {"approved", "rejected"}:
                record.reviewed_at = now
            if state == "promoted":
                record.reviewed_at = record.reviewed_at or now
                record.promoted_at = now
            records[index] = record
            updated = record
            break

        if updated is None:
            raise KeyError(f"Unknown review_id: {review_id}")

        self.save_records(records)
        return updated

    def promote(
        self,
        review_id: str,
        *,
        promoter: Callable[[ProposalReviewRecord], Any] | None = None,
        entities: list[dict[str, Any]] | None = None,
        index_updater: Callable[[list[dict[str, Any]]], Any] | None = None,
        retrain_callback: Callable[[], Any] | None = None,
    ) -> PromotionResult:
        current = next(
            (record for record in self.list_records() if record.review_id == review_id),
            None,
        )
        if current is None:
            raise KeyError(f"Unknown review_id: {review_id}")

        if current.state == "pending_review":
            approved = self.update_state(review_id, "approved")
        else:
            self._validate_transition(current.state, "promoted", review_id)
            approved = current
        if promoter is not None:
            promoter(approved)

        promoted = self.update_state(review_id, "promoted")

        entities_added: list[dict[str, Any]] = []
        proposal = promoted.proposal
        new_entity: dict[str, Any] = {
            "id": proposal.name,
            "name": proposal.name,
            "description": proposal.description,
            "examples": list(proposal.example_samples),
        }
        entities_added.append(new_entity)

        if entities is not None:
            entities.extend(entities_added)

        index_updated = False
        if index_updater is not None:
            index_updater(entities_added)
            index_updated = True

        retrain_required = retrain_callback is not None
        if retrain_callback is not None:
            retrain_callback()

        return PromotionResult(
            review_record=promoted,
            entities_added=entities_added,
            index_updated=index_updated,
            retrain_required=retrain_required,
        )

    def promote_with_index_update(
        self,
        review_id: str,
        matcher: Any,
    ) -> PromotionResult:
        """Promote and automatically update the matcher's entity index.

        Args:
            review_id: The review record to promote.
            matcher: A NovelEntityMatcher or similar object with ``entities``
                and optional ``reindex`` / ``fit`` methods.

        Returns:
            PromotionResult with full details of the promotion.
        """
        entities = list(getattr(matcher, "entities", []))

        def index_updater(new_entities: list[dict[str, Any]]) -> None:
            matcher.entities = entities
            reindex = getattr(matcher, "reindex", None)
            if callable(reindex):
                reindex()
            else:
                fit = getattr(matcher, "fit", None)
                if callable(fit):
                    fit()

        def retrain_callback() -> None:
            pass

        return self.promote(
            review_id,
            entities=entities,
            index_updater=index_updater,
            retrain_callback=retrain_callback,
        )

    def _read_storage(self) -> list[dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Review storage must contain a JSON list")
        return payload

    def _validate_transition(
        self,
        current_state: str,
        new_state: str,
        review_id: str,
    ) -> None:
        allowed = self._ALLOWED_TRANSITIONS.get(current_state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid review state transition for {review_id}: "
                f"{current_state} -> {new_state}"
            )
