"""CLI for human-in-the-loop review of proposed classes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..utils.logging_config import get_logger
from .storage.review import ProposalReviewManager

logger = get_logger(__name__)

DEFAULT_STORAGE_PATH = "./proposals/review_records.json"


def cmd_list(args: argparse.Namespace) -> None:
    """List pending review records."""
    manager = ProposalReviewManager(args.storage_path)
    records = manager.list_records(discovery_id=args.discovery_id)

    if not args.all:
        records = [r for r in records if r.state == (args.state or "pending_review")]

    if not records:
        logger.info("No records found.")
        return

    logger.info(
        f"{'Review ID':<12} {'Discovery ID':<14} {'Proposal':<25} {'State':<16} {'Notes'}"
    )
    logger.info("-" * 90)
    for record in records:
        notes = (record.notes or "")[:30]
        logger.info(
            f"{record.review_id:<12} {record.discovery_id:<14} "
            f"{record.proposal_name:<25} {record.state:<16} {notes}"
        )

    logger.info(f"\nTotal: {len(records)} record(s)")


def cmd_show(args: argparse.Namespace) -> None:
    """Show details of a specific review record."""
    manager = ProposalReviewManager(args.storage_path)
    records = manager.list_records()
    record = next((r for r in records if r.review_id == args.review_id), None)

    if record is None:
        logger.error(f"Review record '{args.review_id}' not found.")
        sys.exit(1)

    proposal = record.proposal

    logger.info(f"Review ID:        {record.review_id}")
    logger.info(f"Discovery ID:     {record.discovery_id}")
    logger.info(f"Proposal Index:   {record.proposal_index}")
    logger.info(f"Proposal Name:    {record.proposal_name}")
    logger.info(f"State:            {record.state}")
    logger.info(f"Created At:       {record.created_at}")
    logger.info(f"Updated At:       {record.updated_at}")
    logger.info(f"Reviewed At:      {record.reviewed_at or 'N/A'}")
    logger.info(f"Promoted At:      {record.promoted_at or 'N/A'}")
    logger.info(f"Notes:            {record.notes or 'N/A'}")
    logger.info("")
    logger.info("--- Proposal Details ---")
    logger.info(f"Name:             {proposal.name}")
    logger.info(f"Description:      {proposal.description}")
    logger.info(f"Confidence:       {proposal.confidence}")
    logger.info(f"Sample Count:     {proposal.sample_count}")
    logger.info(f"Justification:    {proposal.justification}")
    logger.info(f"Suggested Parent: {proposal.suggested_parent or 'N/A'}")
    logger.info("")
    logger.info("--- Example Samples ---")
    for i, sample in enumerate(proposal.example_samples, 1):
        logger.info(f"  {i}. {sample}")
    logger.info("")
    logger.info("--- Provenance ---")
    logger.info(json.dumps(record.provenance, indent=2, default=str))


def cmd_approve(args: argparse.Namespace) -> None:
    """Approve a proposal."""
    manager = ProposalReviewManager(args.storage_path)
    try:
        record = manager.update_state(args.review_id, "approved", notes=args.notes)
        logger.info(f"Approved review '{args.review_id}' -> state: {record.state}")
    except (KeyError, ValueError) as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def cmd_reject(args: argparse.Namespace) -> None:
    """Reject a proposal."""
    manager = ProposalReviewManager(args.storage_path)
    try:
        record = manager.update_state(args.review_id, "rejected", notes=args.notes)
        logger.info(f"Rejected review '{args.review_id}' -> state: {record.state}")
    except (KeyError, ValueError) as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def cmd_promote(args: argparse.Namespace) -> None:
    """Promote an approved proposal to known entities."""
    manager = ProposalReviewManager(args.storage_path)
    try:
        result = manager.promote(args.review_id)
        logger.info(
            f"Promoted review '{args.review_id}' -> state: {result.review_record.state}"
        )
        logger.info(f"Entities added: {len(result.entities_added)}")
        for entity in result.entities_added:
            logger.info(f"  - {entity['name']}")
        logger.info(f"Index updated: {result.index_updated}")
        logger.info(f"Retrain required: {result.retrain_required}")
    except (KeyError, ValueError) as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def cmd_stats(args: argparse.Namespace) -> None:
    """Show review statistics."""
    manager = ProposalReviewManager(args.storage_path)
    records = manager.list_records()

    if not records:
        logger.info("No records found.")
        return

    state_counts: dict[str, int] = {}
    discovery_counts: dict[str, int] = {}

    for record in records:
        state_counts[record.state] = state_counts.get(record.state, 0) + 1
        discovery_counts[record.discovery_id] = (
            discovery_counts.get(record.discovery_id, 0) + 1
        )

    logger.info("--- Review Statistics ---")
    logger.info(f"Total records: {len(records)}")
    logger.info("")
    logger.info("By State:")
    for state, count in sorted(state_counts.items()):
        logger.info(f"  {state:<20} {count}")
    logger.info("")
    logger.info("By Discovery ID:")
    for discovery_id, count in sorted(discovery_counts.items()):
        logger.info(f"  {discovery_id:<20} {count}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="novelentitymatcher-review",
        description="Human-in-the-loop review of proposed classes",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=DEFAULT_STORAGE_PATH,
        help=f"Path to review records JSON (default: {DEFAULT_STORAGE_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    list_parser = subparsers.add_parser("list", help="List pending review records")
    list_parser.add_argument(
        "--discovery-id", default=None, help="Filter by discovery ID"
    )
    list_parser.add_argument(
        "--state",
        default=None,
        choices=["pending_review", "approved", "rejected", "promoted"],
        help="Filter by state",
    )
    list_parser.add_argument(
        "--all", action="store_true", help="Show all records regardless of state"
    )

    # show
    show_parser = subparsers.add_parser(
        "show", help="Show details of a specific review record"
    )
    show_parser.add_argument("review_id", help="Review record ID")

    # approve
    approve_parser = subparsers.add_parser("approve", help="Approve a proposal")
    approve_parser.add_argument("review_id", help="Review record ID")
    approve_parser.add_argument("--notes", default=None, help="Optional review notes")

    # reject
    reject_parser = subparsers.add_parser("reject", help="Reject a proposal")
    reject_parser.add_argument("review_id", help="Review record ID")
    reject_parser.add_argument("--notes", default=None, help="Optional review notes")

    # promote
    promote_parser = subparsers.add_parser(
        "promote", help="Promote an approved proposal"
    )
    promote_parser.add_argument("review_id", help="Review record ID")

    # stats
    subparsers.add_parser("stats", help="Show review statistics")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "list": cmd_list,
        "show": cmd_show,
        "approve": cmd_approve,
        "reject": cmd_reject,
        "promote": cmd_promote,
        "stats": cmd_stats,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
