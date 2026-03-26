"""CLI for running data ingestion scripts."""

import argparse
from pathlib import Path

from novelentitymatcher.ingestion import (
    run_languages,
    run_currencies,
    run_industries,
    run_timezones,
    run_occupations,
    run_products,
    run_universities,
)

INGESTORS = {
    "languages": run_languages,
    "currencies": run_currencies,
    "industries": run_industries,
    "timezones": run_timezones,
    "occupations": run_occupations,
    "products": run_products,
    "universities": run_universities,
    "all": None,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ingest external datasets")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        choices=list(INGESTORS.keys()),
        help="Dataset to ingest (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Base directory for raw downloads (defaults to ./data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Base directory for processed outputs (defaults to ./data/processed)",
    )

    args = parser.parse_args(argv)

    if args.list:
        print("Available datasets:")
        for name in INGESTORS:
            print(f"  - {name}")
        return

    if args.dataset == "all":
        print("Running all ingestions...")
        failures = []
        for name, func in INGESTORS.items():
            if name == "all":
                continue
            print(f"\n{'=' * 50}")
            print(f"Ingesting {name}...")
            print("=" * 50)
            try:
                if func is None:
                    continue
                func(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
            except Exception as e:
                print(f"Error ingesting {name}: {e}")
                failures.append((name, e))
        if failures:
            print("\nIngestion completed with failures:")
            for name, error in failures:
                print(f"  - {name}: {error}")
            raise SystemExit(1)
        print("\nAll ingestions complete!")
    else:
        func = INGESTORS.get(args.dataset)
        if func:
            func(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
        else:
            print(f"Unknown dataset: {args.dataset}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
