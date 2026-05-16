"""Aggregate raw experiment logs into averaged CSV files."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.aggregation import aggregate_results


def build_parser() -> argparse.ArgumentParser:
    """Create the aggregation CLI."""

    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed experiment logs into mean/std CSVs."
    )
    parser.add_argument(
        "--suite",
        choices=("multiseed", "cka_strength"),
        required=True,
        help="Experiment suite represented by the input log directory.",
    )
    parser.add_argument(
        "--log_dir",
        "--log-dir",
        dest="log_dir",
        type=Path,
        required=True,
        help="Raw suite log directory, e.g. results/logs/multiseed/<suite_id>.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory where averaged CSV files will be saved.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Aggregate logs and print saved output paths."""

    args = build_parser().parse_args(argv)
    saved_paths = aggregate_results(args.log_dir, args.output_dir, args.suite)
    for path in saved_paths:
        print(f"Saved averaged result: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
