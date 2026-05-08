"""Command-line entry point for training experiments."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create the training argument parser."""

    parser = argparse.ArgumentParser(
        description="Simulated federated learning experiments on MNIST."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to an experiment config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for logs, checkpoints, and plots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the command-line interface without running training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the training placeholder."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        print("Dry run complete. Training is not implemented yet.")
        return 0

    print("Training scaffold is ready. Full experiment logic is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
