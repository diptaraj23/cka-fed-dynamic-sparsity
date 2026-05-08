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
    parser.add_argument(
        "--data-check",
        action="store_true",
        help="Load MNIST and print simulated client label distributions.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory where MNIST will be downloaded or read.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Number of simulated federated clients.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration for label-skew partitioning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for all MNIST dataloaders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic partitioning and loaders.",
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        default=200,
        help="Balanced reference subset size for CKA.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the training placeholder."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.data_check:
        from .data import DataConfig, load_mnist

        data_config = DataConfig(
            data_dir=args.data_dir,
            num_clients=args.num_clients,
            alpha=args.alpha,
            batch_size=args.batch_size,
            seed=args.seed,
            reference_size=args.reference_size,
        )
        client_loaders, test_loader, reference_loader = load_mnist(data_config)
        print(
            "Data check complete: "
            f"{len(client_loaders)} client loaders, "
            f"{len(test_loader.dataset)} test samples, "
            f"{len(reference_loader.dataset)} reference samples."
        )
        return 0

    if args.dry_run:
        print("Dry run complete. Training is not implemented yet.")
        return 0

    print("Training scaffold is ready. Full experiment logic is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
