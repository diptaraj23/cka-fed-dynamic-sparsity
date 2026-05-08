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
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of FedAvg communication rounds.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local epochs per client each round.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Client learning rate for SGD.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, or cuda.",
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

    from .data import DataConfig, load_mnist
    from .federated import FederatedConfig, run_fedavg
    from .models import get_model
    from .utils import save_csv

    data_config = DataConfig(
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        alpha=args.alpha,
        batch_size=args.batch_size,
        seed=args.seed,
        reference_size=args.reference_size,
    )
    client_loaders, test_loader, _ = load_mnist(data_config)

    model = get_model("small_cnn", "mnist")
    fed_config = FederatedConfig(
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    logs = run_fedavg(model, client_loaders, test_loader, fed_config)

    alpha_text = str(args.alpha).replace(".", "p")
    log_path = (
        args.output_dir
        / "logs"
        / f"fedavg_mnist_clients{args.num_clients}_alpha{alpha_text}_seed{args.seed}.csv"
    )
    save_csv(logs, log_path)
    print(f"Saved logs to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
