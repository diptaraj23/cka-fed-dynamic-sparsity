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
        default=None,
        help="Client learning rate for SGD. Defaults depend on the method.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fedavg", "sparse_fedavg", "feddst"],
        default="fedavg",
        help="Training method to run.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.0,
        help="Target global unstructured sparsity for sparse methods.",
    )
    parser.add_argument(
        "--sparsity-init",
        type=str,
        choices=["random", "magnitude"],
        default="random",
        help="Mask initialization rule for sparse methods.",
    )
    parser.add_argument(
        "--mask-update-interval",
        type=int,
        default=1,
        help="FedDST mask update interval in communication rounds.",
    )
    parser.add_argument(
        "--prune-fraction",
        type=float,
        default=0.1,
        help="Fraction of active weights to prune per layer on FedDST updates.",
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
    from .dst import DSTConfig
    from .federated import FederatedConfig, run_fedavg, run_feddst, run_sparse_fedavg
    from .models import get_model
    from .sparsity import SparsityConfig
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
    learning_rate = args.lr
    if learning_rate is None:
        learning_rate = 0.5 if args.method in {"sparse_fedavg", "feddst"} else 0.05

    fed_config = FederatedConfig(
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=learning_rate,
        seed=args.seed,
        device=args.device,
    )
    if args.method in {"sparse_fedavg", "feddst"}:
        sparsity_config = SparsityConfig(
            target_sparsity=args.sparsity,
            init_method=args.sparsity_init,
            seed=args.seed,
        )

    if args.method == "feddst":
        dst_config = DSTConfig(
            mask_update_interval=args.mask_update_interval,
            prune_fraction=args.prune_fraction,
        )
        logs = run_feddst(
            model,
            client_loaders,
            test_loader,
            fed_config,
            sparsity_config,
            dst_config,
        )
    elif args.method == "sparse_fedavg":
        logs = run_sparse_fedavg(
            model,
            client_loaders,
            test_loader,
            fed_config,
            sparsity_config,
        )
    else:
        logs = run_fedavg(model, client_loaders, test_loader, fed_config)

    alpha_text = str(args.alpha).replace(".", "p")
    sparsity_text = str(args.sparsity).replace(".", "p")
    log_name = (
        f"{args.method}_mnist_clients{args.num_clients}_alpha{alpha_text}"
        f"_sparsity{sparsity_text}_seed{args.seed}.csv"
    )
    log_path = (
        args.output_dir
        / "logs"
        / log_name
    )
    save_csv(logs, log_path)
    print(f"Saved logs to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
