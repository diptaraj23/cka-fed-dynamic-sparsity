"""Command-line entry point for federated learning experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.config import (
    DEFAULT_CONFIG,
    apply_cli_overrides,
    config_to_yaml,
    load_yaml,
    merge_configs,
    save_config,
)
from src.data import DataConfig, load_federated_data, make_split_manifest_path
from src.dst import DSTConfig
from src.federated import (
    FederatedConfig,
    run_cka_feddst,
    run_fedavg,
    run_feddst,
    run_sparse_fedavg,
)
from src.models import get_model
from src.sparsity import SparsityConfig
from src.utils import save_csv, seed_everything


METHODS = ("fedavg", "sparse_fedavg", "feddst", "cka_feddst")


def build_parser() -> argparse.ArgumentParser:
    """Create the training CLI without masking YAML values with defaults."""
    parser = argparse.ArgumentParser(
        description="Run simulated federated learning experiments."
    )
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML file.")
    parser.add_argument(
        "--global_config",
        "--global-config",
        dest="global_config",
        type=Path,
        default=Path("configs/global.yaml"),
        help="Shared YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the merged configuration and exit.",
    )
    parser.add_argument(
        "--data-check",
        action="store_true",
        help="Build data loaders and print client label distributions only.",
    )

    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--disable-cudnn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable cuDNN kernels. Useful on HPC GPU builds with cuDNN engine issues.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--reference-size", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument(
        "--save-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save the final global model checkpoint.",
    )
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--sparsity-init", default=None)
    parser.add_argument("--mask-update-interval", type=int, default=None)
    parser.add_argument("--prune-fraction", type=float, default=None)
    parser.add_argument("--regrowth-method", default=None)

    parser.add_argument(
        "--log-cka",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log client CKA scores for methods that support it.",
    )
    parser.add_argument("--cka-interval", type=int, default=None)
    parser.add_argument("--cka-layers", nargs="+", default=None)
    parser.add_argument("--cka-min-sparsity", type=float, default=None)
    parser.add_argument("--cka-max-sparsity", type=float, default=None)
    parser.add_argument("--cka-strength", type=float, default=None)
    parser.add_argument(
        "--cka-target-strength",
        dest="cka_strength",
        type=float,
        default=None,
        help="Alias for --cka-strength.",
    )
    return parser


def load_final_config(args: argparse.Namespace) -> dict:
    """Load defaults, YAML files, and explicit CLI overrides."""
    global_config = load_yaml(args.global_config)
    experiment_config = load_yaml(args.config)
    config = merge_configs(DEFAULT_CONFIG, global_config, experiment_config)
    config = apply_cli_overrides(config, args)
    config["global_config_path"] = str(args.global_config)
    config["experiment_config_path"] = str(args.config) if args.config else None
    return config


def make_run_name(config: dict) -> str:
    """Create a readable, unique identifier for logs and checkpoints."""
    parts = [
        str(config["method"]),
        str(config["dataset"]),
        f"clients{config['num_clients']}",
        f"alpha{config['alpha']}",
    ]
    if str(config["method"]) != "fedavg":
        parts.append(f"sparsity{config.get('sparsity', 0.0)}")
    if str(config["method"]) == "cka_feddst":
        parts.append(f"cka{config.get('cka_strength', 'NA')}")
    parts.extend(
        [
            f"seed{config['seed']}",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        ]
    )
    return "_".join(_safe_token(part) for part in parts)


def _safe_token(value: object) -> str:
    """Keep generated filenames portable and easy to read."""
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in text)


def prepare_output_paths(config: dict) -> dict[str, Path]:
    """Create output directories and return run-specific file paths."""
    output_dir = Path(config["output_dir"])
    log_dir = Path(config["log_dir"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    plot_dir = Path(config["plot_dir"])

    for directory in (output_dir, log_dir, checkpoint_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base_run_name = make_run_name(config)
    for index in range(10_000):
        run_name = base_run_name if index == 0 else f"{base_run_name}_{index:04d}"
        paths = _paths_for_run_name(run_name, log_dir, checkpoint_dir)
        if not any(path.exists() for path in paths.values()):
            return paths

    raise RuntimeError(f"Could not create unique output paths for {base_run_name}.")


def _paths_for_run_name(
    run_name: str,
    log_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, Path]:
    """Build output paths for a run identifier."""

    return {
        "log": log_dir / f"{run_name}.csv",
        "config": log_dir / f"{run_name}_config.yaml",
        "cka_log": log_dir / f"{run_name}_cka.csv",
        "checkpoint": checkpoint_dir / f"{run_name}.pt",
    }


def make_data_config(config: dict) -> DataConfig:
    """Build the data configuration from the merged experiment config."""
    split_dir = config.get("split_dir")
    return DataConfig(
        dataset=str(config["dataset"]),
        data_dir=Path(config["data_dir"]),
        num_clients=int(config["num_clients"]),
        alpha=float(config["alpha"]),
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
        reference_size=int(config["reference_size"]),
        num_workers=int(config["num_workers"]),
        split_dir=Path(split_dir) if split_dir else None,
    )


def make_federated_config(config: dict) -> FederatedConfig:
    """Build the federated training configuration."""
    return FederatedConfig(
        num_clients=int(config["num_clients"]),
        rounds=int(config["rounds"]),
        local_epochs=int(config["local_epochs"]),
        lr=float(config["lr"]),
        seed=int(config["seed"]),
        device=str(config["device"]),
        eval_interval=int(config["eval_interval"]),
    )


def make_sparsity_config(config: dict) -> SparsityConfig:
    """Build the fixed-sparsity configuration."""
    return SparsityConfig(
        target_sparsity=float(config.get("sparsity", 0.0)),
        init_method=str(config.get("sparsity_init", "random")),
        seed=int(config["seed"]),
    )


def make_dst_config(config: dict) -> DSTConfig:
    """Build the dynamic sparse training configuration."""
    return DSTConfig(
        mask_update_interval=int(config["mask_update_interval"]),
        prune_fraction=float(config["prune_fraction"]),
        regrowth_method=str(config["regrowth_method"]),
    )


def run_experiment(config: dict, paths: dict[str, Path]) -> list[dict]:
    """Run the configured federated learning method."""
    seed_everything(int(config["seed"]))
    if bool(config.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled for this run.")
    data_config = make_data_config(config)
    client_loaders, test_loader, reference_loader = load_federated_data(data_config)
    if bool(config.get("data_check", False)):
        return []

    model = get_model(str(config["model"]), str(config["dataset"]))
    federated_config = make_federated_config(config)
    method = str(config["method"])
    cka_layers = tuple(config.get("cka_layers") or ("conv1", "conv2", "fc1"))
    should_log_cka = _should_write_cka_log(config)
    cka_log_path = paths["cka_log"] if should_log_cka else None
    optional_reference_loader = reference_loader if should_log_cka else None
    run_metadata = base_log_metadata(config)

    if method == "fedavg":
        history = run_fedavg(
            model,
            client_loaders,
            test_loader,
            federated_config,
            reference_loader=optional_reference_loader,
            cka_log_path=cka_log_path,
            cka_layers=cka_layers,
            run_metadata=run_metadata,
        )
    elif method == "sparse_fedavg":
        history = run_sparse_fedavg(
            model,
            client_loaders,
            test_loader,
            federated_config,
            make_sparsity_config(config),
            reference_loader=optional_reference_loader,
            cka_log_path=cka_log_path,
            cka_layers=cka_layers,
            run_metadata=run_metadata,
        )
    elif method == "feddst":
        history = run_feddst(
            model,
            client_loaders,
            test_loader,
            federated_config,
            make_sparsity_config(config),
            make_dst_config(config),
            reference_loader=optional_reference_loader,
            cka_log_path=cka_log_path,
            cka_layers=cka_layers,
            run_metadata=run_metadata,
        )
    elif method == "cka_feddst":
        history = run_cka_feddst(
            model,
            client_loaders,
            test_loader,
            reference_loader,
            federated_config,
            make_sparsity_config(config),
            make_dst_config(config),
            cka_interval=int(config["cka_interval"]),
            cka_target_strength=float(config["cka_strength"]),
            cka_layers=cka_layers,
            cka_min_sparsity=float(config["cka_min_sparsity"]),
            cka_max_sparsity=float(config["cka_max_sparsity"]),
            cka_log_path=cka_log_path,
            run_metadata=run_metadata,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    history = add_log_metadata(history, config)
    save_csv(history, paths["log"])
    if bool(config["save_checkpoint"]):
        torch.save(model.state_dict(), paths["checkpoint"])
    return history


def add_log_metadata(history: list[dict], config: dict) -> list[dict]:
    """Add run-level metadata to every training log row."""

    metadata = base_log_metadata(config)
    return [{**metadata, **row} for row in history]


def base_log_metadata(config: dict) -> dict:
    """Return run-level metadata shared by training and CKA logs."""

    sparsity = (
        0.0
        if str(config["method"]) == "fedavg"
        else config.get("sparsity", 0.0)
    )
    return {
        "method": str(config["method"]),
        "dataset": str(config["dataset"]),
        "sparsity": float(sparsity),
        "seed": int(config["seed"]),
        "split_manifest_path": str(config.get("split_manifest_path", "")),
        "cka_strength": (
            float(config["cka_strength"])
            if str(config["method"]) == "cka_feddst"
            else ""
        ),
    }


def _should_write_cka_log(config: dict) -> bool:
    """CKA-FedDST always needs CKA logs; other methods opt in."""
    return str(config["method"]) == "cka_feddst" or bool(config.get("log_cka", False))


def main() -> None:
    """Parse arguments, prepare configuration, and launch training."""
    parser = build_parser()
    args = parser.parse_args()
    config = load_final_config(args)
    config["dry_run"] = bool(args.dry_run)
    config["data_check"] = bool(args.data_check)

    print("Final merged configuration:")
    print(config_to_yaml(config))

    if config["dry_run"]:
        return

    paths = prepare_output_paths(config)
    split_dir = (
        Path(config["split_dir"])
        if config.get("split_dir")
        else paths["log"].parent / "splits"
    )
    config["split_dir"] = str(split_dir)
    config["split_manifest_path"] = str(
        make_split_manifest_path(make_data_config(config))
    )
    save_config(config, paths["config"])
    print(f"Saved merged config to: {paths['config']}")

    history = run_experiment(config, paths)
    if history:
        print(f"Saved training log to: {paths['log']}")
        if _should_write_cka_log(config):
            print(f"Saved CKA log to: {paths['cka_log']}")
        if bool(config["save_checkpoint"]):
            print(f"Saved checkpoint to: {paths['checkpoint']}")


if __name__ == "__main__":
    main()
