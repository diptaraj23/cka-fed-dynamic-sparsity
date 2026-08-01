"""YAML configuration utilities for experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "dataset": "mnist",
    "model": "small_cnn",
    "method": "fedavg",
    "num_clients": 5,
    "rounds": 1,
    "local_epochs": 1,
    "batch_size": 64,
    "lr": 0.01,
    "seed": 0,
    "alpha": 0.5,
    "device": "auto",
    "disable_cudnn": False,
    "data_dir": "data",
    "output_dir": "results",
    "log_dir": "results/logs",
    "checkpoint_dir": "results/checkpoints",
    "plot_dir": "results/plots",
    "split_dir": None,
    "reference_size": 200,
    "eval_interval": 1,
    "save_checkpoint": False,
    "num_workers": 0,
    "sparsity": 0.0,
    "sparsity_init": "random",
    "mask_update_interval": 1,
    "prune_fraction": 0.1,
    "regrowth_method": "gradient",
    "cka_interval": 1,
    "cka_layers": ["conv1", "conv2", "fc1"],
    "cka_min_sparsity": 0.0,
    "cka_max_sparsity": 0.99,
    "cka_strength": 0.5,
    "log_cka": False,
}


def load_yaml(path: str | Path | None) -> dict:
    """Load a YAML file into a dictionary."""

    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top of {path}")
    return data


def merge_configs(*configs: dict) -> dict:
    """Merge configs from lowest to highest precedence."""

    merged = {}
    for config in configs:
        merged = _deep_merge(merged, config or {})
    return merged


def apply_cli_overrides(config: dict, args) -> dict:
    """Apply explicitly provided command-line arguments to a config."""

    merged = deepcopy(config)
    args_dict = vars(args)

    skip_keys = {"config", "global_config", "dry_run", "data_check"}
    for key, value in args_dict.items():
        if key in skip_keys or value is None:
            continue
        merged[key] = str(value) if isinstance(value, Path) else value

    return merged


def save_config(config: dict, output_path: str | Path) -> Path:
    """Save a config dictionary as YAML."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    return output_path


def config_to_yaml(config: dict) -> str:
    """Render a config dictionary for terminal output."""

    return yaml.safe_dump(config, sort_keys=True).strip()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries."""

    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
