"""Run MNIST sparse methods across multiple sparsity levels."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPARSITIES = (0.5, 0.7, 0.8, 0.9, 0.95)
FEDAVG_CONFIG = REPO_ROOT / "configs" / "fedavg_mnist.yaml"
SPARSE_CONFIGS = {
    "sparse_fedavg": REPO_ROOT / "configs" / "sparse_fedavg_mnist.yaml",
    "feddst": REPO_ROOT / "configs" / "feddst_mnist.yaml",
    "cka_feddst": REPO_ROOT / "configs" / "cka_feddst_mnist.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    """Create the sparsity sweep CLI."""

    parser = argparse.ArgumentParser(
        description="Run MNIST sparse baselines across sparsity levels."
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=list(DEFAULT_SPARSITIES),
        help="Sparsity levels to run for sparse methods.",
    )
    parser.add_argument(
        "--skip-fedavg",
        action="store_true",
        help="Skip the dense FedAvg baseline run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dense baseline and sparse-method sparsity sweep."""

    args = build_parser().parse_args(argv)
    commands = build_commands(args.sparsities, skip_fedavg=args.skip_fedavg)

    print(f"Planned runs: {len(commands)}")
    for command in commands:
        print("\nRunning:", format_command(command), flush=True)
        if args.dry_run:
            continue

        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print("\nFAILED COMMAND:", format_command(command), file=sys.stderr)
            print(f"Exit code: {exc.returncode}", file=sys.stderr)
            return exc.returncode

    if args.dry_run:
        print("\nDry run complete. No experiments were launched.")
    else:
        print("\nSparsity sweep completed.")
    return 0


def build_commands(sparsities: list[float], skip_fedavg: bool = False) -> list[list[str]]:
    """Build all training commands for the sweep."""

    commands = []
    if not skip_fedavg:
        commands.append(base_command(FEDAVG_CONFIG))

    for sparsity in sparsities:
        for config_path in SPARSE_CONFIGS.values():
            command = base_command(config_path)
            command.extend(["--sparsity", format_sparsity(sparsity)])
            commands.append(command)

    return commands


def base_command(config_path: Path) -> list[str]:
    """Build a training command for one YAML config."""

    return [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(config_path),
    ]


def format_sparsity(value: float) -> str:
    """Format a sparsity value compactly for CLI use."""

    return f"{value:g}"


def format_command(command: list[str]) -> str:
    """Format a subprocess command for clear terminal output."""

    return subprocess.list2cmdline(command)


if __name__ == "__main__":
    raise SystemExit(main())
