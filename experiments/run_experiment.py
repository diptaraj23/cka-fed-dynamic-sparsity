"""Run the main MNIST sparsity sweep experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPARSITIES = (0.5, 0.7, 0.8, 0.9, 0.95)
DEFAULT_METHODS = ("sparse_fedavg", "feddst", "cka_feddst")
CONFIGS = {
    "fedavg": Path("configs/fedavg_mnist.yaml"),
    "sparse_fedavg": Path("configs/sparse_fedavg_mnist.yaml"),
    "feddst": Path("configs/feddst_mnist.yaml"),
    "cka_feddst": Path("configs/cka_feddst_mnist.yaml"),
}


@dataclass(frozen=True)
class RunSpec:
    """One experiment command to execute."""

    method: str
    config_path: Path
    sparsity: float | None = None


def build_parser() -> argparse.ArgumentParser:
    """Create the experiment runner CLI."""

    parser = argparse.ArgumentParser(
        description="Run MNIST sparse methods across sparsity levels."
    )
    parser.add_argument(
        "--include_fedavg_baseline",
        "--include-fedavg-baseline",
        type=parse_bool,
        default=True,
        metavar="{true,false}",
        help="Run dense FedAvg once before sparse methods.",
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=list(DEFAULT_SPARSITIES),
        help="Sparsity levels for sparse methods.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
        help="Sparse methods to run.",
    )
    parser.add_argument(
        "--continue_on_error",
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs after a failed command.",
    )
    parser.add_argument(
        "--dry_run",
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run all requested experiments sequentially."""

    args = build_parser().parse_args(argv)
    specs = build_run_specs(
        methods=args.methods,
        sparsities=args.sparsities,
        include_fedavg_baseline=args.include_fedavg_baseline,
    )
    total_runs = len(specs)
    failures = []

    print(f"Planned runs: {total_runs}")
    for run_index, spec in enumerate(specs, start=1):
        command = build_command(spec)
        sparsity_label = "dense" if spec.sparsity is None else format_sparsity(spec.sparsity)

        print(
            f"\n[Run {run_index}/{total_runs}] "
            f"method={spec.method} sparsity={sparsity_label}",
            flush=True,
        )
        print(f"Command: {format_command(command)}", flush=True)

        if args.dry_run:
            continue

        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((spec, exc.returncode, command))
            print("\nFAILED COMMAND:", format_command(command), file=sys.stderr)
            print(f"Exit code: {exc.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                return exc.returncode

    if args.dry_run:
        print("\nDry run complete. No experiments were launched.")
        return 0

    if failures:
        print("\nSome runs failed:", file=sys.stderr)
        for spec, return_code, command in failures:
            sparsity_label = "dense" if spec.sparsity is None else spec.sparsity
            print(
                f"- method={spec.method} sparsity={sparsity_label} "
                f"exit_code={return_code}: {format_command(command)}",
                file=sys.stderr,
            )
        return 1

    print("\nAll experiments completed.")
    return 0


def build_run_specs(
    methods: list[str],
    sparsities: list[float],
    include_fedavg_baseline: bool = True,
) -> list[RunSpec]:
    """Build the ordered list of experiment runs."""

    specs = []
    if include_fedavg_baseline:
        specs.append(RunSpec(method="fedavg", config_path=CONFIGS["fedavg"]))

    for sparsity in sparsities:
        for method in methods:
            specs.append(
                RunSpec(
                    method=method,
                    config_path=CONFIGS[method],
                    sparsity=sparsity,
                )
            )
    return specs


def build_command(spec: RunSpec) -> list[str]:
    """Build the subprocess command for one experiment."""

    command = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(spec.config_path),
    ]
    if spec.sparsity is not None:
        command.extend(["--sparsity", format_sparsity(spec.sparsity)])
    return command


def parse_bool(value: str) -> bool:
    """Parse a user-facing boolean value."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def format_sparsity(value: float) -> str:
    """Format a sparsity value compactly for CLI use."""

    return f"{value:g}"


def format_command(command: list[str]) -> str:
    """Format a subprocess command for clear terminal output."""

    return subprocess.list2cmdline(command)


if __name__ == "__main__":
    raise SystemExit(main())
