"""Run dataset-specific experiment suites with isolated output folders."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPARSITIES = (0.5, 0.7, 0.8, 0.9, 0.95)
DEFAULT_SEEDS = (42, 7, 13, 21, 100)
DEFAULT_CKA_STRENGTHS = (0.2, 0.5, 0.8, 0.9, 1.0)
DEFAULT_METHODS = ("sparse_fedavg", "feddst", "cka_feddst")
SUITES = ("sparsity", "multiseed", "cka_strength", "all")
DATASETS = ("mnist", "fashion_mnist", "cifar10")
GLOBAL_CONFIGS = {
    "mnist": Path("configs/global.yaml"),
    "fashion_mnist": Path("configs/global_fashion_mnist.yaml"),
    "cifar10": Path("configs/global_cifar10.yaml"),
}
CONFIGS = {
    "fedavg": Path("configs/fedavg_mnist.yaml"),
    "sparse_fedavg": Path("configs/sparse_fedavg_mnist.yaml"),
    "feddst": Path("configs/feddst_mnist.yaml"),
    "cka_feddst": Path("configs/cka_feddst_mnist.yaml"),
}
MANIFEST_COLUMNS = (
    "suite_id",
    "suite",
    "run_index",
    "total_runs",
    "method",
    "seed",
    "sparsity",
    "cka_strength",
    "cka_signal",
    "global_config",
    "config",
    "log_dir",
    "checkpoint_dir",
    "plot_dir",
    "command",
    "status",
)


@dataclass(frozen=True)
class RunSpec:
    """One experiment command to execute."""

    suite: str
    suite_id: str
    method: str
    global_config_path: Path
    config_path: Path
    seed: int
    sparsity: float | None = None
    cka_strength: float | None = None
    cka_signal: str | None = None
    log_dir: Path | None = None
    checkpoint_dir: Path | None = None
    plot_dir: Path | None = None
    manifest_path: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    """Create the experiment runner CLI."""

    parser = argparse.ArgumentParser(
        description="Run federated-learning experiment suites."
    )
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default="sparsity",
        help="Experiment suite to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="mnist",
        help="Dataset global configuration to use.",
    )
    parser.add_argument(
        "--suite-id",
        default=None,
        help="Optional suite folder name for organized multi-run experiments.",
    )
    parser.add_argument(
        "--include_fedavg_baseline",
        "--include-fedavg-baseline",
        type=parse_bool,
        default=True,
        metavar="{true,false}",
        help="Run dense FedAvg before sparse methods when the suite supports it.",
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
        help="Sparse methods to run for sparsity and multiseed suites.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Seeds for multiseed and CKA-strength suites.",
    )
    parser.add_argument(
        "--cka-strengths",
        "--cka_strengths",
        dest="cka_strengths",
        type=float,
        nargs="+",
        default=list(DEFAULT_CKA_STRENGTHS),
        help="CKA-strength values for the CKA-strength suite.",
    )
    parser.add_argument(
        "--cka-signal",
        choices=("similarity", "drift"),
        default=None,
        help="Optional CKA-FedDST signal mode override.",
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
        help="Print commands without running them or writing manifests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run all requested experiments sequentially."""

    args = build_parser().parse_args(argv)
    suite_id = (
        safe_token(args.suite_id)
        if args.suite_id
        else make_suite_id(args.suite, args.dataset)
    )
    specs = build_run_specs(args, suite_id)
    total_runs = len(specs)
    manifest_rows = build_manifest_rows(specs)
    failures = []

    print(f"Suite: {args.suite}")
    print(f"Dataset: {args.dataset}")
    print(f"Suite id: {suite_id}")
    print(f"Planned runs: {total_runs}")
    effective_seeds = sorted({spec.seed for spec in specs})
    print(f"Seeds: {' '.join(str(seed) for seed in effective_seeds)}")
    print(f"Sparsities: {' '.join(format_float(value) for value in args.sparsities)}")
    if args.suite in {"cka_strength", "all"}:
        strengths = " ".join(format_float(value) for value in args.cka_strengths)
        print(f"CKA strengths: {strengths}")

    if not args.dry_run:
        write_manifests(manifest_rows)

    for run_index, spec in enumerate(specs, start=1):
        command = build_command(spec)
        row = manifest_rows[run_index - 1] if manifest_rows else None
        sparsity_label = "dense" if spec.sparsity is None else format_float(spec.sparsity)
        cka_label = "yaml" if spec.cka_strength is None else format_float(spec.cka_strength)

        print(
            f"\n[Run {run_index}/{total_runs}] "
            f"suite={spec.suite} method={spec.method} seed={spec.seed} "
            f"sparsity={sparsity_label} cka_strength={cka_label}",
            flush=True,
        )
        if spec.log_dir is not None:
            print(f"Output log dir: {spec.log_dir}", flush=True)
        print(f"Command: {format_command(command)}", flush=True)

        if args.dry_run:
            continue

        if row is not None:
            row["status"] = "running"
            write_manifests(manifest_rows)

        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((spec, exc.returncode, command))
            if row is not None:
                row["status"] = "failed"
                if not args.continue_on_error:
                    mark_remaining_skipped(manifest_rows, run_index)
                write_manifests(manifest_rows)
            print("\nFAILED COMMAND:", format_command(command), file=sys.stderr)
            print(f"Exit code: {exc.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                return exc.returncode
        else:
            if row is not None:
                row["status"] = "passed"
                write_manifests(manifest_rows)

    if args.dry_run:
        print("\nDry run complete. No experiments were launched.")
        return 0

    if failures:
        print("\nSome runs failed:", file=sys.stderr)
        for spec, return_code, command in failures:
            sparsity_label = "dense" if spec.sparsity is None else spec.sparsity
            print(
                f"- suite={spec.suite} method={spec.method} seed={spec.seed} "
                f"sparsity={sparsity_label} exit_code={return_code}: "
                f"{format_command(command)}",
                file=sys.stderr,
            )
        return 1

    print("\nAll experiments completed.")
    return 0


def build_run_specs(args: argparse.Namespace, suite_id: str) -> list[RunSpec]:
    """Build the ordered list of experiment runs."""

    if args.suite == "sparsity":
        return build_sparsity_specs(args, suite_id)
    if args.suite == "multiseed":
        return build_multiseed_specs(args, suite_id)
    if args.suite == "cka_strength":
        return build_cka_strength_specs(args, suite_id)
    return [
        *build_multiseed_specs(args, suite_id),
        *build_cka_strength_specs(args, suite_id),
    ]


def build_sparsity_specs(args: argparse.Namespace, suite_id: str) -> list[RunSpec]:
    """Build the original single-seed sparsity sweep."""

    specs = []
    if args.include_fedavg_baseline:
        specs.append(
            RunSpec(
                suite="sparsity",
                suite_id=suite_id,
                method="fedavg",
                global_config_path=global_config_path(args.dataset),
                config_path=CONFIGS["fedavg"],
                seed=42,
            )
        )

    for sparsity in args.sparsities:
        for method in args.methods:
            specs.append(
                RunSpec(
                    suite="sparsity",
                    suite_id=suite_id,
                    method=method,
                    global_config_path=global_config_path(args.dataset),
                    config_path=CONFIGS[method],
                    seed=42,
                    sparsity=sparsity,
                    cka_signal=args.cka_signal if method == "cka_feddst" else None,
                )
            )
    return specs


def build_multiseed_specs(args: argparse.Namespace, suite_id: str) -> list[RunSpec]:
    """Build the full baseline comparison across seeds."""

    specs = []
    manifest_path = Path("results/logs/multiseed") / suite_id / "manifest.csv"
    plot_dir = Path("results/plots/multiseed") / suite_id

    for seed in args.seeds:
        log_dir = Path("results/logs/multiseed") / suite_id / f"seed_{seed}"
        checkpoint_dir = (
            Path("results/checkpoints/multiseed") / suite_id / f"seed_{seed}"
        )
        if args.include_fedavg_baseline:
            specs.append(
                RunSpec(
                    suite="multiseed",
                    suite_id=suite_id,
                    method="fedavg",
                    global_config_path=global_config_path(args.dataset),
                    config_path=CONFIGS["fedavg"],
                    seed=seed,
                    log_dir=log_dir,
                    checkpoint_dir=checkpoint_dir,
                    plot_dir=plot_dir,
                    manifest_path=manifest_path,
                )
            )

        for sparsity in args.sparsities:
            for method in args.methods:
                specs.append(
                    RunSpec(
                        suite="multiseed",
                        suite_id=suite_id,
                        method=method,
                        global_config_path=global_config_path(args.dataset),
                        config_path=CONFIGS[method],
                        seed=seed,
                        sparsity=sparsity,
                        cka_signal=args.cka_signal if method == "cka_feddst" else None,
                        log_dir=log_dir,
                        checkpoint_dir=checkpoint_dir,
                        plot_dir=plot_dir,
                        manifest_path=manifest_path,
                    )
                )
    return specs


def build_cka_strength_specs(args: argparse.Namespace, suite_id: str) -> list[RunSpec]:
    """Build the full-factorial CKA-strength sweep."""

    specs = []
    manifest_path = (
        Path("results/logs/cka_strength_sweep") / suite_id / "manifest.csv"
    )
    plot_dir = Path("results/plots/cka_strength_sweep") / suite_id

    for strength in args.cka_strengths:
        strength_dir = f"strength_{format_path_float(strength)}"
        for seed in args.seeds:
            log_dir = (
                Path("results/logs/cka_strength_sweep")
                / suite_id
                / strength_dir
                / f"seed_{seed}"
            )
            checkpoint_dir = (
                Path("results/checkpoints/cka_strength_sweep")
                / suite_id
                / strength_dir
                / f"seed_{seed}"
            )
            for sparsity in args.sparsities:
                specs.append(
                    RunSpec(
                        suite="cka_strength",
                        suite_id=suite_id,
                        method="cka_feddst",
                        global_config_path=global_config_path(args.dataset),
                        config_path=CONFIGS["cka_feddst"],
                        seed=seed,
                        sparsity=sparsity,
                        cka_strength=strength,
                        cka_signal=args.cka_signal,
                        log_dir=log_dir,
                        checkpoint_dir=checkpoint_dir,
                        plot_dir=plot_dir,
                        manifest_path=manifest_path,
                    )
                )
    return specs


def build_command(spec: RunSpec) -> list[str]:
    """Build the subprocess command for one experiment."""

    command = [
        sys.executable,
        "-m",
        "src.train",
        "--global-config",
        str(spec.global_config_path),
        "--config",
        str(spec.config_path),
        "--seed",
        str(spec.seed),
    ]
    if spec.sparsity is not None:
        command.extend(["--sparsity", format_float(spec.sparsity)])
    if spec.cka_strength is not None:
        command.extend(["--cka-strength", format_float(spec.cka_strength)])
    if spec.cka_signal is not None:
        command.extend(["--cka-signal", spec.cka_signal])
    if spec.log_dir is not None:
        command.extend(["--log-dir", str(spec.log_dir)])
    if spec.checkpoint_dir is not None:
        command.extend(["--checkpoint-dir", str(spec.checkpoint_dir)])
    if spec.plot_dir is not None:
        command.extend(["--plot-dir", str(spec.plot_dir)])
    return command


def build_manifest_rows(specs: list[RunSpec]) -> list[dict]:
    """Build manifest rows with pending statuses."""

    rows = []
    total_runs = len(specs)
    for run_index, spec in enumerate(specs, start=1):
        rows.append(
            {
                "suite_id": spec.suite_id,
                "suite": spec.suite,
                "run_index": run_index,
                "total_runs": total_runs,
                "method": spec.method,
                "seed": spec.seed,
                "sparsity": "" if spec.sparsity is None else format_float(spec.sparsity),
                "cka_strength": (
                    "" if spec.cka_strength is None else format_float(spec.cka_strength)
                ),
                "cka_signal": "" if spec.cka_signal is None else spec.cka_signal,
                "global_config": str(spec.global_config_path),
                "config": str(spec.config_path),
                "log_dir": "" if spec.log_dir is None else str(spec.log_dir),
                "checkpoint_dir": (
                    "" if spec.checkpoint_dir is None else str(spec.checkpoint_dir)
                ),
                "plot_dir": "" if spec.plot_dir is None else str(spec.plot_dir),
                "command": format_command(build_command(spec)),
                "status": "pending",
                "_manifest_path": spec.manifest_path,
            }
        )
    return rows


def write_manifests(rows: list[dict]) -> None:
    """Write one manifest CSV for each organized suite."""

    paths = sorted(
        {
            row["_manifest_path"]
            for row in rows
            if row.get("_manifest_path") is not None
        }
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        suite_rows = [row for row in rows if row.get("_manifest_path") == path]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in suite_rows:
                writer.writerow({column: row[column] for column in MANIFEST_COLUMNS})


def mark_remaining_skipped(rows: list[dict], completed_run_index: int) -> None:
    """Mark not-yet-started rows as skipped after a stop-on-error failure."""

    for row in rows[completed_run_index:]:
        if row["status"] == "pending":
            row["status"] = "skipped"


def parse_bool(value: str) -> bool:
    """Parse a user-facing boolean value."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def global_config_path(dataset: str) -> Path:
    """Return the shared global config for a dataset."""

    return GLOBAL_CONFIGS[dataset]


def make_suite_id(suite: str, dataset: str = "mnist") -> str:
    """Create a timestamped suite id."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{dataset}_{suite}"
    return safe_token(f"{prefix}_{timestamp}")


def safe_token(value: object) -> str:
    """Keep generated folder names portable and readable."""

    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in text)


def format_float(value: float) -> str:
    """Format a numeric CLI value compactly."""

    return f"{value:g}"


def format_path_float(value: float) -> str:
    """Format a numeric value for folder names."""

    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text.replace(".", "p")


def format_command(command: list[str]) -> str:
    """Format a subprocess command for clear terminal output."""

    return subprocess.list2cmdline(command)


if __name__ == "__main__":
    raise SystemExit(main())
