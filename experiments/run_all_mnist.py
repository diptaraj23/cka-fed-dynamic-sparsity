"""Run the main MNIST federated learning baselines sequentially.

The script writes each run into a temporary output directory first, then moves
the generated CSV logs into ``results/logs`` with a unique run identifier. This
keeps repeated experiment launches from overwriting prior results.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_LOG_DIR = REPO_ROOT / "results" / "logs"
TEMP_ROOT = REPO_ROOT / "results" / "_run_all_tmp"

METHODS = ("fedavg", "sparse_fedavg", "feddst", "cka_feddst")
DATASET = "mnist"

COMMON_ARGS = {
    "num_clients": 5,
    "rounds": 20,
    "local_epochs": 1,
    "alpha": 0.3,
    "sparsity": 0.8,
    "seed": 42,
    "batch_size": 512,
    "reference_size": 100,
    "device": "cpu",
}


def main() -> int:
    """Run all configured MNIST methods."""

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_id}_{uuid.uuid4().hex[:8]}"
    RESULTS_LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Run id: {run_id}")
    print(f"Dataset: {DATASET}")
    for method in METHODS:
        temp_output_dir = TEMP_ROOT / run_id / method
        command = build_command(method, temp_output_dir)
        print("\nRunning:", format_command(command), flush=True)

        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print("\nFAILED COMMAND:", format_command(command), file=sys.stderr)
            print(f"Exit code: {exc.returncode}", file=sys.stderr)
            print(f"Temporary output kept at: {temp_output_dir}", file=sys.stderr)
            return exc.returncode

        try:
            moved_logs = move_logs(temp_output_dir / "logs", run_id)
        except Exception as exc:
            print("\nFAILED COMMAND:", format_command(command), file=sys.stderr)
            print(f"Log handling error: {exc}", file=sys.stderr)
            print(f"Temporary output kept at: {temp_output_dir}", file=sys.stderr)
            return 1

        for path in moved_logs:
            print(f"Saved unique log: {path.relative_to(REPO_ROOT)}")

    cleanup_run_temp(run_id)
    print("\nAll MNIST runs completed.")
    return 0


def build_command(method: str, output_dir: Path) -> list[str]:
    """Build the training command for one method."""

    command = [
        sys.executable,
        "-m",
        "src.train",
        "--method",
        method,
        "--num-clients",
        str(COMMON_ARGS["num_clients"]),
        "--rounds",
        str(COMMON_ARGS["rounds"]),
        "--local-epochs",
        str(COMMON_ARGS["local_epochs"]),
        "--alpha",
        str(COMMON_ARGS["alpha"]),
        "--sparsity",
        str(COMMON_ARGS["sparsity"]),
        "--seed",
        str(COMMON_ARGS["seed"]),
        "--batch-size",
        str(COMMON_ARGS["batch_size"]),
        "--reference-size",
        str(COMMON_ARGS["reference_size"]),
        "--device",
        str(COMMON_ARGS["device"]),
        "--output-dir",
        str(output_dir),
    ]

    if method in {"feddst", "cka_feddst"}:
        command.extend(["--mask-update-interval", "1", "--prune-fraction", "0.1"])
    if method == "cka_feddst":
        command.extend(["--cka-interval", "5", "--cka-target-strength", "0.5"])

    return command


def move_logs(source_log_dir: Path, run_id: str) -> list[Path]:
    """Move CSV logs into results/logs with unique names."""

    if not source_log_dir.exists():
        raise FileNotFoundError(f"No log directory found at {source_log_dir}")

    moved = []
    source_paths = sorted(source_log_dir.glob("*.csv"))
    if not source_paths:
        raise FileNotFoundError(f"No CSV logs found in {source_log_dir}")

    for source_path in source_paths:
        destination = unique_log_path(source_path, run_id)
        shutil.move(str(source_path), destination)
        moved.append(destination)
    return moved


def unique_log_path(source_path: Path, run_id: str) -> Path:
    """Return a destination path that will not overwrite an existing log."""

    candidate = RESULTS_LOG_DIR / f"{source_path.stem}_run{run_id}{source_path.suffix}"
    if not candidate.exists():
        return candidate

    for index in range(1, 10_000):
        indexed = RESULTS_LOG_DIR / (
            f"{source_path.stem}_run{run_id}_{index:04d}{source_path.suffix}"
        )
        if not indexed.exists():
            return indexed

    raise RuntimeError(f"Could not create a unique log name for {source_path.name}")


def cleanup_run_temp(run_id: str) -> None:
    """Remove the temporary directory for a successful run."""

    run_temp = TEMP_ROOT / run_id
    if run_temp.exists():
        shutil.rmtree(run_temp)


def format_command(command: list[str]) -> str:
    """Format a command for clear terminal output on Windows."""

    return subprocess.list2cmdline(command)


if __name__ == "__main__":
    raise SystemExit(main())
