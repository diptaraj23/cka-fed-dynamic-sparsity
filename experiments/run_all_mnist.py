"""Legacy helper to run one MNIST run per method.

The main reproducible suite runner is ``experiments/run_experiment.py``. This
script is kept only as a small convenience entry point for quick local checks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "fedavg": REPO_ROOT / "configs" / "fedavg_mnist.yaml",
    "sparse_fedavg": REPO_ROOT / "configs" / "sparse_fedavg_mnist.yaml",
    "feddst": REPO_ROOT / "configs" / "feddst_mnist.yaml",
    "cka_feddst": REPO_ROOT / "configs" / "cka_feddst_mnist.yaml",
}


def main() -> int:
    """Run the four configured baselines one after another."""

    for method, config_path in CONFIGS.items():
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            str(config_path),
        ]
        print("\nRunning:", format_command(command), flush=True)

        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print("\nFAILED COMMAND:", format_command(command), file=sys.stderr)
            print(f"Exit code: {exc.returncode}", file=sys.stderr)
            return exc.returncode

    print("\nAll MNIST runs completed.")
    return 0


def format_command(command: list[str]) -> str:
    """Format a subprocess command for clear terminal output."""

    return subprocess.list2cmdline(command)


if __name__ == "__main__":
    raise SystemExit(main())
