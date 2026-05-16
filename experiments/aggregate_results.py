"""Aggregate raw experiment logs into averaged CSV files."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.aggregation import aggregate_results


SUITE_FOLDERS = {
    "multiseed": "multiseed",
    "cka_strength": "cka_strength_sweep",
}


def build_parser() -> argparse.ArgumentParser:
    """Create the aggregation CLI."""

    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed experiment logs into mean/std CSVs."
    )
    parser.add_argument(
        "--suite",
        choices=("multiseed", "cka_strength", "all"),
        default="all",
        help=(
            "Experiment suite to aggregate. Defaults to all, which discovers "
            "both multiseed and CKA-strength suite folders."
        ),
    )
    parser.add_argument(
        "--log_dir",
        "--log-dir",
        dest="log_dir",
        type=Path,
        default=None,
        help=(
            "Raw suite log directory, e.g. results/logs/multiseed/<suite_id>. "
            "If omitted, all suite folders are discovered automatically."
        ),
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help=(
            "Directory where averaged CSV files will be saved. If omitted, "
            "the path is derived under results/averaged/."
        ),
    )
    parser.add_argument(
        "--logs_root",
        "--logs-root",
        dest="logs_root",
        type=Path,
        default=Path("results/logs"),
        help="Root directory containing suite log folders.",
    )
    parser.add_argument(
        "--averaged_root",
        "--averaged-root",
        dest="averaged_root",
        type=Path,
        default=Path("results/averaged"),
        help="Root directory where averaged suite folders are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Aggregate logs and print saved output paths."""

    args = build_parser().parse_args(argv)
    tasks = build_aggregation_tasks(args)
    if not tasks:
        print(
            "Warning: no suite log folders found. Run experiments first, or pass "
            "--log_dir and --suite explicitly."
        )
        return 1

    for index, (suite, log_dir, output_dir) in enumerate(tasks, start=1):
        print(
            f"[Aggregate {index}/{len(tasks)}] "
            f"suite={suite} log_dir={log_dir} output_dir={output_dir}"
        )
        saved_paths = aggregate_results(log_dir, output_dir, suite)
        for path in saved_paths:
            print(f"Saved averaged result: {path}")
    return 0


def build_aggregation_tasks(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    """Return the suite/log/output combinations requested by the CLI."""

    if args.log_dir is not None:
        suite = args.suite
        if suite == "all":
            suite = infer_suite_from_path(args.log_dir)
            if suite is None:
                raise SystemExit(
                    "Could not infer suite from --log_dir. Pass "
                    "--suite multiseed or --suite cka_strength."
                )
        output_dir = args.output_dir or default_output_dir(
            args.log_dir,
            suite,
            args.averaged_root,
        )
        return [(suite, args.log_dir, output_dir)]

    if args.output_dir is not None:
        raise SystemExit("--output_dir requires --log_dir for a targeted aggregation.")

    tasks: list[tuple[str, Path, Path]] = []
    suites = tuple(SUITE_FOLDERS) if args.suite == "all" else (args.suite,)
    for suite in suites:
        for log_dir in discover_suite_log_dirs(args.logs_root, suite):
            tasks.append(
                (
                    suite,
                    log_dir,
                    default_output_dir(log_dir, suite, args.averaged_root),
                )
            )
    return tasks


def discover_suite_log_dirs(logs_root: Path, suite: str) -> list[Path]:
    """Find timestamped/raw suite folders for one experiment suite."""

    suite_root = Path(logs_root) / suite_folder(suite)
    if not suite_root.exists():
        return []
    return sorted(
        path
        for path in suite_root.iterdir()
        if path.is_dir() and has_candidate_training_logs(path)
    )


def has_candidate_training_logs(path: Path) -> bool:
    """Return true when a suite folder appears to contain raw training logs."""

    for csv_path in path.rglob("*.csv"):
        if csv_path.name == "manifest.csv" or csv_path.name.endswith("_cka.csv"):
            continue
        return True
    return False


def default_output_dir(log_dir: Path, suite: str, averaged_root: Path) -> Path:
    """Map a raw log suite folder to its averaged output folder."""

    return Path(averaged_root) / suite_folder(suite) / Path(log_dir).name


def infer_suite_from_path(path: Path) -> str | None:
    """Infer suite name from a log path."""

    parts = set(Path(path).parts)
    for suite, folder in SUITE_FOLDERS.items():
        if folder in parts:
            return suite
    return None


def suite_folder(suite: str) -> str:
    """Return the directory name used for a suite."""

    return SUITE_FOLDERS[suite]


if __name__ == "__main__":
    raise SystemExit(main())
