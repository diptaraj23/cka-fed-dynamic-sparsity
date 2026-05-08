"""Plotting helpers for experiment CSV logs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = ("fedavg", "sparse_fedavg", "feddst", "cka_feddst")
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "sparse_fedavg": "Sparse FedAvg",
    "feddst": "FedDST",
    "cka_feddst": "CKA-FedDST",
}


@dataclass
class ExperimentLog:
    """Parsed experiment log with numeric CSV values."""

    method: str
    path: Path
    rows: list[dict[str, float | str | None]]
    fieldnames: list[str]


def generate_all_plots(
    log_dir: Path = Path("results/logs"),
    output_dir: Path = Path("results/plots"),
) -> list[Path]:
    """Create the standard result plots from CSV logs."""

    logs = load_latest_method_logs(log_dir)
    if not logs:
        raise FileNotFoundError(f"No training CSV logs found in {log_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [
        plot_metric(
            logs,
            metric="test_accuracy",
            output_path=output_dir / "test_accuracy_vs_rounds.png",
            title="Test Accuracy vs Communication Rounds",
            ylabel="Test Accuracy",
        ),
        plot_metric(
            logs,
            metric="test_loss",
            output_path=output_dir / "test_loss_vs_rounds.png",
            title="Test Loss vs Communication Rounds",
            ylabel="Test Loss",
        ),
        plot_metric(
            [log for log in logs if "total_sparsity" in log.fieldnames],
            metric="total_sparsity",
            output_path=output_dir / "total_sparsity_vs_rounds.png",
            title="Total Sparsity vs Communication Rounds",
            ylabel="Total Sparsity",
        ),
        plot_final_accuracy_bar(
            logs,
            output_path=output_dir / "final_accuracy_comparison.png",
        ),
    ]

    for log in logs:
        if sparsity_columns(log):
            saved_paths.append(
                plot_layer_columns(
                    log,
                    columns=sparsity_columns(log),
                    output_path=output_dir / f"layer_sparsity_{log.method}.png",
                    title=f"{METHOD_LABELS[log.method]} Layer-wise Sparsity",
                    ylabel="Layer Sparsity",
                )
            )

        if log.method == "cka_feddst" and cka_columns(log):
            saved_paths.append(
                plot_layer_columns(
                    log,
                    columns=cka_columns(log),
                    output_path=output_dir / "layer_cka_cka_feddst.png",
                    title="CKA-FedDST Layer-wise CKA",
                    ylabel="Average Pairwise CKA",
                )
            )

    return saved_paths


def load_latest_method_logs(log_dir: Path) -> list[ExperimentLog]:
    """Load the newest training log for each supported method."""

    candidates: dict[str, list[Path]] = {method: [] for method in METHOD_ORDER}
    for path in sorted(log_dir.glob("*.csv")):
        method = infer_method(path)
        if method is None or is_pairwise_cka_log(path):
            continue
        fieldnames = read_fieldnames(path)
        if "round" not in fieldnames or "test_accuracy" not in fieldnames:
            continue
        candidates[method].append(path)

    logs = []
    for method in METHOD_ORDER:
        paths = candidates[method]
        if not paths:
            continue
        latest = max(paths, key=lambda item: item.stat().st_mtime)
        logs.append(read_experiment_log(latest, method))
    return logs


def read_experiment_log(path: Path, method: str | None = None) -> ExperimentLog:
    """Read one CSV log and coerce numeric-looking values to floats."""

    if method is None:
        method = infer_method(path)
    if method is None:
        raise ValueError(f"Could not infer method from {path.name}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            {key: parse_value(value) for key, value in row.items()}
            for row in reader
        ]

    return ExperimentLog(method=method, path=path, rows=rows, fieldnames=fieldnames)


def plot_metric(
    logs: list[ExperimentLog],
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> Path:
    """Plot one round-wise metric for all available methods."""

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for log in logs:
        if metric not in log.fieldnames:
            continue
        rounds, values = series(log, metric)
        if not values:
            continue
        ax.plot(rounds, values, marker="o", linewidth=2, label=METHOD_LABELS[log.method])
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, f"No {metric} data found", ha="center", va="center")

    style_axes(ax, title, "Communication Round", ylabel)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_layer_columns(
    log: ExperimentLog,
    columns: list[str],
    output_path: Path,
    title: str,
    ylabel: str,
) -> Path:
    """Plot layer-wise columns from one method log."""

    fig, ax = plt.subplots(figsize=(8, 5))
    for column in columns:
        rounds, values = series(log, column)
        if values:
            ax.plot(rounds, values, marker="o", linewidth=2, label=pretty_layer(column))

    style_axes(ax, title, "Communication Round", ylabel)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_final_accuracy_bar(logs: list[ExperimentLog], output_path: Path) -> Path:
    """Plot final test accuracy for each method."""

    labels = []
    values = []
    for log in logs:
        final_value = final_numeric_value(log, "test_accuracy")
        if final_value is not None:
            labels.append(METHOD_LABELS[log.method])
            values.append(final_value)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1"])
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, max(values + [1.0]) * 1.08)
    style_axes(ax, "Final Test Accuracy Comparison", "Method", "Final Test Accuracy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_training_curve(history, output_path: Path):
    """Save a simple training curve from in-memory metric dictionaries."""

    rows = list(history)
    if not rows:
        raise ValueError("history must contain at least one row.")
    log = ExperimentLog(
        method="fedavg",
        path=output_path,
        rows=rows,
        fieldnames=list(rows[0]),
    )
    return plot_metric(
        [log],
        metric="test_accuracy",
        output_path=output_path,
        title="Training Curve",
        ylabel="Test Accuracy",
    )


def series(log: ExperimentLog, column: str) -> tuple[list[float], list[float]]:
    """Return numeric round/value pairs for a column."""

    rounds = []
    values = []
    for row in log.rows:
        round_value = row.get("round")
        value = row.get(column)
        if isinstance(round_value, (int, float)) and isinstance(value, (int, float)):
            rounds.append(round_value)
            values.append(value)
    return rounds, values


def final_numeric_value(log: ExperimentLog, column: str) -> float | None:
    """Return the last numeric value in a column."""

    for row in reversed(log.rows):
        value = row.get(column)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def infer_method(path: Path) -> str | None:
    """Infer experiment method from a CSV filename."""

    name = path.name
    for method in ("sparse_fedavg", "cka_feddst", "feddst", "fedavg"):
        if name.startswith(method):
            return method
    return None


def is_pairwise_cka_log(path: Path) -> bool:
    """Return true for pairwise CKA matrix CSVs."""

    fields = read_fieldnames(path)
    return {"client_i", "client_j", "cka"}.issubset(fields)


def read_fieldnames(path: Path) -> list[str]:
    """Read only the header row from a CSV file."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def sparsity_columns(log: ExperimentLog) -> list[str]:
    """Return layer-wise sparsity columns."""

    return [
        column
        for column in log.fieldnames
        if column.startswith("sparsity_") and column != "total_sparsity"
    ]


def cka_columns(log: ExperimentLog) -> list[str]:
    """Return layer-wise CKA columns from a main training log."""

    return [
        column
        for column in log.fieldnames
        if column.startswith("cka_") and column != "cka_computed"
    ]


def parse_value(value: str | None) -> float | str | None:
    """Convert CSV strings to floats when possible."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def pretty_layer(column: str) -> str:
    """Convert a metric column name into a compact legend label."""

    for prefix in ("sparsity_", "cka_", "target_sparsity_"):
        if column.startswith(prefix):
            column = column.removeprefix(prefix)
            break
    return column.replace("_weight", "").replace("_", ".")


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent labels, legend, and grid."""

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for plotting."""

    parser = argparse.ArgumentParser(description="Plot experiment CSV logs.")
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for result plotting."""

    args = build_parser().parse_args(argv)
    saved_paths = generate_all_plots(args.log_dir, args.output_dir)
    for path in saved_paths:
        print(f"Saved plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
