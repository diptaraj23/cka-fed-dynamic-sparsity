"""Plotting helpers for experiment CSV logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ("fedavg", "sparse_fedavg", "feddst", "cka_feddst")
SPARSE_METHODS = ("sparse_fedavg", "feddst", "cka_feddst")
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "sparse_fedavg": "Sparse FedAvg",
    "feddst": "FedDST",
    "cka_feddst": "CKA-FedDST",
}
COMMUNICATION_COLUMNS = (
    "communication_cost",
    "active_params_transmitted",
    "active_params",
)
SPARSITY_PATTERN = re.compile(r"sparsity(?P<value>\d+(?:[p.]\d+)?)")


def generate_all_plots(
    log_dir: Path = Path("results/logs"),
    plot_dir: Path = Path("results/plots"),
) -> list[Path]:
    """Create sparsity-sweep plots from all available training logs."""

    logs = load_training_logs(log_dir)
    if logs.empty:
        print(f"Warning: no training CSV logs found in {log_dir}.")
        return []

    latest_logs = latest_by_method_sparsity(logs)
    pairwise_cka = load_pairwise_cka_logs(log_dir)

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    plotters = [
        plot_accuracy_vs_rounds,
        plot_final_accuracy_vs_sparsity,
        plot_best_accuracy_vs_sparsity,
        plot_accuracy_vs_communication_cost,
        plot_cka_feddst_layerwise_sparsity,
    ]

    for plotter in plotters:
        path = plotter(latest_logs, plot_dir)
        if path is not None:
            saved_paths.append(path)

    cka_path = plot_cka_feddst_layerwise_cka(latest_logs, pairwise_cka, plot_dir)
    if cka_path is not None:
        saved_paths.append(cka_path)

    return saved_paths


def load_training_logs(log_dir: Path) -> pd.DataFrame:
    """Read all non-CKA-matrix training CSV logs."""

    frames = []
    for path in sorted(log_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"Warning: could not read {path}: {exc}")
            continue

        if frame.empty:
            print(f"Warning: skipping empty log {path}.")
            continue
        if is_pairwise_cka_frame(frame):
            continue
        if not {"round", "test_accuracy"}.issubset(frame.columns):
            print(f"Warning: skipping {path.name}; missing round/test_accuracy.")
            continue

        method = infer_method(frame, path)
        if method is None:
            print(f"Warning: skipping {path.name}; could not infer method.")
            continue

        frame = frame.copy()
        frame["method"] = method
        frame["dataset"] = infer_dataset(frame, path)
        frame["sparsity"] = infer_sparsity(frame, path, method)
        frame["source_file"] = path.name
        frame["source_mtime"] = path.stat().st_mtime
        numeric_columns = [
            "round",
            "test_accuracy",
            "test_loss",
            "sparsity",
            "seed",
            "total_sparsity",
            "active_params",
            "communication_cost",
            "active_params_transmitted",
        ]
        frame = coerce_numeric_columns(frame, numeric_columns)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_pairwise_cka_logs(log_dir: Path) -> pd.DataFrame:
    """Read pairwise CKA matrix logs when available."""

    frames = []
    for path in sorted(log_dir.glob("*_cka.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"Warning: could not read CKA log {path}: {exc}")
            continue

        if frame.empty or not is_pairwise_cka_frame(frame):
            continue

        method = infer_method(frame, path)
        if method is None:
            continue

        frame = frame.copy()
        frame["method"] = method
        frame["sparsity"] = infer_sparsity(frame, path, method)
        frame["source_file"] = path.name
        frame["source_mtime"] = path.stat().st_mtime
        frame = coerce_numeric_columns(
            frame,
            ["round", "cka", "average_layer_cka", "sparsity"],
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def latest_by_method_sparsity(logs: pd.DataFrame) -> pd.DataFrame:
    """Keep the newest source file for each method/sparsity combination."""

    if logs.empty:
        return logs

    sources = logs[
        ["method", "sparsity", "source_file", "source_mtime"]
    ].drop_duplicates()
    sources = sources.sort_values("source_mtime")
    latest_sources = sources.drop_duplicates(
        subset=["method", "sparsity"],
        keep="last",
    )["source_file"]
    return logs[logs["source_file"].isin(set(latest_sources))].copy()


def plot_accuracy_vs_rounds(logs: pd.DataFrame, plot_dir: Path) -> Path | None:
    """Plot test accuracy over rounds for each method/sparsity combination."""

    if logs.empty or "test_accuracy" not in logs.columns:
        print("Warning: cannot plot accuracy vs rounds; no accuracy data.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, group in sorted_sources(logs):
        group = group.sort_values("round")
        label = method_sparsity_label(group)
        ax.plot(group["round"], group["test_accuracy"], marker="o", label=label)

    style_axes(
        ax,
        "Test Accuracy vs Communication Rounds",
        "Communication Round",
        "Test Accuracy",
    )
    return save_figure(fig, plot_dir / "accuracy_vs_rounds_all_sparsities.png")


def plot_final_accuracy_vs_sparsity(logs: pd.DataFrame, plot_dir: Path) -> Path | None:
    """Plot final test accuracy against sparsity for sparse methods."""

    sparse_logs = sparse_only(logs)
    if sparse_logs.empty:
        print("Warning: cannot plot final accuracy vs sparsity; no sparse logs.")
        return None

    finals = final_rows(sparse_logs)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for method in SPARSE_METHODS:
        method_rows = finals[finals["method"] == method].sort_values("sparsity")
        if method_rows.empty:
            continue
        ax.plot(
            method_rows["sparsity"],
            method_rows["test_accuracy"],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS[method],
        )
        plotted = True

    if not plotted:
        print("Warning: no sparse method data found for final accuracy plot.")
        plt.close(fig)
        return None

    style_axes(
        ax,
        "Final Accuracy vs Sparsity",
        "Sparsity",
        "Final Test Accuracy",
    )
    return save_figure(fig, plot_dir / "final_accuracy_vs_sparsity.png")


def plot_best_accuracy_vs_sparsity(logs: pd.DataFrame, plot_dir: Path) -> Path | None:
    """Plot best achieved test accuracy against sparsity for sparse methods."""

    sparse_logs = sparse_only(logs)
    if sparse_logs.empty:
        print("Warning: cannot plot best accuracy vs sparsity; no sparse logs.")
        return None

    best = best_rows(sparse_logs)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for method in SPARSE_METHODS:
        method_rows = best[best["method"] == method].sort_values("sparsity")
        if method_rows.empty:
            continue
        ax.plot(
            method_rows["sparsity"],
            method_rows["test_accuracy"],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS[method],
        )
        plotted = True

    if not plotted:
        print("Warning: no sparse method data found for best accuracy plot.")
        plt.close(fig)
        return None

    style_axes(
        ax,
        "Best Accuracy vs Sparsity",
        "Sparsity",
        "Best Test Accuracy",
    )
    return save_figure(fig, plot_dir / "best_accuracy_vs_sparsity.png")


def plot_accuracy_vs_communication_cost(
    logs: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """Plot accuracy against communication-cost proxy when available."""

    x_column = next(
        (column for column in COMMUNICATION_COLUMNS if column in logs.columns),
        None,
    )
    if x_column is None:
        print("Warning: communication cost is not logged; skipping cost plot.")
        return None

    cost_logs = logs.dropna(subset=[x_column, "test_accuracy"])
    if cost_logs.empty:
        print("Warning: communication cost values are empty; skipping cost plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, group in sorted_sources(cost_logs):
        group = group.sort_values("round")
        label = method_sparsity_label(group)
        ax.plot(group[x_column], group["test_accuracy"], marker="o", label=label)

    xlabel = {
        "communication_cost": "Communication Cost",
        "active_params_transmitted": "Active Parameters Transmitted",
        "active_params": "Active Parameters",
    }[x_column]
    style_axes(ax, "Accuracy vs Communication Cost", xlabel, "Test Accuracy")
    return save_figure(fig, plot_dir / "accuracy_vs_communication_cost.png")


def plot_cka_feddst_layerwise_sparsity(
    logs: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """Plot CKA-FedDST layer-wise sparsity over rounds."""

    cka_logs = logs[logs["method"] == "cka_feddst"]
    columns = layer_sparsity_columns(cka_logs)
    if cka_logs.empty or not columns:
        print("Warning: no CKA-FedDST layer-wise sparsity columns found.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, group in sorted_sources(cka_logs):
        sparsity = format_sparsity(group["sparsity"].iloc[0])
        group = group.sort_values("round")
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().any():
                label = f"s={sparsity} {pretty_layer(column)}"
                ax.plot(group["round"], values, marker="o", label=label)

    style_axes(
        ax,
        "CKA-FedDST Layer-wise Sparsity",
        "Communication Round",
        "Layer Sparsity",
    )
    return save_figure(fig, plot_dir / "cka_feddst_layerwise_sparsity.png")


def plot_cka_feddst_layerwise_cka(
    logs: pd.DataFrame,
    pairwise_cka: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """Plot CKA-FedDST layer-wise CKA over rounds."""

    cka_logs = logs[logs["method"] == "cka_feddst"]
    columns = cka_value_columns(cka_logs)
    if not cka_logs.empty and columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for _, group in sorted_sources(cka_logs):
            sparsity = format_sparsity(group["sparsity"].iloc[0])
            group = group.sort_values("round")
            for column in columns:
                values = pd.to_numeric(group[column], errors="coerce")
                if values.notna().any():
                    label = f"s={sparsity} {pretty_layer(column)}"
                    ax.plot(group["round"], values, marker="o", label=label)

        style_axes(
            ax,
            "CKA-FedDST Layer-wise CKA",
            "Communication Round",
            "Average Pairwise CKA",
        )
        return save_figure(fig, plot_dir / "cka_feddst_layerwise_cka.png")

    pairwise = pairwise_cka[pairwise_cka["method"] == "cka_feddst"]
    if pairwise.empty or "average_layer_cka" not in pairwise.columns:
        print("Warning: no CKA-FedDST CKA columns or pairwise CKA logs found.")
        return None

    pairwise = latest_by_method_sparsity(pairwise)
    pairwise = pairwise.dropna(subset=["round", "average_layer_cka"])
    if pairwise.empty:
        print("Warning: CKA pairwise logs contain no plottable CKA values.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped = pairwise.groupby(["source_file", "layer"], sort=False)
    for (_, layer), group in grouped:
        sparsity = format_sparsity(group["sparsity"].iloc[0])
        group = group.sort_values("round")
        by_round = group.groupby("round", as_index=False)["average_layer_cka"].mean()
        ax.plot(
            by_round["round"],
            by_round["average_layer_cka"],
            marker="o",
            label=f"s={sparsity} {layer}",
        )

    style_axes(
        ax,
        "CKA-FedDST Layer-wise CKA",
        "Communication Round",
        "Average Pairwise CKA",
    )
    return save_figure(fig, plot_dir / "cka_feddst_layerwise_cka.png")


def sparse_only(logs: pd.DataFrame) -> pd.DataFrame:
    """Return logs for sparse methods with numeric sparsity values."""

    if logs.empty:
        return logs
    sparse_logs = logs[logs["method"].isin(SPARSE_METHODS)].copy()
    return sparse_logs.dropna(subset=["sparsity"])


def sorted_sources(logs: pd.DataFrame):
    """Yield source-file groups in method/sparsity order."""

    source_rows = logs[
        ["method", "sparsity", "source_file"]
    ].drop_duplicates()
    source_rows["method_order"] = source_rows["method"].map(method_rank)
    source_rows = source_rows.sort_values(["method_order", "sparsity", "source_file"])
    for _, row in source_rows.iterrows():
        yield row["source_file"], logs[logs["source_file"] == row["source_file"]]


def final_rows(logs: pd.DataFrame) -> pd.DataFrame:
    """Return the last round from each source log."""

    if logs.empty:
        return logs
    return (
        logs.sort_values(["source_file", "round"])
        .groupby("source_file", as_index=False)
        .tail(1)
    )


def best_rows(logs: pd.DataFrame) -> pd.DataFrame:
    """Return the best test-accuracy row from each source log."""

    if logs.empty:
        return logs
    idx = logs.groupby("source_file")["test_accuracy"].idxmax()
    return logs.loc[idx].copy()


def infer_method(frame: pd.DataFrame, path: Path) -> str | None:
    """Infer method from a CSV column first, then from the filename."""

    if "method" in frame.columns:
        values = frame["method"].dropna()
        if not values.empty:
            value = str(values.iloc[0])
            if value in METHOD_ORDER:
                return value

    name = path.name
    for method in ("sparse_fedavg", "cka_feddst", "feddst", "fedavg"):
        if name.startswith(method):
            return method
    return None


def infer_dataset(frame: pd.DataFrame, path: Path) -> str:
    """Infer dataset from a CSV column first, then from the filename."""

    if "dataset" in frame.columns:
        values = frame["dataset"].dropna()
        if not values.empty:
            return str(values.iloc[0])
    if "_mnist_" in path.name:
        return "mnist"
    return "unknown"


def infer_sparsity(frame: pd.DataFrame, path: Path, method: str) -> float:
    """Infer sparsity from a CSV column first, then from the filename."""

    if method == "fedavg":
        return 0.0

    if "sparsity" in frame.columns:
        values = pd.to_numeric(frame["sparsity"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])

    match = SPARSITY_PATTERN.search(path.name)
    if match:
        return float(match.group("value").replace("p", "."))

    return float("nan")


def is_pairwise_cka_frame(frame: pd.DataFrame) -> bool:
    """Return true for pairwise CKA matrix CSVs."""

    return {"client_i", "client_j", "cka"}.issubset(frame.columns)


def coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric when present."""

    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def method_sparsity_label(group: pd.DataFrame) -> str:
    """Return a legend label for a source-file group."""

    method = str(group["method"].iloc[0])
    label = METHOD_LABELS.get(method, method)
    if method == "fedavg":
        return f"{label} dense"
    return f"{label} s={format_sparsity(group['sparsity'].iloc[0])}"


def format_sparsity(value) -> str:
    """Format sparsity values for labels."""

    if pd.isna(value):
        return "unknown"
    return f"{float(value):g}"


def method_rank(method: str) -> int:
    """Return a stable order for methods."""

    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def layer_sparsity_columns(frame: pd.DataFrame) -> list[str]:
    """Return layer-wise sparsity columns."""

    return [
        column
        for column in frame.columns
        if column.startswith("sparsity_") and column != "total_sparsity"
    ]


def cka_value_columns(frame: pd.DataFrame) -> list[str]:
    """Return layer-wise average CKA columns from main logs."""

    return [
        column
        for column in frame.columns
        if column.startswith("cka_") and column != "cka_computed"
    ]


def pretty_layer(column: str) -> str:
    """Convert a metric column name into a compact legend label."""

    for prefix in ("sparsity_", "cka_avg_", "cka_", "target_sparsity_"):
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
        ax.legend(fontsize=8)


def save_figure(fig, output_path: Path) -> Path:
    """Save a matplotlib figure and close it."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_training_curve(history, output_path: Path) -> Path:
    """Save a simple training curve from in-memory metric dictionaries."""

    rows = list(history)
    if not rows:
        raise ValueError("history must contain at least one row.")

    frame = pd.DataFrame(rows)
    if "method" not in frame.columns:
        frame["method"] = "fedavg"
    if "sparsity" not in frame.columns:
        frame["sparsity"] = 0.0
    if "source_file" not in frame.columns:
        frame["source_file"] = output_path.name
    return plot_accuracy_vs_rounds(frame, output_path.parent) or output_path


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for plotting."""

    parser = argparse.ArgumentParser(description="Plot experiment CSV logs.")
    parser.add_argument(
        "--log_dir",
        "--log-dir",
        dest="log_dir",
        type=Path,
        default=Path("results/logs"),
    )
    parser.add_argument(
        "--plot_dir",
        "--plot-dir",
        "--output-dir",
        dest="plot_dir",
        type=Path,
        default=Path("results/plots"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for result plotting."""

    args = build_parser().parse_args(argv)
    saved_paths = generate_all_plots(args.log_dir, args.plot_dir)
    for path in saved_paths:
        print(f"Saved plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
