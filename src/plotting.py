"""Plotting helpers for experiment CSV logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


METHOD_ORDER = ("fedavg", "sparse_fedavg", "feddst", "cka_feddst")
SPARSE_METHODS = ("sparse_fedavg", "feddst", "cka_feddst")
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "sparse_fedavg": "Sparse FedAvg",
    "feddst": "FedDST",
    "cka_feddst": "CKA-FedDST",
}
METHOD_COLORS = {
    "fedavg": "#333333",
    "sparse_fedavg": "#4C78A8",
    "feddst": "#F58518",
    "cka_feddst": "#54A24B",
}
METHOD_MARKERS = {
    "fedavg": "*",
    "sparse_fedavg": "o",
    "feddst": "s",
    "cka_feddst": "^",
}
SPARSITY_STYLES = {
    0.0: {"marker": "*", "linestyle": "-", "label": "dense"},
    0.5: {"marker": "o", "linestyle": "-", "label": "s=0.5"},
    0.7: {"marker": "s", "linestyle": "--", "label": "s=0.7"},
    0.8: {"marker": "^", "linestyle": "-.", "label": "s=0.8"},
    0.9: {"marker": "D", "linestyle": ":", "label": "s=0.9"},
    0.95: {"marker": "X", "linestyle": (0, (3, 1, 1, 1)), "label": "s=0.95"},
}
SPARSITY_COLORS = {
    0.0: "#333333",
    0.5: "#4C78A8",
    0.7: "#F58518",
    0.8: "#54A24B",
    0.9: "#B279A2",
    0.95: "#E45756",
}
LAYER_LINESTYLES = {
    "conv1": "-",
    "conv2": "--",
    "fc1": "-.",
    "fc2": ":",
}
COMMUNICATION_COLUMNS = (
    "communication_cost",
    "active_params_transmitted",
    "active_params",
)
SPARSITY_PATTERN = re.compile(r"sparsity(?P<value>\d+(?:[p.]\d+)?)")
SEED_PATTERN = re.compile(r"(?:^|_)seed(?P<value>\d+)(?:_|\.|$)")
CKA_STRENGTH_PATTERN = re.compile(r"(?:^|_)cka(?P<value>\d+(?:[p.]\d+)?)(?:_|\.|$)")
CKA_STRENGTH_DIR_PATTERN = re.compile(r"strength_(?P<value>\d+(?:[p.]\d+)?)")
CKA_STRENGTH_COLORS = {
    0.2: "#333333",
    0.5: "#4C78A8",
    0.8: "#F58518",
    0.9: "#54A24B",
    1.0: "#B279A2",
}
CKA_STRENGTH_MARKERS = {
    0.2: "o",
    0.5: "s",
    0.8: "^",
    0.9: "D",
    1.0: "X",
}


def generate_all_plots(
    log_dir: Path = Path("results/logs"),
    plot_dir: Path = Path("results/plots"),
) -> list[Path]:
    """Create sparsity-sweep plots from all available training logs."""

    logs = load_training_logs(log_dir)
    if logs.empty:
        print(f"Warning: no training CSV logs found in {log_dir}.")
        return []

    warn_if_mixed_suite_logs(logs, log_dir)
    latest_logs = latest_by_run_identity(logs)
    pairwise_cka = load_pairwise_cka_logs(log_dir)

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    plotters = [
        plot_accuracy_vs_rounds,
        plot_final_accuracy_vs_sparsity,
        plot_best_accuracy_vs_sparsity,
        plot_accuracy_vs_rounds_mean_std,
        plot_final_accuracy_mean_std_vs_sparsity,
        plot_best_accuracy_mean_std_vs_sparsity,
        plot_cka_strength_accuracy_vs_rounds,
        plot_cka_strength_final_accuracy_vs_sparsity,
        plot_cka_strength_best_accuracy_vs_sparsity,
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
    for path in sorted(log_dir.rglob("*.csv")):
        if path.name == "manifest.csv" or path.name.endswith("_cka.csv"):
            continue
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
        frame["seed"] = infer_seed(frame, path)
        frame["cka_strength"] = infer_cka_strength(frame, path, method)
        frame["source_file"] = path.name
        frame["source_path"] = str(path)
        frame["source_mtime"] = path.stat().st_mtime
        numeric_columns = [
            "round",
            "test_accuracy",
            "test_loss",
            "sparsity",
            "seed",
            "cka_strength",
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
    for path in sorted(log_dir.rglob("*_cka.csv")):
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
        frame["seed"] = infer_seed(frame, path)
        frame["cka_strength"] = infer_cka_strength(frame, path, method)
        frame["source_file"] = path.name
        frame["source_path"] = str(path)
        frame["source_mtime"] = path.stat().st_mtime
        frame = coerce_numeric_columns(
            frame,
            ["round", "cka", "average_layer_cka", "sparsity", "seed", "cka_strength"],
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def warn_if_mixed_suite_logs(logs: pd.DataFrame, log_dir: Path) -> None:
    """Warn when plotting a folder that mixes older flat logs and suite logs."""

    if logs.empty or "source_path" not in logs.columns:
        return

    log_dir = Path(log_dir)
    source_paths = [Path(path) for path in logs["source_path"].dropna().unique()]
    suite_markers = {"multiseed", "cka_strength_sweep"}
    found_suites = {
        part
        for path in source_paths
        for part in path.parts
        if part in suite_markers
    }
    has_flat_logs = any(path.parent == log_dir for path in source_paths)

    if len(found_suites) > 1 or (found_suites and has_flat_logs):
        print(
            "Warning: this log directory mixes multiple experiment suites and/or "
            "older flat logs. For paper figures, prefer plotting one suite folder "
            "at a time with --log_dir results/logs/<suite>/<suite_id>."
        )


def latest_by_method_sparsity(logs: pd.DataFrame) -> pd.DataFrame:
    """Keep the newest source file for each method/sparsity combination."""

    return latest_by_run_identity(logs)


def latest_by_run_identity(logs: pd.DataFrame) -> pd.DataFrame:
    """Keep newest logs without merging different seeds or CKA strengths."""

    if logs.empty:
        return logs

    source_col = source_column(logs)
    keys = [
        column
        for column in ("method", "sparsity", "seed", "cka_strength")
        if column in logs.columns
    ]
    sources = logs[keys + [source_col, "source_mtime"]].drop_duplicates()
    sources = sources.sort_values("source_mtime")
    latest_sources = sources.drop_duplicates(
        subset=keys,
        keep="last",
    )[source_col]
    return logs[logs[source_col].isin(set(latest_sources))].copy()


def plot_accuracy_vs_rounds(logs: pd.DataFrame, plot_dir: Path) -> Path | None:
    """Plot test accuracy over rounds for each method/sparsity combination."""

    if logs.empty or "test_accuracy" not in logs.columns:
        print("Warning: cannot plot accuracy vs rounds; no accuracy data.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for _, group in sorted_sources(logs):
        group = group.sort_values("round")
        plot_method_sparsity_line(ax, group, "round", "test_accuracy")

    style_axes(
        ax,
        "Test Accuracy vs Communication Rounds",
        "Communication Round",
        "Test Accuracy",
        show_legend=False,
    )
    add_method_sparsity_legends(ax, logs)
    return save_figure(fig, plot_dir / "accuracy_vs_rounds_all_sparsities.png")


def plot_final_accuracy_vs_sparsity(logs: pd.DataFrame, plot_dir: Path) -> Path | None:
    """Plot final test accuracy against sparsity for sparse methods."""

    sparse_logs = sparse_only(logs)
    if sparse_logs.empty:
        print("Warning: cannot plot final accuracy vs sparsity; no sparse logs.")
        return None

    finals = aggregate_metric(
        final_rows(sparse_logs),
        ["method", "sparsity", "cka_strength"],
        "test_accuracy",
    )
    if finals.empty:
        print("Warning: no sparse method data found for final accuracy plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    add_dense_mean_reference(ax, final_rows(logs), "FedAvg dense baseline")
    plotted = plot_accuracy_summary_by_sparsity(
        ax,
        finals,
        include_cka_strength=has_multiple_cka_strengths(finals),
    )

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
    ax.legend(title="Method", fontsize=8)
    return save_figure(fig, plot_dir / "final_accuracy_vs_sparsity.png")


def plot_best_accuracy_vs_sparsity(logs: pd.DataFrame, plot_dir: Path) -> Path | None:
    """Plot best achieved test accuracy against sparsity for sparse methods."""

    sparse_logs = sparse_only(logs)
    if sparse_logs.empty:
        print("Warning: cannot plot best accuracy vs sparsity; no sparse logs.")
        return None

    best = aggregate_metric(
        best_rows(sparse_logs),
        ["method", "sparsity", "cka_strength"],
        "test_accuracy",
    )
    if best.empty:
        print("Warning: no sparse method data found for best accuracy plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    add_dense_mean_reference(ax, best_rows(logs), "FedAvg dense best")
    plotted = plot_accuracy_summary_by_sparsity(
        ax,
        best,
        include_cka_strength=has_multiple_cka_strengths(best),
    )

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
    ax.legend(title="Method", fontsize=8)
    return save_figure(fig, plot_dir / "best_accuracy_vs_sparsity.png")


def plot_accuracy_vs_rounds_mean_std(
    logs: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """Plot mean accuracy with standard-deviation bands across seeds."""

    if logs.empty or "test_accuracy" not in logs.columns:
        return None

    rows = logs.dropna(subset=["round", "test_accuracy"]).copy()
    if rows.empty:
        return None

    group_cols = ["method", "sparsity", "cka_strength", "round"]
    aggregate = aggregate_metric(rows, group_cols, "test_accuracy")
    if aggregate.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6.5))
    include_cka = has_multiple_cka_strengths(aggregate)
    for (method, sparsity, cka_strength), group in aggregate.groupby(
        ["method", "sparsity", "cka_strength"],
        dropna=False,
        sort=False,
    ):
        group = group.sort_values("round")
        style = sparsity_style(sparsity)
        color = method_color(method)
        ax.plot(
            group["round"],
            group["mean"],
            color=color,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=5,
            label=aggregate_method_label(method, sparsity, cka_strength, include_cka),
        )
        if group["count"].max() > 1:
            mean = group["mean"].astype(float)
            std = group["std"].fillna(0.0).astype(float)
            ax.fill_between(
                group["round"].astype(float),
                mean - std,
                mean + std,
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    style_axes(
        ax,
        "Mean Test Accuracy vs Communication Rounds",
        "Communication Round",
        "Mean Test Accuracy",
    )
    return save_figure(fig, plot_dir / "accuracy_vs_rounds_mean_std.png")


def plot_final_accuracy_mean_std_vs_sparsity(
    logs: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """Plot final accuracy mean/std across seeds for sparse methods."""

    sparse_logs = sparse_only(logs)
    if sparse_logs.empty:
        return None

    finals = final_rows(sparse_logs)
    aggregate = aggregate_metric(
        finals,
        ["method", "sparsity", "cka_strength"],
        "test_accuracy",
    )
    if aggregate.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    add_dense_mean_reference(ax, final_rows(logs), "FedAvg dense mean")
    include_cka = has_multiple_cka_strengths(aggregate)
    plotted = plot_accuracy_summary_by_sparsity(ax, aggregate, include_cka)
    if not plotted:
        plt.close(fig)
        return None

    style_axes(
        ax,
        "Final Accuracy Mean/Std vs Sparsity",
        "Sparsity",
        "Final Test Accuracy",
    )
    ax.legend(title="Method", fontsize=8)
    return save_figure(fig, plot_dir / "final_accuracy_mean_std_vs_sparsity.png")


def plot_best_accuracy_mean_std_vs_sparsity(
    logs: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """Plot best accuracy mean/std across seeds for sparse methods."""

    sparse_logs = sparse_only(logs)
    if sparse_logs.empty:
        return None

    best = best_rows(sparse_logs)
    aggregate = aggregate_metric(
        best,
        ["method", "sparsity", "cka_strength"],
        "test_accuracy",
    )
    if aggregate.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    add_dense_mean_reference(ax, best_rows(logs), "FedAvg dense best mean")
    include_cka = has_multiple_cka_strengths(aggregate)
    plotted = plot_accuracy_summary_by_sparsity(ax, aggregate, include_cka)
    if not plotted:
        plt.close(fig)
        return None

    style_axes(
        ax,
        "Best Accuracy Mean/Std vs Sparsity",
        "Sparsity",
        "Best Test Accuracy",
    )
    ax.legend(title="Method", fontsize=8)
    return save_figure(fig, plot_dir / "best_accuracy_mean_std_vs_sparsity.png")


def plot_cka_strength_accuracy_vs_rounds(
    logs: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """Plot CKA-FedDST accuracy by CKA strength and sparsity."""

    cka_logs = cka_strength_logs(logs)
    if cka_logs.empty or len(unique_cka_strengths(cka_logs)) < 2:
        return None

    aggregate = aggregate_metric(
        cka_logs.dropna(subset=["round", "test_accuracy"]),
        ["cka_strength", "sparsity", "round"],
        "test_accuracy",
    )
    if aggregate.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for (strength, sparsity), group in aggregate.groupby(
        ["cka_strength", "sparsity"],
        sort=False,
    ):
        group = group.sort_values("round")
        style = sparsity_style(sparsity)
        color = cka_strength_color(strength)
        ax.plot(
            group["round"],
            group["mean"],
            color=color,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=5,
            label="_nolegend_",
        )
        if group["count"].max() > 1:
            mean = group["mean"].astype(float)
            std = group["std"].fillna(0.0).astype(float)
            ax.fill_between(
                group["round"].astype(float),
                mean - std,
                mean + std,
                color=color,
                alpha=0.10,
                linewidth=0,
            )

    style_axes(
        ax,
        "CKA-FedDST Accuracy by CKA Strength",
        "Communication Round",
        "Mean Test Accuracy",
        show_legend=False,
    )
    add_cka_strength_sparsity_legends(ax, cka_logs)
    return save_figure(fig, plot_dir / "cka_strength_accuracy_vs_rounds.png")


def plot_cka_strength_final_accuracy_vs_sparsity(
    logs: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """Plot final CKA-FedDST accuracy by CKA strength."""

    cka_logs = cka_strength_logs(logs)
    if cka_logs.empty or len(unique_cka_strengths(cka_logs)) < 2:
        return None
    return plot_cka_strength_summary(
        final_rows(cka_logs),
        plot_dir / "cka_strength_final_accuracy_vs_sparsity.png",
        "CKA Strength Final Accuracy vs Sparsity",
        "Final Test Accuracy",
    )


def plot_cka_strength_best_accuracy_vs_sparsity(
    logs: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """Plot best CKA-FedDST accuracy by CKA strength."""

    cka_logs = cka_strength_logs(logs)
    if cka_logs.empty or len(unique_cka_strengths(cka_logs)) < 2:
        return None
    return plot_cka_strength_summary(
        best_rows(cka_logs),
        plot_dir / "cka_strength_best_accuracy_vs_sparsity.png",
        "CKA Strength Best Accuracy vs Sparsity",
        "Best Test Accuracy",
    )


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

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for _, group in sorted_sources(cost_logs):
        group = group.sort_values("round")
        plot_method_sparsity_line(ax, group, x_column, "test_accuracy")

    xlabel = {
        "communication_cost": "Communication Cost",
        "active_params_transmitted": "Active Parameters Transmitted",
        "active_params": "Active Parameters",
    }[x_column]
    style_axes(
        ax,
        "Accuracy vs Communication Cost",
        xlabel,
        "Test Accuracy",
        show_legend=False,
    )
    add_method_sparsity_legends(ax, cost_logs)
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

    fig, ax = plt.subplots(figsize=(12, 6.5))
    plotted_layers: set[str] = set()
    for _, group in sorted_sources(cka_logs):
        sparsity = group["sparsity"].iloc[0]
        style = sparsity_style(sparsity)
        group = group.sort_values("round")
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().any():
                layer = pretty_layer(column)
                plotted_layers.add(layer)
                ax.plot(
                    group["round"],
                    values,
                    color=sparsity_color(sparsity),
                    marker=style["marker"],
                    linestyle=layer_linestyle(layer),
                    linewidth=1.8,
                    markersize=5,
                    label="_nolegend_",
                )

    style_axes(
        ax,
        "CKA-FedDST Layer-wise Sparsity",
        "Communication Round",
        "Layer Sparsity",
        show_legend=False,
    )
    add_sparsity_layer_legends(ax, cka_logs, sorted(plotted_layers))
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
        fig, ax = plt.subplots(figsize=(12, 6.5))
        plotted_layers: set[str] = set()
        for _, group in sorted_sources(cka_logs):
            sparsity = group["sparsity"].iloc[0]
            style = sparsity_style(sparsity)
            group = group.sort_values("round")
            for column in columns:
                values = pd.to_numeric(group[column], errors="coerce")
                if values.notna().any():
                    layer = pretty_layer(column)
                    plotted_layers.add(layer)
                    ax.plot(
                        group["round"],
                        values,
                        color=sparsity_color(sparsity),
                        marker=style["marker"],
                        linestyle=layer_linestyle(layer),
                        linewidth=1.8,
                        markersize=5,
                        label="_nolegend_",
                    )

        style_axes(
            ax,
            "CKA-FedDST Layer-wise CKA",
            "Communication Round",
            "Average Pairwise CKA",
            show_legend=False,
        )
        add_sparsity_layer_legends(ax, cka_logs, sorted(plotted_layers))
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

    fig, ax = plt.subplots(figsize=(12, 6.5))
    grouped = pairwise.groupby(["source_file", "layer"], sort=False)
    plotted_layers: set[str] = set()
    for (_, layer), group in grouped:
        sparsity = group["sparsity"].iloc[0]
        style = sparsity_style(sparsity)
        plotted_layers.add(str(layer))
        group = group.sort_values("round")
        by_round = group.groupby("round", as_index=False)["average_layer_cka"].mean()
        ax.plot(
            by_round["round"],
            by_round["average_layer_cka"],
            color=sparsity_color(sparsity),
            marker=style["marker"],
            linestyle=layer_linestyle(str(layer)),
            linewidth=1.8,
            markersize=5,
            label="_nolegend_",
        )

    style_axes(
        ax,
        "CKA-FedDST Layer-wise CKA",
        "Communication Round",
        "Average Pairwise CKA",
        show_legend=False,
    )
    add_sparsity_layer_legends(ax, pairwise, sorted(plotted_layers))
    return save_figure(fig, plot_dir / "cka_feddst_layerwise_cka.png")


def aggregate_metric(
    rows: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    """Aggregate a metric by mean/std/count while preserving NaN groups."""

    if rows.empty or value_col not in rows.columns:
        return pd.DataFrame()
    usable = rows.dropna(subset=[value_col]).copy()
    if usable.empty:
        return pd.DataFrame()
    return (
        usable.groupby(group_cols, dropna=False)[value_col]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def plot_accuracy_summary_by_sparsity(
    ax,
    aggregate: pd.DataFrame,
    include_cka_strength: bool = False,
) -> bool:
    """Plot summary mean/std accuracy curves over sparsity."""

    plotted = False
    group_cols = ["method", "cka_strength"]
    for (method, cka_strength), group in aggregate.groupby(
        group_cols,
        dropna=False,
        sort=False,
    ):
        if method not in SPARSE_METHODS:
            continue
        group = group.sort_values("sparsity")
        color = (
            cka_strength_color(cka_strength)
            if method == "cka_feddst" and include_cka_strength
            else method_color(method)
        )
        marker = (
            cka_strength_marker(cka_strength)
            if method == "cka_feddst" and include_cka_strength
            else METHOD_MARKERS.get(method, "o")
        )
        ax.errorbar(
            group["sparsity"],
            group["mean"],
            yerr=group["std"].fillna(0.0),
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            capsize=3,
            label=aggregate_method_name(method, cka_strength, include_cka_strength),
        )
        plotted = True
    return plotted


def plot_cka_strength_summary(
    rows: pd.DataFrame,
    output_path: Path,
    title: str,
    ylabel: str,
) -> Path | None:
    """Plot a CKA-strength accuracy summary over sparsity."""

    aggregate = aggregate_metric(
        rows,
        ["cka_strength", "sparsity"],
        "test_accuracy",
    )
    if aggregate.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for strength, group in aggregate.groupby("cka_strength", sort=True):
        group = group.sort_values("sparsity")
        ax.errorbar(
            group["sparsity"],
            group["mean"],
            yerr=group["std"].fillna(0.0),
            color=cka_strength_color(strength),
            marker=cka_strength_marker(strength),
            linewidth=2.2,
            markersize=6,
            capsize=3,
            label=f"cka_strength={format_sparsity(strength)}",
        )

    style_axes(ax, title, "Sparsity", ylabel)
    ax.legend(title="CKA strength", fontsize=8)
    return save_figure(fig, output_path)


def add_dense_mean_reference(
    ax,
    rows: pd.DataFrame,
    label: str,
) -> None:
    """Draw mean FedAvg accuracy as a dense reference line."""

    if rows.empty or "test_accuracy" not in rows.columns:
        return
    fedavg = rows[rows["method"] == "fedavg"]
    values = pd.to_numeric(fedavg["test_accuracy"], errors="coerce").dropna()
    if values.empty:
        return
    mean = values.mean()
    std = values.std()
    ax.axhline(
        mean,
        color=method_color("fedavg"),
        linestyle="--",
        linewidth=2,
        label=label,
    )
    if len(values) > 1 and not pd.isna(std):
        ax.axhspan(
            mean - std,
            mean + std,
            color=method_color("fedavg"),
            alpha=0.08,
        )


def cka_strength_logs(logs: pd.DataFrame) -> pd.DataFrame:
    """Return CKA-FedDST logs with numeric CKA-strength values."""

    if logs.empty or "cka_strength" not in logs.columns:
        return pd.DataFrame()
    rows = logs[logs["method"] == "cka_feddst"].copy()
    rows["cka_strength"] = pd.to_numeric(rows["cka_strength"], errors="coerce")
    return rows.dropna(subset=["cka_strength"])


def unique_cka_strengths(logs: pd.DataFrame) -> list[float]:
    """Return sorted CKA-strength values."""

    if logs.empty or "cka_strength" not in logs.columns:
        return []
    values = pd.to_numeric(logs["cka_strength"], errors="coerce").dropna()
    return sorted({round(float(value), 4) for value in values})


def has_multiple_cka_strengths(logs: pd.DataFrame) -> bool:
    """Return true when logs include multiple CKA-strength settings."""

    return len(unique_cka_strengths(logs)) > 1


def aggregate_method_label(
    method: str,
    sparsity,
    cka_strength,
    include_cka_strength: bool = False,
) -> str:
    """Return a label for mean/std round curves."""

    label = method_sparsity_label_from_values(method, sparsity)
    if method == "cka_feddst" and include_cka_strength and not pd.isna(cka_strength):
        label = f"{label} cka={format_sparsity(cka_strength)}"
    return label


def aggregate_method_name(
    method: str,
    cka_strength,
    include_cka_strength: bool = False,
) -> str:
    """Return a summary-plot method label."""

    label = METHOD_LABELS.get(method, method)
    if method == "cka_feddst" and include_cka_strength and not pd.isna(cka_strength):
        label = f"{label} cka={format_sparsity(cka_strength)}"
    return label


def method_sparsity_label_from_values(method: str, sparsity) -> str:
    """Return method/sparsity label from scalar values."""

    label = METHOD_LABELS.get(method, method)
    if method == "fedavg":
        return f"{label} dense"
    return f"{label} s={format_sparsity(sparsity)}"


def cka_strength_color(value) -> str:
    """Return the color assigned to a CKA-strength value."""

    key = strength_key(value)
    return CKA_STRENGTH_COLORS.get(key, "#777777")


def cka_strength_marker(value) -> str:
    """Return the marker assigned to a CKA-strength value."""

    key = strength_key(value)
    return CKA_STRENGTH_MARKERS.get(key, "o")


def strength_key(value) -> float:
    """Normalize CKA-strength values for style lookups."""

    if pd.isna(value):
        return float("nan")
    value = float(value)
    known = sorted(CKA_STRENGTH_COLORS)
    nearest = min(known, key=lambda candidate: abs(candidate - value))
    if abs(nearest - value) < 1e-6:
        return nearest
    return round(value, 4)


def add_cka_strength_sparsity_legends(ax, logs: pd.DataFrame) -> None:
    """Add separate legends for CKA strength and sparsity."""

    strength_handles = [
        Line2D(
            [0],
            [0],
            color=cka_strength_color(value),
            marker=cka_strength_marker(value),
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=f"cka={format_sparsity(value)}",
        )
        for value in unique_cka_strengths(logs)
    ]
    sparsity_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker=sparsity_style(value)["marker"],
            linestyle=sparsity_style(value)["linestyle"],
            linewidth=2,
            markersize=6,
            label=sparsity_label(value),
        )
        for value in unique_sparsities(logs)
    ]

    if strength_handles:
        strength_legend = ax.legend(
            handles=strength_handles,
            title="CKA strength",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
        )
        ax.add_artist(strength_legend)
    if sparsity_handles:
        ax.legend(
            handles=sparsity_handles,
            title="Sparsity",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            fontsize=8,
        )


def sparse_only(logs: pd.DataFrame) -> pd.DataFrame:
    """Return logs for sparse methods with numeric sparsity values."""

    if logs.empty:
        return logs
    sparse_logs = logs[logs["method"].isin(SPARSE_METHODS)].copy()
    return sparse_logs.dropna(subset=["sparsity"])


def sorted_sources(logs: pd.DataFrame):
    """Yield source-file groups in method/sparsity order."""

    source_col = source_column(logs)
    columns = [
        column
        for column in ("method", "sparsity", "seed", "cka_strength", "source_file")
        if column in logs.columns
    ]
    if source_col not in columns:
        columns.append(source_col)
    source_rows = logs[columns].drop_duplicates()
    source_rows["method_order"] = source_rows["method"].map(method_rank)
    sort_columns = [
        column
        for column in ("method_order", "sparsity", "cka_strength", "seed", "source_file")
        if column in source_rows.columns
    ]
    source_rows = source_rows.sort_values(sort_columns)
    for _, row in source_rows.iterrows():
        yield row[source_col], logs[logs[source_col] == row[source_col]]


def final_rows(logs: pd.DataFrame) -> pd.DataFrame:
    """Return the last round from each source log."""

    if logs.empty:
        return logs
    source_col = source_column(logs)
    return (
        logs.sort_values([source_col, "round"])
        .groupby(source_col, as_index=False)
        .tail(1)
    )


def best_rows(logs: pd.DataFrame) -> pd.DataFrame:
    """Return the best test-accuracy row from each source log."""

    if logs.empty:
        return logs
    idx = logs.groupby(source_column(logs))["test_accuracy"].idxmax()
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


def infer_seed(frame: pd.DataFrame, path: Path) -> int | float:
    """Infer seed from a CSV column first, then from the filename."""

    if "seed" in frame.columns:
        values = pd.to_numeric(frame["seed"], errors="coerce").dropna()
        if not values.empty:
            return int(values.iloc[0])

    match = SEED_PATTERN.search(path.name)
    if match:
        return int(match.group("value"))

    return float("nan")


def infer_cka_strength(frame: pd.DataFrame, path: Path, method: str) -> float:
    """Infer CKA strength from a CSV column, filename, or suite folder."""

    if method != "cka_feddst":
        return float("nan")

    if "cka_strength" in frame.columns:
        values = pd.to_numeric(frame["cka_strength"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])

    for text in (path.name, *[part.name for part in path.parents]):
        match = CKA_STRENGTH_PATTERN.search(text)
        if match:
            return float(match.group("value").replace("p", "."))
        match = CKA_STRENGTH_DIR_PATTERN.search(text)
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


def source_column(logs: pd.DataFrame) -> str:
    """Return the source identity column available in a log frame."""

    return "source_path" if "source_path" in logs.columns else "source_file"


def method_sparsity_label(group: pd.DataFrame) -> str:
    """Return a legend label for a source-file group."""

    method = str(group["method"].iloc[0])
    label = METHOD_LABELS.get(method, method)
    if method == "fedavg":
        return f"{label} dense"
    return f"{label} s={format_sparsity(group['sparsity'].iloc[0])}"


def plot_method_sparsity_line(
    ax,
    group: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> None:
    """Plot one source log with color as method and marker/style as sparsity."""

    method = str(group["method"].iloc[0])
    sparsity = group["sparsity"].iloc[0]
    style = sparsity_style(sparsity)
    ax.plot(
        group[x_column],
        group[y_column],
        color=method_color(method),
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=2.2 if method == "fedavg" else 1.8,
        markersize=7 if method == "fedavg" else 5.5,
        label="_nolegend_",
    )


def add_dense_reference_line(
    ax,
    rows: pd.DataFrame,
    label: str,
) -> None:
    """Draw FedAvg as a dense reference on sparsity summary plots."""

    fedavg_rows = rows[rows["method"] == "fedavg"]
    if fedavg_rows.empty or "test_accuracy" not in fedavg_rows.columns:
        return
    value = pd.to_numeric(fedavg_rows["test_accuracy"], errors="coerce").dropna()
    if value.empty:
        return
    ax.axhline(
        value.iloc[0],
        color=method_color("fedavg"),
        linestyle="--",
        linewidth=2,
        label=label,
    )


def add_method_sparsity_legends(ax, logs: pd.DataFrame) -> None:
    """Add separate legends for method colors and sparsity symbols."""

    methods = [method for method in METHOD_ORDER if method in set(logs["method"])]
    method_handles = [
        Line2D(
            [0],
            [0],
            color=method_color(method),
            linewidth=3,
            label=METHOD_LABELS.get(method, method),
        )
        for method in methods
    ]
    sparsity_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker=sparsity_style(value)["marker"],
            linestyle=sparsity_style(value)["linestyle"],
            linewidth=2,
            markersize=6,
            label=sparsity_label(value),
        )
        for value in unique_sparsities(logs)
    ]

    if method_handles:
        method_legend = ax.legend(
            handles=method_handles,
            title="Method color",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
        )
        ax.add_artist(method_legend)
    if sparsity_handles:
        ax.legend(
            handles=sparsity_handles,
            title="Sparsity symbol",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            fontsize=8,
        )


def add_sparsity_layer_legends(
    ax,
    logs: pd.DataFrame,
    layers: list[str],
) -> None:
    """Add separate legends for sparsity colors/symbols and layer line styles."""

    sparsity_handles = [
        Line2D(
            [0],
            [0],
            color=sparsity_color(value),
            marker=sparsity_style(value)["marker"],
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=sparsity_label(value),
        )
        for value in unique_sparsities(logs)
        if value != 0.0
    ]
    layer_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=layer_linestyle(layer),
            linewidth=2,
            label=layer,
        )
        for layer in layers
    ]

    if sparsity_handles:
        sparsity_legend = ax.legend(
            handles=sparsity_handles,
            title="Sparsity",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
        )
        ax.add_artist(sparsity_legend)
    if layer_handles:
        ax.legend(
            handles=layer_handles,
            title="Layer",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            fontsize=8,
        )


def unique_sparsities(logs: pd.DataFrame) -> list[float]:
    """Return stable, sorted sparsity values from a log frame."""

    if logs.empty or "sparsity" not in logs.columns:
        return []
    values = pd.to_numeric(logs["sparsity"], errors="coerce").dropna()
    unique = sorted({sparsity_key(value) for value in values})
    return [value for value in unique if not pd.isna(value)]


def method_color(method: str) -> str:
    """Return the color assigned to a method."""

    return METHOD_COLORS.get(method, "#777777")


def sparsity_color(value) -> str:
    """Return a color assigned to a sparsity level."""

    key = sparsity_key(value)
    return SPARSITY_COLORS.get(key, "#777777")


def sparsity_style(value) -> dict:
    """Return marker and line style assigned to a sparsity level."""

    key = sparsity_key(value)
    if key in SPARSITY_STYLES:
        return SPARSITY_STYLES[key]
    return {"marker": "o", "linestyle": "-", "label": sparsity_label(value)}


def sparsity_label(value) -> str:
    """Return a human-readable sparsity label."""

    key = sparsity_key(value)
    if key in SPARSITY_STYLES:
        return SPARSITY_STYLES[key]["label"]
    if pd.isna(key):
        return "s=unknown"
    return f"s={format_sparsity(key)}"


def sparsity_key(value) -> float:
    """Normalize sparsity floats so legends are stable across CSVs."""

    if pd.isna(value):
        return float("nan")
    value = float(value)
    known = sorted(SPARSITY_STYLES)
    nearest = min(known, key=lambda candidate: abs(candidate - value))
    if abs(nearest - value) < 1e-6:
        return nearest
    return round(value, 4)


def layer_linestyle(layer: str):
    """Return the line style assigned to a CKA layer."""

    return LAYER_LINESTYLES.get(layer, "-")


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


def style_axes(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    show_legend: bool = True,
) -> None:
    """Apply consistent labels, legend, and grid."""

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.margins(x=0.03)
    if not show_legend:
        return
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)


def save_figure(fig, output_path: Path) -> Path:
    """Save a matplotlib figure and close it."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
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
