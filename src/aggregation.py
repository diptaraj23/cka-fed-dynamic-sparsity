"""Aggregate raw experiment logs into reusable mean/std CSVs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .plotting import (
    best_rows,
    cka_value_columns,
    final_rows,
    latest_by_run_identity,
    layer_sparsity_columns,
    load_training_logs,
    pretty_layer,
)


ROUND_METRICS = (
    "test_accuracy",
    "test_loss",
    "avg_train_loss",
    "total_sparsity",
    "active_params",
    "total_params",
    "pruned_weights",
    "regrown_weights",
    "mask_changes",
    "cka_computed",
)
SUMMARY_METRICS = (
    "test_accuracy",
    "test_loss",
    "avg_train_loss",
    "total_sparsity",
    "active_params",
    "total_params",
    "pruned_weights",
    "regrown_weights",
    "mask_changes",
)
METRIC_COLUMNS = [
    "suite",
    "metric",
    "method",
    "dataset",
    "sparsity",
    "cka_strength",
    "mean",
    "std",
    "count",
    "seed_count",
    "seeds",
]
ROUND_METRIC_COLUMNS = [
    "suite",
    "metric",
    "method",
    "dataset",
    "sparsity",
    "cka_strength",
    "round",
    "mean",
    "std",
    "count",
    "seed_count",
    "seeds",
]
LAYER_COLUMNS = [
    "suite",
    "metric",
    "method",
    "dataset",
    "sparsity",
    "cka_strength",
    "round",
    "layer",
    "mean",
    "std",
    "count",
    "seed_count",
    "seeds",
]
MANIFEST_COLUMNS = [
    "suite",
    "log_dir",
    "output_dir",
    "source_files",
    "raw_rows",
    "unique_runs",
    "seeds",
    "created_at",
]


def aggregate_results(log_dir: Path, output_dir: Path, suite: str) -> list[Path]:
    """Aggregate raw run logs across seeds and save mean/std CSV files."""

    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_logs = load_training_logs(log_dir)
    logs = latest_by_run_identity(raw_logs)
    saved_paths = []

    round_metrics = aggregate_metric_table(
        logs,
        suite=suite,
        metrics=ROUND_METRICS,
        group_cols=["method", "dataset", "sparsity", "cka_strength", "round"],
        output_columns=ROUND_METRIC_COLUMNS,
    )
    saved_paths.append(
        write_frame(
            round_metrics,
            output_dir / "round_metrics_mean_std.csv",
            ROUND_METRIC_COLUMNS,
        )
    )

    finals = final_rows(logs)
    final_metrics = aggregate_metric_table(
        finals,
        suite=suite,
        metrics=SUMMARY_METRICS,
        group_cols=["method", "dataset", "sparsity", "cka_strength"],
        output_columns=METRIC_COLUMNS,
    )
    saved_paths.append(
        write_frame(
            final_metrics,
            output_dir / "final_metrics_mean_std.csv",
            METRIC_COLUMNS,
        )
    )

    best = safe_best_rows(logs)
    best_metrics = aggregate_metric_table(
        best,
        suite=suite,
        metrics=SUMMARY_METRICS,
        group_cols=["method", "dataset", "sparsity", "cka_strength"],
        output_columns=METRIC_COLUMNS,
    )
    saved_paths.append(
        write_frame(
            best_metrics,
            output_dir / "best_metrics_mean_std.csv",
            METRIC_COLUMNS,
        )
    )

    layerwise_sparsity = aggregate_layer_table(
        logs,
        suite=suite,
        columns=layer_sparsity_columns(logs),
        metric_name="actual_sparsity",
    )
    target_sparsity = aggregate_layer_table(
        logs,
        suite=suite,
        columns=target_sparsity_columns(logs),
        metric_name="target_sparsity",
    )
    layer_frames = [
        frame for frame in (layerwise_sparsity, target_sparsity)
        if not frame.empty
    ]
    layerwise_sparsity = (
        pd.concat(layer_frames, ignore_index=True, sort=False)
        if layer_frames
        else pd.DataFrame(columns=LAYER_COLUMNS)
    )
    saved_paths.append(
        write_frame(
            layerwise_sparsity,
            output_dir / "layerwise_sparsity_mean_std.csv",
            LAYER_COLUMNS,
        )
    )

    layerwise_cka = aggregate_layer_table(
        logs,
        suite=suite,
        columns=cka_value_columns(logs),
        metric_name="cka",
    )
    saved_paths.append(
        write_frame(
            layerwise_cka,
            output_dir / "layerwise_cka_mean_std.csv",
            LAYER_COLUMNS,
        )
    )

    manifest = build_aggregation_manifest(raw_logs, logs, suite, log_dir, output_dir)
    saved_paths.append(
        write_frame(
            manifest,
            output_dir / "aggregation_manifest.csv",
            MANIFEST_COLUMNS,
        )
    )

    return saved_paths


def aggregate_metric_table(
    rows: pd.DataFrame,
    suite: str,
    metrics: tuple[str, ...],
    group_cols: list[str],
    output_columns: list[str],
) -> pd.DataFrame:
    """Aggregate selected scalar metrics into a long mean/std table."""

    if rows.empty:
        return pd.DataFrame(columns=output_columns)

    frames = []
    for metric in metrics:
        if metric not in rows.columns:
            continue
        frame = rows.copy()
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        stats = aggregate_value(frame, group_cols, metric)
        if stats.empty:
            continue
        stats.insert(0, "metric", metric)
        stats.insert(0, "suite", suite)
        frames.append(stats)

    if not frames:
        return pd.DataFrame(columns=output_columns)
    return pd.concat(frames, ignore_index=True, sort=False)[output_columns]


def aggregate_layer_table(
    rows: pd.DataFrame,
    suite: str,
    columns: list[str],
    metric_name: str,
) -> pd.DataFrame:
    """Aggregate per-layer round metrics into a long mean/std table."""

    group_cols = ["method", "dataset", "sparsity", "cka_strength", "round", "layer"]
    if rows.empty or not columns:
        return pd.DataFrame(columns=LAYER_COLUMNS)

    frames = []
    for column in columns:
        if column not in rows.columns:
            continue
        frame = rows.copy()
        frame["layer"] = pretty_layer(column)
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        stats = aggregate_value(frame, group_cols, column)
        if stats.empty:
            continue
        stats.insert(0, "metric", metric_name)
        stats.insert(0, "suite", suite)
        frames.append(stats)

    if not frames:
        return pd.DataFrame(columns=LAYER_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False)[LAYER_COLUMNS]


def aggregate_value(
    rows: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    """Aggregate one numeric column with seed counts and seed lists."""

    if rows.empty or value_col not in rows.columns:
        return pd.DataFrame()

    usable = rows.dropna(subset=[value_col]).copy()
    if usable.empty:
        return pd.DataFrame()

    for column in group_cols:
        if column not in usable.columns:
            usable[column] = pd.NA

    grouped = usable.groupby(group_cols, dropna=False)
    return (
        grouped.agg(
            mean=(value_col, "mean"),
            std=(value_col, "std"),
            count=(value_col, "count"),
            seed_count=("seed", seed_count),
            seeds=("seed", seed_list),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def safe_best_rows(logs: pd.DataFrame) -> pd.DataFrame:
    """Return best rows while ignoring sources with no evaluated accuracy."""

    if logs.empty or "test_accuracy" not in logs.columns:
        return logs.iloc[0:0].copy()
    usable = logs.dropna(subset=["test_accuracy"]).copy()
    if usable.empty:
        return usable
    return best_rows(usable)


def target_sparsity_columns(frame: pd.DataFrame) -> list[str]:
    """Return CKA-guided target sparsity columns."""

    return [
        column
        for column in frame.columns
        if column.startswith("target_sparsity_")
    ]


def seed_count(values: pd.Series) -> int:
    """Return the number of unique non-null seeds."""

    return len(seed_values(values))


def seed_list(values: pd.Series) -> str:
    """Return a compact comma-separated list of unique seeds."""

    return ",".join(str(seed) for seed in seed_values(values))


def seed_values(values: pd.Series) -> list[int]:
    """Return sorted unique integer seeds from a Series."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return sorted({int(value) for value in numeric})


def build_aggregation_manifest(
    raw_logs: pd.DataFrame,
    logs: pd.DataFrame,
    suite: str,
    log_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Build a one-row manifest describing the aggregation input and output."""

    source_files = ""
    if not logs.empty and "source_path" in logs.columns:
        source_files = ";".join(sorted(logs["source_path"].dropna().unique()))

    seeds = ""
    if not logs.empty and "seed" in logs.columns:
        seeds = seed_list(logs["seed"])

    unique_runs = 0
    if not logs.empty and "source_path" in logs.columns:
        unique_runs = int(logs["source_path"].nunique())

    return pd.DataFrame(
        [
            {
                "suite": suite,
                "log_dir": str(log_dir),
                "output_dir": str(output_dir),
                "source_files": source_files,
                "raw_rows": int(len(raw_logs)),
                "unique_runs": unique_runs,
                "seeds": seeds,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        ],
        columns=MANIFEST_COLUMNS,
    )


def write_frame(frame: pd.DataFrame, path: Path, columns: list[str]) -> Path:
    """Write a CSV with stable columns, even when the frame is empty."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, index=False)
    return path
