"""Plotting helpers for experiment outputs."""

from pathlib import Path


def save_training_curve(history, output_path: Path):
    """Save a training curve plot.

    Args:
        history: Training metrics collected over time.
        output_path: Destination path for the figure.

    Raises:
        NotImplementedError: Plotting logic is not implemented yet.
    """

    raise NotImplementedError("Plotting is not implemented in this scaffold.")
