"""Shared utility functions for experiment code."""

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path
