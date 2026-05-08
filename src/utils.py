"""Shared utility functions for experiment code."""

import random
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    """Seed common random number generators for reproducible experiments."""

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ModuleNotFoundError:
        pass


def seed_worker(worker_id: int) -> None:
    """Seed dataloader workers from PyTorch's worker seed."""

    try:
        import numpy as np
        import torch

        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    except ModuleNotFoundError:
        random.seed(worker_id)


def make_torch_generator(seed: int):
    """Create a seeded PyTorch generator for deterministic dataloaders."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for dataloader construction. "
            "Install torch before running data experiments."
        ) from exc

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
