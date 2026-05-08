"""Dataset loading and client partitioning placeholders.

This module will hold MNIST loading logic and simulated client splits. The
implementation is intentionally deferred so the initial scaffold stays small.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset preparation."""

    data_dir: Path = Path("data")
    num_clients: int = 10
    iid: bool = True


def load_mnist(config: DataConfig):
    """Load MNIST datasets for later federated experiments.

    Args:
        config: Dataset and partitioning settings.

    Raises:
        NotImplementedError: The real MNIST loader has not been added yet.
    """

    raise NotImplementedError("MNIST loading is not implemented in this scaffold.")


def partition_clients(dataset, config: DataConfig):
    """Split a dataset into simulated client datasets.

    Args:
        dataset: Dataset-like object to partition.
        config: Partitioning settings.

    Raises:
        NotImplementedError: Client partitioning has not been added yet.
    """

    raise NotImplementedError("Client partitioning is not implemented yet.")
