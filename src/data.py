"""MNIST data pipeline for simulated federated learning experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import make_torch_generator, seed_everything, seed_worker


@dataclass(frozen=True)
class DataConfig:
    """Configuration for MNIST data preparation."""

    data_dir: Path = Path("data")
    num_clients: int = 10
    alpha: float = 0.5
    batch_size: int = 64
    seed: int = 0
    reference_size: int = 200
    num_workers: int = 0
    download: bool = True
    print_stats: bool = True


def load_mnist(config: DataConfig | None = None):
    """Build federated MNIST train loaders, a test loader, and CKA reference loader.

    Args:
        config: Dataset, partitioning, and dataloader settings.

    Returns:
        A tuple containing client train dataloaders, the global test dataloader,
        and a small balanced reference dataloader for representation analysis.
    """

    if config is None:
        config = DataConfig()

    _validate_config(config)
    seed_everything(config.seed)

    datasets, transforms = _load_torchvision()
    data_dir = Path(config.data_dir)
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=config.download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=config.download,
        transform=transform,
    )

    client_datasets = partition_clients(train_dataset, config)
    if config.print_stats:
        print_client_label_distributions(client_datasets)

    DataLoader = _load_dataloader()
    client_loaders = [
        DataLoader(
            client_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            generator=make_torch_generator(config.seed + client_id),
        )
        for client_id, client_dataset in enumerate(client_datasets)
    ]
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(config.seed + 10_000),
    )
    reference_dataset = make_balanced_reference_dataset(
        test_dataset,
        size=config.reference_size,
        seed=config.seed,
    )
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(config.seed + 20_000),
    )

    return client_loaders, test_loader, reference_loader


def partition_clients(dataset, config: DataConfig):
    """Split a labeled dataset into Dirichlet label-skew client subsets.

    Args:
        dataset: Dataset with a ``targets`` attribute, such as torchvision MNIST.
        config: Partitioning settings, including number of clients and alpha.

    Returns:
        A list of ``torch.utils.data.Subset`` objects, one per simulated client.
    """

    _validate_config(config)
    Subset = _load_subset()
    labels = _dataset_targets(dataset)
    classes = np.unique(labels)

    for attempt in range(100):
        rng = np.random.default_rng(config.seed + attempt)
        client_indices = [[] for _ in range(config.num_clients)]

        for label in classes:
            label_indices = np.flatnonzero(labels == label)
            rng.shuffle(label_indices)

            proportions = rng.dirichlet(
                np.full(config.num_clients, config.alpha, dtype=np.float64)
            )
            split_counts = rng.multinomial(len(label_indices), proportions)
            split_points = np.cumsum(split_counts)[:-1]

            for client_id, split in enumerate(np.split(label_indices, split_points)):
                client_indices[client_id].extend(split.tolist())

        for indices in client_indices:
            rng.shuffle(indices)

        if all(indices for indices in client_indices):
            return [Subset(dataset, indices) for indices in client_indices]

    raise RuntimeError(
        "Dirichlet partitioning produced an empty client after 100 attempts. "
        "Try a larger alpha or fewer clients."
    )


def make_balanced_reference_dataset(dataset, size: int, seed: int):
    """Create a small balanced subset for shared CKA/reference evaluation."""

    if size <= 0:
        raise ValueError("reference_size must be positive.")

    Subset = _load_subset()
    labels = _dataset_targets(dataset)
    classes = np.unique(labels)
    if size < len(classes):
        raise ValueError(
            f"reference_size must be at least the number of labels ({len(classes)})."
        )

    rng = np.random.default_rng(seed)
    per_class = size // len(classes)
    remainder = size % len(classes)
    reference_indices = []

    for class_position, label in enumerate(classes):
        take = per_class + int(class_position < remainder)
        label_indices = np.flatnonzero(labels == label)
        if len(label_indices) < take:
            raise ValueError(f"Not enough examples for label {label} to build reference.")
        rng.shuffle(label_indices)
        reference_indices.extend(label_indices[:take].tolist())

    rng.shuffle(reference_indices)
    return Subset(dataset, reference_indices)


def print_client_label_distributions(client_datasets) -> None:
    """Print sample counts and label histograms for each client subset."""

    if not client_datasets:
        print("No client datasets were created.")
        return

    base_dataset = client_datasets[0].dataset
    labels = _dataset_targets(base_dataset)
    num_classes = int(np.max(labels)) + 1

    print("Client label distributions:")
    for client_id, client_dataset in enumerate(client_datasets):
        indices = np.asarray(client_dataset.indices, dtype=np.int64)
        counts = label_distribution(labels[indices], num_classes=num_classes)
        print(
            f"  client {client_id:02d}: "
            f"n={len(indices):5d}, labels={counts}"
        )


def label_distribution(labels, num_classes: int = 10) -> list[int]:
    """Return a fixed-width label histogram as plain Python integers."""

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    return counts.astype(int).tolist()


def _validate_config(config: DataConfig) -> None:
    """Validate user-facing data settings early."""

    if config.num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if config.alpha <= 0:
        raise ValueError("alpha must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")


def _dataset_targets(dataset) -> np.ndarray:
    """Return dataset targets as a one-dimensional NumPy array."""

    if not hasattr(dataset, "targets"):
        raise AttributeError("Dataset must expose a 'targets' attribute.")

    targets = dataset.targets
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    return np.asarray(targets, dtype=np.int64)


def _load_torchvision():
    """Import torchvision lazily so CLI help works without data dependencies."""

    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required to load MNIST. Install torchvision before "
            "running data experiments."
        ) from exc

    return datasets, transforms


def _load_dataloader():
    """Import DataLoader lazily."""

    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for dataloaders. Install torch before running "
            "data experiments."
        ) from exc

    return DataLoader


def _load_subset():
    """Import Subset lazily."""

    try:
        from torch.utils.data import Subset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for dataset subsets. Install torch before running "
            "data experiments."
        ) from exc

    return Subset
