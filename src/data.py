"""Data pipeline for simulated federated learning image experiments."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .utils import make_torch_generator, seed_everything, seed_worker


SUPPORTED_DATASETS = {"mnist", "fashion_mnist"}


@dataclass(frozen=True)
class DataConfig:
    """Configuration for federated data preparation."""

    dataset: str = "mnist"
    data_dir: Path = Path("data")
    num_clients: int = 10
    alpha: float = 0.5
    batch_size: int = 64
    seed: int = 0
    reference_size: int = 200
    num_workers: int = 0
    split_dir: Path | None = None
    download: bool = True
    print_stats: bool = True


def load_mnist(config: DataConfig | None = None):
    """Build federated MNIST loaders.

    This compatibility wrapper preserves older imports while the generic
    loader supports additional MNIST-style datasets.
    """

    if config is None:
        config = DataConfig(dataset="mnist")
    else:
        config = replace(config, dataset="mnist")
    return load_federated_data(config)


def load_federated_data(config: DataConfig | None = None):
    """Build federated train loaders, a test loader, and a CKA reference loader.

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
    dataset_cls = _dataset_class(datasets, config.dataset)

    train_dataset = dataset_cls(
        root=str(data_dir),
        train=True,
        download=config.download,
        transform=transform,
    )
    test_dataset = dataset_cls(
        root=str(data_dir),
        train=False,
        download=config.download,
        transform=transform,
    )

    reference_dataset = make_balanced_reference_dataset(
        train_dataset,
        size=config.reference_size,
        seed=config.seed,
    )
    reference_indices = set(get_subset_indices(reference_dataset))

    client_datasets = partition_clients(
        train_dataset,
        config,
        exclude_indices=reference_indices,
    )
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
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(config.seed + 20_000),
    )

    if config.split_dir is not None:
        split_path = make_split_manifest_path(config)
        save_split_manifest(config, client_datasets, reference_dataset, split_path)

    return client_loaders, test_loader, reference_loader


def partition_clients(
    dataset,
    config: DataConfig,
    exclude_indices: set[int] | None = None,
):
    """Split a labeled dataset into Dirichlet label-skew client subsets.

    Args:
        dataset: Dataset with a ``targets`` attribute, such as torchvision MNIST.
        config: Partitioning settings, including number of clients and alpha.
        exclude_indices: Optional dataset indices reserved outside client training.

    Returns:
        A list of ``torch.utils.data.Subset`` objects, one per simulated client.
    """

    _validate_config(config)
    Subset = _load_subset()
    labels = _dataset_targets(dataset)
    candidate_indices = np.arange(len(labels))
    if exclude_indices:
        keep_mask = np.ones(len(labels), dtype=bool)
        keep_mask[np.asarray(sorted(exclude_indices), dtype=np.int64)] = False
        candidate_indices = candidate_indices[keep_mask]

    client_indices = partition_client_indices(
        labels=labels,
        num_clients=config.num_clients,
        alpha=config.alpha,
        seed=config.seed,
        candidate_indices=candidate_indices,
    )
    return [Subset(dataset, indices) for indices in client_indices]


def partition_client_indices(
    labels,
    num_clients: int,
    alpha: float,
    seed: int,
    candidate_indices=None,
) -> list[list[int]]:
    """Return deterministic Dirichlet-partitioned indices for each client."""

    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    if candidate_indices is None:
        candidate_indices = np.arange(len(labels))
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_labels = labels[candidate_indices]

    for attempt in range(100):
        rng = np.random.default_rng(seed + attempt)
        client_indices = [[] for _ in range(num_clients)]

        for label in classes:
            label_indices = candidate_indices[candidate_labels == label]
            rng.shuffle(label_indices)

            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            split_counts = rng.multinomial(len(label_indices), proportions)
            split_points = np.cumsum(split_counts)[:-1]

            for client_id, split in enumerate(np.split(label_indices, split_points)):
                client_indices[client_id].extend(split.tolist())

        for indices in client_indices:
            rng.shuffle(indices)

        if all(indices for indices in client_indices):
            return client_indices

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


def get_subset_indices(subset) -> list[int]:
    """Return subset indices as plain Python integers for saving and comparison."""

    return [int(index) for index in subset.indices]


def make_split_manifest_path(config: DataConfig) -> Path:
    """Create a deterministic filename for the saved data split manifest."""

    if config.split_dir is None:
        raise ValueError("split_dir must be set before creating a split manifest path.")

    alpha_token = str(config.alpha).replace(".", "p")
    filename = (
        f"{_dataset_token(config.dataset)}_split_seed{config.seed}_"
        f"clients{config.num_clients}_"
        f"alpha{alpha_token}_ref{config.reference_size}.json"
    )
    return Path(config.split_dir) / filename


def save_split_manifest(
    config: DataConfig,
    client_datasets,
    reference_dataset,
    output_path: Path,
) -> None:
    """Save exact client and reference indices for run reproducibility."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_dataset = (
        client_datasets[0].dataset
        if client_datasets
        else reference_dataset.dataset
    )
    labels = _dataset_targets(base_dataset)
    num_classes = int(np.max(labels)) + 1

    clients = []
    for client_id, client_dataset in enumerate(client_datasets):
        indices = np.asarray(get_subset_indices(client_dataset), dtype=np.int64)
        clients.append(
            {
                "client_id": client_id,
                "num_samples": int(len(indices)),
                "label_distribution": label_distribution(
                    labels[indices],
                    num_classes=num_classes,
                ),
                "indices": indices.astype(int).tolist(),
            }
        )

    reference_indices = np.asarray(
        get_subset_indices(reference_dataset),
        dtype=np.int64,
    )
    manifest = {
        "dataset": _dataset_token(config.dataset),
        "seed": int(config.seed),
        "num_clients": int(config.num_clients),
        "alpha": float(config.alpha),
        "reference_size": int(config.reference_size),
        "num_train_samples_reserved_for_reference": int(len(reference_indices)),
        "reference_source": f"{_dataset_token(config.dataset)}_train",
        "reference_label_distribution": label_distribution(
            labels[reference_indices],
            num_classes=num_classes,
        ),
        "reference_indices": reference_indices.astype(int).tolist(),
        "clients": clients,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _validate_config(config: DataConfig) -> None:
    """Validate user-facing data settings early."""

    if _dataset_token(config.dataset) not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported dataset '{config.dataset}'. Expected one of: {supported}."
        )
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
            "torchvision is required to load image datasets. Install torchvision "
            "before running data experiments."
        ) from exc

    return datasets, transforms


def _dataset_class(datasets, dataset: str):
    """Return the torchvision dataset class for a supported dataset name."""

    token = _dataset_token(dataset)
    if token == "mnist":
        return datasets.MNIST
    if token == "fashion_mnist":
        return datasets.FashionMNIST
    supported = ", ".join(sorted(SUPPORTED_DATASETS))
    raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {supported}.")


def _dataset_token(dataset: str) -> str:
    """Normalize dataset names used by configs and filenames."""

    return str(dataset).lower().replace("-", "_")


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
