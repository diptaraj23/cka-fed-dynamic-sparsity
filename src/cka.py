"""Layer-wise linear CKA for comparing client representations."""

import csv
import math
from pathlib import Path

import torch


DEFAULT_CKA_LAYERS = ("conv1", "conv2", "fc1")


def collect_activations(
    model,
    reference_loader,
    device=None,
    layers: tuple[str, ...] = DEFAULT_CKA_LAYERS,
) -> dict[str, torch.Tensor]:
    """Collect flattened layer activations on a shared reference dataloader.

    Args:
        model: Model that supports ``return_activations=True``.
        reference_loader: Shared dataloader used for representation comparison.
        device: Device for the forward pass. Defaults to the model device.
        layers: Activation names to collect.

    Returns:
        A dictionary mapping layer names to tensors of shape
        ``[num_samples, features]``.
    """

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    was_training = model.training
    model.to(device)
    model.eval()

    collected = {layer: [] for layer in layers}
    with torch.no_grad():
        for images, _ in reference_loader:
            images = images.to(device)
            _, activations = model(images, return_activations=True)

            for layer in layers:
                if layer not in activations:
                    raise KeyError(f"Layer '{layer}' was not returned by the model.")
                flattened = torch.flatten(activations[layer].detach().cpu(), start_dim=1)
                collected[layer].append(flattened)

    if was_training:
        model.train()

    flattened_activations = {}
    for layer, chunks in collected.items():
        if not chunks:
            raise ValueError("Reference dataloader produced no batches.")
        flattened_activations[layer] = torch.cat(chunks, dim=0).float()
    return flattened_activations


def linear_cka(
    activations_a: torch.Tensor,
    activations_b: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Compute linear CKA between two flattened activation matrices.

    Both inputs must have shape ``[num_samples, features]``. The computation
    centers features first, then uses sample Gram matrices for numerical and
    memory efficiency.
    """

    if activations_a.ndim != 2 or activations_b.ndim != 2:
        raise ValueError("CKA inputs must be flattened to [num_samples, features].")
    if activations_a.shape[0] != activations_b.shape[0]:
        raise ValueError("CKA inputs must have the same number of samples.")

    x = activations_a.float()
    y = activations_b.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    gram_x = x @ x.T
    gram_y = y @ y.T

    hsic = (gram_x * gram_y).sum()
    norm_x = torch.linalg.norm(gram_x)
    norm_y = torch.linalg.norm(gram_y)
    denominator = norm_x * norm_y
    if denominator <= eps:
        return 0.0

    value = (hsic / denominator).item()
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def compute_cka(activations_a, activations_b) -> dict[str, float] | float:
    """Compute linear CKA for one layer or matching activation dictionaries."""

    if isinstance(activations_a, dict) and isinstance(activations_b, dict):
        shared_layers = sorted(set(activations_a) & set(activations_b))
        if not shared_layers:
            raise ValueError("Activation dictionaries do not share any layers.")
        return {
            layer: linear_cka(activations_a[layer], activations_b[layer])
            for layer in shared_layers
        }

    return linear_cka(activations_a, activations_b)


def compute_client_cka(
    client_models,
    reference_loader,
    device=None,
    layers: tuple[str, ...] = DEFAULT_CKA_LAYERS,
) -> dict:
    """Compute pairwise client CKA matrices and average scores per layer."""

    client_models = list(client_models)
    if not client_models:
        raise ValueError("At least one client model is required for CKA.")

    activations = [
        collect_activations(model, reference_loader, device=device, layers=layers)
        for model in client_models
    ]

    matrices = {}
    averages = {}
    for layer in layers:
        matrix = torch.eye(len(client_models), dtype=torch.float32)
        for i in range(len(client_models)):
            for j in range(i + 1, len(client_models)):
                score = linear_cka(activations[i][layer], activations[j][layer])
                matrix[i, j] = score
                matrix[j, i] = score

        matrices[layer] = matrix
        averages[layer] = average_upper_triangle(matrix)

    return {
        "matrices": matrices,
        "average_cka": averages,
        "layers": tuple(layers),
        "num_clients": len(client_models),
        "num_reference_samples": next(iter(activations[0].values())).shape[0],
    }


def average_upper_triangle(matrix: torch.Tensor) -> float:
    """Average the strict upper triangle of a square CKA matrix."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square pairwise CKA matrix.")
    if matrix.shape[0] < 2:
        return 1.0

    indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
    value = matrix[indices[0], indices[1]].mean().item()
    if not math.isfinite(value):
        return 0.0
    return float(value)


def cka_to_rows(cka_result: dict, round_id: int | None = None) -> list[dict]:
    """Convert CKA matrices to tidy CSV rows."""

    rows = []
    for layer, matrix in cka_result["matrices"].items():
        average = cka_result["average_cka"][layer]
        for client_i in range(matrix.shape[0]):
            for client_j in range(matrix.shape[1]):
                row = {
                    "layer": layer,
                    "client_i": client_i,
                    "client_j": client_j,
                    "cka": float(matrix[client_i, client_j].item()),
                    "average_layer_cka": average,
                    "num_reference_samples": cka_result["num_reference_samples"],
                }
                if round_id is not None:
                    row = {"round": round_id, **row}
                rows.append(row)
    return rows


def save_cka_csv(rows: list[dict], path: Path) -> Path:
    """Save CKA rows to a CSV file."""

    if not rows:
        raise ValueError("Cannot save empty CKA results.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
