"""Standard FedAvg training for simulated federated learning."""

import copy
from dataclasses import dataclass, replace

import torch
from torch import nn

from .cka import (
    DEFAULT_CKA_LAYERS,
    average_cka_scores,
    cka_to_rows,
    compute_client_cka,
    save_cka_csv,
)
from .dst import DSTConfig, update_sparse_topology
from .evaluate import evaluate_model
from .sparsity import (
    SparsityConfig,
    apply_masks,
    cka_scores_to_layer_sparsities,
    create_masks,
    format_layer_sparsity,
    format_layer_values,
    sparsity_summary,
)
from .utils import seed_everything


@dataclass(frozen=True)
class FederatedConfig:
    """Configuration for a simulated federated run."""

    num_clients: int = 10
    rounds: int = 3
    local_epochs: int = 1
    lr: float = 0.01
    seed: int = 0
    device: str = "auto"


def train_client(
    global_model,
    train_loader,
    config: FederatedConfig,
    device,
    masks: dict[str, torch.Tensor] | None = None,
    collect_grad_scores: bool = False,
    return_model: bool = False,
):
    """Train one client from the current global model."""

    client_model = copy.deepcopy(global_model).to(device)
    apply_masks(client_model, masks or {})
    client_model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(client_model.parameters(), lr=config.lr)

    total_loss = 0.0
    total_samples = 0
    grad_scores = _empty_gradient_scores(masks) if collect_grad_scores else None

    for _ in range(config.local_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = client_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_scores is not None:
                _accumulate_gradient_scores(client_model, grad_scores, labels.size(0))
            optimizer.step()
            apply_masks(client_model, masks or {})

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Client dataloader produced no samples.")

    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in client_model.state_dict().items()
    }
    result = {
        "state_dict": state_dict,
        "num_samples": len(train_loader.dataset),
        "train_loss": total_loss / total_samples,
    }
    if grad_scores is not None:
        result["grad_scores"] = grad_scores
    if return_model:
        result["model"] = client_model.cpu()
    return result


def aggregate_weights(client_updates: list[dict]) -> dict[str, torch.Tensor]:
    """Aggregate client weights with sample-count weighting."""

    if not client_updates:
        raise ValueError("Cannot aggregate an empty list of client updates.")

    total_samples = sum(update["num_samples"] for update in client_updates)
    if total_samples <= 0:
        raise ValueError("Total client sample count must be positive.")

    first_state = client_updates[0]["state_dict"]
    aggregated = {}

    for name, first_tensor in first_state.items():
        if torch.is_floating_point(first_tensor):
            value = torch.zeros_like(first_tensor)
            for update in client_updates:
                weight = update["num_samples"] / total_samples
                value += update["state_dict"][name] * weight
            aggregated[name] = value
        else:
            aggregated[name] = first_tensor.clone()

    return aggregated


def run_federated_round(
    global_model,
    client_loaders,
    config: FederatedConfig,
    device,
    masks: dict[str, torch.Tensor] | None = None,
    collect_grad_scores: bool = False,
    reference_loader=None,
    cka_layers: tuple[str, ...] = DEFAULT_CKA_LAYERS,
):
    """Run one simulated communication round.

    Args:
        global_model: Global model before the round.
        client_loaders: Per-client training dataloaders.
        config: Federated simulation settings.
        device: Torch device used for local client training.

    Returns:
        Average client training loss for the round.
    """

    client_updates = [
        train_client(
            global_model,
            client_loader,
            config,
            device,
            masks=masks,
            collect_grad_scores=collect_grad_scores,
            return_model=reference_loader is not None,
        )
        for client_loader in client_loaders
    ]
    cka_result = None
    if reference_loader is not None:
        cka_result = compute_client_cka(
            [update["model"] for update in client_updates],
            reference_loader,
            device=device,
            layers=cka_layers,
        )

    global_model.load_state_dict(aggregate_weights(client_updates))
    apply_masks(global_model, masks or {})

    total_samples = sum(update["num_samples"] for update in client_updates)
    avg_train_loss = sum(
        update["train_loss"] * update["num_samples"] / total_samples
        for update in client_updates
    )
    if collect_grad_scores:
        grad_scores = aggregate_gradient_scores(client_updates, masks or {})
        if cka_result is not None:
            return avg_train_loss, grad_scores, cka_result
        return avg_train_loss, grad_scores
    if cka_result is not None:
        return avg_train_loss, cka_result
    return avg_train_loss


def run_fedavg(
    global_model,
    client_loaders,
    test_loader,
    config: FederatedConfig,
    reference_loader=None,
    cka_log_path=None,
    cka_layers: tuple[str, ...] = DEFAULT_CKA_LAYERS,
):
    """Run standard FedAvg and return one metrics dictionary per round."""

    _validate_config(config, client_loaders)
    seed_everything(config.seed)

    device = _resolve_device(config.device)
    global_model.to(device)

    logs = []
    cka_rows = []
    for round_id in range(1, config.rounds + 1):
        round_result = run_federated_round(
            global_model=global_model,
            client_loaders=client_loaders,
            config=config,
            device=device,
            reference_loader=reference_loader,
            cka_layers=cka_layers,
        )
        if reference_loader is not None:
            avg_train_loss, cka_result = round_result
        else:
            avg_train_loss = round_result
            cka_result = None

        metrics = evaluate_model(global_model, test_loader, device=device)
        row = {
            "round": round_id,
            "test_accuracy": metrics["accuracy"],
            "test_loss": metrics["loss"],
            "avg_train_loss": avg_train_loss,
        }
        _add_cka_average_columns(row, cka_result)
        logs.append(row)
        _extend_cka_rows(cka_rows, cka_result, round_id)

        print(
            f"Round {round_id:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"test_loss={metrics['loss']:.4f} | "
            f"test_acc={metrics['accuracy']:.4f}"
        )

    _save_cka_rows_if_requested(cka_rows, cka_log_path)
    return logs


def run_sparse_fedavg(
    global_model,
    client_loaders,
    test_loader,
    config: FederatedConfig,
    sparsity_config: SparsityConfig,
    reference_loader=None,
    cka_log_path=None,
    cka_layers: tuple[str, ...] = DEFAULT_CKA_LAYERS,
):
    """Run FedAvg with a fixed unstructured sparsity mask."""

    _validate_config(config, client_loaders)
    seed_everything(config.seed)

    device = _resolve_device(config.device)
    global_model.to(device)

    masks = create_masks(global_model, sparsity_config)
    apply_masks(global_model, masks)

    initial_summary = sparsity_summary(global_model, masks)
    print(
        "Initial sparsity | "
        f"total={initial_summary['total_sparsity']:.4f} | "
        f"active={initial_summary['active_params']}/{initial_summary['total_params']}"
    )
    print(
        "Layer sparsity | "
        f"{format_layer_sparsity(initial_summary['layer_sparsity'])}"
    )

    logs = []
    cka_rows = []
    for round_id in range(1, config.rounds + 1):
        round_result = run_federated_round(
            global_model=global_model,
            client_loaders=client_loaders,
            config=config,
            device=device,
            masks=masks,
            reference_loader=reference_loader,
            cka_layers=cka_layers,
        )
        if reference_loader is not None:
            avg_train_loss, cka_result = round_result
        else:
            avg_train_loss = round_result
            cka_result = None

        metrics = evaluate_model(global_model, test_loader, device=device)
        summary = sparsity_summary(global_model, masks)
        row = {
            "round": round_id,
            "test_accuracy": metrics["accuracy"],
            "test_loss": metrics["loss"],
            "avg_train_loss": avg_train_loss,
            "total_sparsity": summary["total_sparsity"],
            "active_params": summary["active_params"],
            "total_params": summary["total_params"],
            "layer_sparsity": format_layer_sparsity(summary["layer_sparsity"]),
        }
        row.update(_layer_sparsity_columns(summary["layer_sparsity"]))
        _add_cka_average_columns(row, cka_result)
        logs.append(row)
        _extend_cka_rows(cka_rows, cka_result, round_id)

        print(
            f"Round {round_id:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"test_loss={metrics['loss']:.4f} | "
            f"test_acc={metrics['accuracy']:.4f} | "
            f"sparsity={summary['total_sparsity']:.4f}"
        )

    _save_cka_rows_if_requested(cka_rows, cka_log_path)
    return logs


def run_feddst(
    global_model,
    client_loaders,
    test_loader,
    config: FederatedConfig,
    sparsity_config: SparsityConfig,
    dst_config: DSTConfig,
    reference_loader=None,
    cka_log_path=None,
    cka_layers: tuple[str, ...] = DEFAULT_CKA_LAYERS,
):
    """Run a simplified FedDST/RigL-style dynamic sparse baseline."""

    _validate_config(config, client_loaders)
    seed_everything(config.seed)

    device = _resolve_device(config.device)
    global_model.to(device)

    masks = create_masks(global_model, sparsity_config)
    apply_masks(global_model, masks)

    initial_summary = sparsity_summary(global_model, masks)
    print(
        "Initial sparsity | "
        f"total={initial_summary['total_sparsity']:.4f} | "
        f"active={initial_summary['active_params']}/{initial_summary['total_params']}"
    )
    print(
        "Layer sparsity | "
        f"{format_layer_sparsity(initial_summary['layer_sparsity'])}"
    )

    logs = []
    cka_rows = []
    for round_id in range(1, config.rounds + 1):
        should_update_mask = round_id % dst_config.mask_update_interval == 0
        round_result = run_federated_round(
            global_model=global_model,
            client_loaders=client_loaders,
            config=config,
            device=device,
            masks=masks,
            collect_grad_scores=should_update_mask,
            reference_loader=reference_loader,
            cka_layers=cka_layers,
        )

        update_stats = _empty_dst_stats(global_model, masks)
        cka_result = None
        if should_update_mask and reference_loader is not None:
            avg_train_loss, grad_scores, cka_result = round_result
            masks, update_stats = update_sparse_topology(
                global_model,
                masks,
                grad_scores,
                dst_config,
            )
        elif should_update_mask:
            avg_train_loss, grad_scores = round_result
            masks, update_stats = update_sparse_topology(
                global_model,
                masks,
                grad_scores,
                dst_config,
            )
        elif reference_loader is not None:
            avg_train_loss, cka_result = round_result
        else:
            avg_train_loss = round_result

        metrics = evaluate_model(global_model, test_loader, device=device)
        summary = sparsity_summary(global_model, masks)
        row = {
            "round": round_id,
            "test_accuracy": metrics["accuracy"],
            "test_loss": metrics["loss"],
            "avg_train_loss": avg_train_loss,
            "pruned_weights": update_stats["pruned"],
            "regrown_weights": update_stats["regrown"],
            "mask_changes": update_stats["mask_changes"],
            "total_sparsity": summary["total_sparsity"],
            "active_params": summary["active_params"],
            "total_params": summary["total_params"],
            "layer_sparsity": format_layer_sparsity(summary["layer_sparsity"]),
        }
        row.update(_layer_sparsity_columns(summary["layer_sparsity"]))
        _add_cka_average_columns(row, cka_result)
        logs.append(row)
        _extend_cka_rows(cka_rows, cka_result, round_id)

        print(
            f"Round {round_id:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"test_loss={metrics['loss']:.4f} | "
            f"test_acc={metrics['accuracy']:.4f} | "
            f"sparsity={summary['total_sparsity']:.4f} | "
            f"pruned={update_stats['pruned']} | "
            f"regrown={update_stats['regrown']} | "
            f"mask_changes={update_stats['mask_changes']}"
        )

    _save_cka_rows_if_requested(cka_rows, cka_log_path)
    return logs


def run_cka_feddst(
    global_model,
    client_loaders,
    test_loader,
    reference_loader,
    config: FederatedConfig,
    sparsity_config: SparsityConfig,
    dst_config: DSTConfig,
    cka_interval: int = 1,
    cka_target_strength: float = 0.5,
    cka_layers: tuple[str, ...] = DEFAULT_CKA_LAYERS,
    cka_min_sparsity: float = 0.0,
    cka_max_sparsity: float = 0.99,
    cka_log_path=None,
):
    """Run CKA-guided FedDST with adaptive layer-wise sparsity targets."""

    _validate_config(config, client_loaders)
    _validate_cka_config(
        cka_interval,
        cka_target_strength,
        cka_min_sparsity,
        cka_max_sparsity,
    )
    seed_everything(config.seed)

    device = _resolve_device(config.device)
    global_model.to(device)

    masks = create_masks(global_model, sparsity_config)
    apply_masks(global_model, masks)

    initial_summary = sparsity_summary(global_model, masks)
    print(
        "Initial sparsity | "
        f"total={initial_summary['total_sparsity']:.4f} | "
        f"active={initial_summary['active_params']}/{initial_summary['total_params']}"
    )
    print(
        "Layer sparsity | "
        f"{format_layer_sparsity(initial_summary['layer_sparsity'])}"
    )

    logs = []
    cka_rows = []
    latest_cka_scores = {}
    layer_targets = {}
    cka_layer_names = cka_layers
    sparse_layer_names = tuple(masks)

    for round_id in range(1, config.rounds + 1):
        should_update_mask = round_id % dst_config.mask_update_interval == 0
        should_compute_cka = round_id % cka_interval == 0
        round_result = run_federated_round(
            global_model=global_model,
            client_loaders=client_loaders,
            config=config,
            device=device,
            masks=masks,
            collect_grad_scores=should_update_mask,
            reference_loader=reference_loader if should_compute_cka else None,
            cka_layers=cka_layers,
        )

        cka_result = None
        if should_update_mask and should_compute_cka:
            avg_train_loss, grad_scores, cka_result = round_result
        elif should_update_mask:
            avg_train_loss, grad_scores = round_result
        elif should_compute_cka:
            avg_train_loss, cka_result = round_result
            grad_scores = None
        else:
            avg_train_loss = round_result
            grad_scores = None

        if cka_result is not None:
            latest_cka_scores = average_cka_scores(cka_result)
            layer_targets = cka_scores_to_layer_sparsities(
                masks=masks,
                cka_scores=latest_cka_scores,
                base_sparsity=sparsity_config.target_sparsity,
                strength=cka_target_strength,
                min_sparsity=cka_min_sparsity,
                max_sparsity=cka_max_sparsity,
            )
            _extend_cka_rows(cka_rows, cka_result, round_id)

        update_stats = _empty_dst_stats(global_model, masks)
        if should_update_mask:
            guided_dst_config = replace(
                dst_config,
                layer_sparsities=layer_targets or None,
            )
            masks, update_stats = update_sparse_topology(
                global_model,
                masks,
                grad_scores or {},
                guided_dst_config,
            )

        metrics = evaluate_model(global_model, test_loader, device=device)
        summary = sparsity_summary(global_model, masks)
        row = {
            "round": round_id,
            "test_accuracy": metrics["accuracy"],
            "test_loss": metrics["loss"],
            "avg_train_loss": avg_train_loss,
            "cka_computed": int(should_compute_cka),
            "pruned_weights": update_stats["pruned"],
            "regrown_weights": update_stats["regrown"],
            "mask_changes": update_stats["mask_changes"],
            "total_sparsity": summary["total_sparsity"],
            "active_params": summary["active_params"],
            "total_params": summary["total_params"],
            "layer_sparsity": format_layer_sparsity(summary["layer_sparsity"]),
            "layer_cka": format_layer_values(latest_cka_scores),
            "layer_target_sparsity": format_layer_sparsity(layer_targets),
        }
        row.update(_layer_sparsity_columns(summary["layer_sparsity"]))
        row.update(_layer_value_columns("cka", latest_cka_scores, cka_layer_names))
        row.update(
            _layer_value_columns(
                "target_sparsity",
                layer_targets,
                sparse_layer_names,
            )
        )
        logs.append(row)

        print(
            f"Round {round_id:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"test_loss={metrics['loss']:.4f} | "
            f"test_acc={metrics['accuracy']:.4f} | "
            f"sparsity={summary['total_sparsity']:.4f} | "
            f"cka={int(should_compute_cka)} | "
            f"targets={format_layer_sparsity(layer_targets)}"
        )

    _save_cka_rows_if_requested(cka_rows, cka_log_path)
    return logs


def _resolve_device(device_name: str):
    """Resolve 'auto' to CUDA when available, otherwise CPU."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _validate_config(config: FederatedConfig, client_loaders) -> None:
    """Validate FedAvg settings before training starts."""

    if config.num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if config.rounds <= 0:
        raise ValueError("rounds must be positive.")
    if config.local_epochs <= 0:
        raise ValueError("local_epochs must be positive.")
    if config.lr <= 0:
        raise ValueError("lr must be positive.")
    if len(client_loaders) != config.num_clients:
        raise ValueError(
            f"Expected {config.num_clients} client loaders, got {len(client_loaders)}."
        )


def _validate_cka_config(
    cka_interval: int,
    cka_target_strength: float,
    cka_min_sparsity: float = 0.0,
    cka_max_sparsity: float = 0.99,
) -> None:
    """Validate CKA-guided sparse training settings."""

    if cka_interval <= 0:
        raise ValueError("cka_interval must be positive.")
    if cka_target_strength < 0.0:
        raise ValueError("cka_target_strength must be non-negative.")
    if not 0.0 <= cka_min_sparsity <= cka_max_sparsity < 1.0:
        raise ValueError(
            "Expected 0.0 <= cka_min_sparsity <= cka_max_sparsity < 1.0."
        )


def _layer_sparsity_columns(layer_sparsity: dict[str, float]) -> dict[str, float]:
    """Return CSV-friendly per-layer sparsity columns."""

    return {
        f"sparsity_{name.replace('.', '_')}": value
        for name, value in sorted(layer_sparsity.items())
    }


def _layer_value_columns(
    prefix: str,
    values: dict[str, float],
    names=None,
) -> dict[str, float | None]:
    """Return stable CSV columns for layer-wise values."""

    if names is None:
        names = sorted(values)
    return {
        f"{prefix}_{name.replace('.', '_')}": values.get(name)
        for name in names
    }


def _add_cka_average_columns(row: dict, cka_result: dict | None) -> None:
    """Add average layer CKA values to a round log row."""

    if cka_result is None:
        return
    for layer, value in cka_result["average_cka"].items():
        row[f"cka_avg_{layer}"] = value


def _extend_cka_rows(
    rows: list[dict],
    cka_result: dict | None,
    round_id: int,
) -> None:
    """Append pairwise CKA matrix rows for a round."""

    if cka_result is not None:
        rows.extend(cka_to_rows(cka_result, round_id=round_id))


def _save_cka_rows_if_requested(rows: list[dict], cka_log_path) -> None:
    """Save pairwise CKA rows when a log path is provided."""

    if cka_log_path is not None and rows:
        save_cka_csv(rows, cka_log_path)


def _empty_gradient_scores(
    masks: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Create zero-filled gradient score buffers matching sparse masks."""

    return {
        name: torch.zeros_like(mask, dtype=torch.float32, device="cpu")
        for name, mask in (masks or {}).items()
    }


def _accumulate_gradient_scores(model, grad_scores, batch_size: int) -> None:
    """Accumulate absolute gradients for sparse mask regrowth."""

    for name, param in model.named_parameters():
        if name in grad_scores and param.grad is not None:
            grad_scores[name] += param.grad.detach().abs().cpu() * batch_size


def aggregate_gradient_scores(
    client_updates: list[dict],
    masks: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Average client gradient scores by client sample count."""

    if not masks:
        return {}

    total_samples = sum(update["num_samples"] for update in client_updates)
    if total_samples <= 0:
        raise ValueError("Total client sample count must be positive.")

    aggregated = _empty_gradient_scores(masks)
    for update in client_updates:
        grad_scores = update.get("grad_scores")
        if grad_scores is None:
            continue
        weight = update["num_samples"] / total_samples
        for name in aggregated:
            aggregated[name] += grad_scores[name] * weight
    return aggregated


def _empty_dst_stats(global_model, masks: dict[str, torch.Tensor]) -> dict:
    """Return zero topology-change stats for rounds without mask updates."""

    summary = sparsity_summary(global_model, masks)
    return {
        "pruned": 0,
        "regrown": 0,
        "mask_changes": 0,
        "layer_stats": {},
        "total_sparsity": summary["total_sparsity"],
        "layer_sparsity": summary["layer_sparsity"],
        "active_params": summary["active_params"],
        "total_params": summary["total_params"],
    }
