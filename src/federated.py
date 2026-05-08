"""Standard FedAvg training for simulated federated learning."""

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .evaluate import evaluate_model
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


def train_client(global_model, train_loader, config: FederatedConfig, device):
    """Train one client from the current global model."""

    client_model = copy.deepcopy(global_model).to(device)
    client_model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(client_model.parameters(), lr=config.lr)

    total_loss = 0.0
    total_samples = 0

    for _ in range(config.local_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = client_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Client dataloader produced no samples.")

    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in client_model.state_dict().items()
    }
    return {
        "state_dict": state_dict,
        "num_samples": len(train_loader.dataset),
        "train_loss": total_loss / total_samples,
    }


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


def run_federated_round(global_model, client_loaders, config: FederatedConfig, device):
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
        train_client(global_model, client_loader, config, device)
        for client_loader in client_loaders
    ]
    global_model.load_state_dict(aggregate_weights(client_updates))

    total_samples = sum(update["num_samples"] for update in client_updates)
    return sum(
        update["train_loss"] * update["num_samples"] / total_samples
        for update in client_updates
    )


def run_fedavg(global_model, client_loaders, test_loader, config: FederatedConfig):
    """Run standard FedAvg and return one metrics dictionary per round."""

    _validate_config(config, client_loaders)
    seed_everything(config.seed)

    device = _resolve_device(config.device)
    global_model.to(device)

    logs = []
    for round_id in range(1, config.rounds + 1):
        avg_train_loss = run_federated_round(
            global_model=global_model,
            client_loaders=client_loaders,
            config=config,
            device=device,
        )
        metrics = evaluate_model(global_model, test_loader, device=device)
        row = {
            "round": round_id,
            "test_accuracy": metrics["accuracy"],
            "test_loss": metrics["loss"],
            "avg_train_loss": avg_train_loss,
        }
        logs.append(row)

        print(
            f"Round {round_id:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"test_loss={metrics['loss']:.4f} | "
            f"test_acc={metrics['accuracy']:.4f}"
        )

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
