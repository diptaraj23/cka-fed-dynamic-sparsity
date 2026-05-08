"""Federated learning simulation placeholders."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FederatedConfig:
    """Configuration for a simulated federated run."""

    num_clients: int = 10
    clients_per_round: int = 5
    rounds: int = 1


def run_federated_round(server_model, client_datasets, config: FederatedConfig):
    """Run one simulated communication round.

    Args:
        server_model: Global model before the round.
        client_datasets: Per-client training datasets.
        config: Federated simulation settings.

    Raises:
        NotImplementedError: Federated orchestration is not implemented yet.
    """

    raise NotImplementedError("Federated training is not implemented in this scaffold.")
