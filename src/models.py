"""Model definitions for image classification experiments."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model construction."""

    name: str = "small_cnn"
    dataset: str = "mnist"
    num_classes: int = 10


class SmallCNN(nn.Module):
    """A compact CNN for MNIST and Fashion-MNIST.

    The network is intentionally small so it is easy to inspect and quick to
    use in simulated federated learning experiments.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        """Create the convolutional and fully connected layers."""

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run a forward pass.

        Args:
            x: Batch of grayscale images with shape ``[batch, 1, 28, 28]``.
            return_activations: If true, also return selected layer activations.

        Returns:
            Logits, or ``(logits, activations)`` when requested. Activations
            are recorded after the nonlinearity for ``conv1``, ``conv2``, and
            ``fc1``.
        """

        activations: dict[str, torch.Tensor] = {}

        x = F.relu(self.conv1(x))
        activations["conv1"] = x
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv2(x))
        activations["conv2"] = x
        x = F.max_pool2d(x, kernel_size=2)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        activations["fc1"] = x
        logits = self.fc2(x)

        if return_activations:
            return logits, activations
        return logits


class SmallCIFARCNN(nn.Module):
    """A compact CNN for CIFAR-10 experiments.

    The network is intentionally modest for simulated federated experiments,
    but deeper than the MNIST CNN so CKA can inspect more representation levels.
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Create convolutional blocks and the classifier."""

        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run a forward pass for CIFAR-10 images.

        Args:
            x: Batch of RGB images with shape ``[batch, 3, 32, 32]``.
            return_activations: If true, also return selected layer activations.

        Returns:
            Logits, or ``(logits, activations)`` when requested. Activations
            are recorded after the nonlinearity for ``conv1``, ``conv2``,
            ``conv3``, and ``fc1``.
        """

        activations: dict[str, torch.Tensor] = {}

        x = F.relu(self.conv1(x))
        activations["conv1"] = x
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv2(x))
        activations["conv2"] = x
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv3(x))
        activations["conv3"] = x
        x = F.max_pool2d(x, kernel_size=2)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        activations["fc1"] = x
        logits = self.fc2(x)

        if return_activations:
            return logits, activations
        return logits


def get_model(model_name: str = "small_cnn", dataset: str = "mnist") -> nn.Module:
    """Return a model by name for a supported dataset."""

    normalized_model = model_name.lower().replace("-", "_")
    normalized_dataset = dataset.lower().replace("-", "_")

    if normalized_dataset not in {"mnist", "fashion_mnist", "cifar10"}:
        raise ValueError(
            "Unsupported dataset "
            f"'{dataset}'. Expected 'mnist', 'fashion_mnist', or 'cifar10'."
        )

    if normalized_dataset == "cifar10":
        if normalized_model in {"cifar_cnn", "small_cifar_cnn"}:
            return SmallCIFARCNN(num_classes=10)
        raise ValueError(
            "Unsupported model "
            f"'{model_name}' for CIFAR-10. Available models: cifar_cnn."
        )

    if normalized_model in {"small_cnn", "mnist_cnn", "cnn"}:
        return SmallCNN(in_channels=1, num_classes=10)

    raise ValueError(
        f"Unsupported model '{model_name}'. Available models: small_cnn."
    )


def build_model(config: ModelConfig):
    """Build and return a PyTorch model from a config object.

    Args:
        config: Model selection and shape settings.
    """

    return get_model(config.name, config.dataset)
