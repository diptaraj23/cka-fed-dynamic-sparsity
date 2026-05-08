"""Model definitions for MNIST experiments.

Architectures will be added here once the experiment method is specified.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model construction."""

    name: str = "mnist_cnn"
    num_classes: int = 10


def build_model(config: ModelConfig):
    """Build and return a PyTorch model.

    Args:
        config: Model selection and shape settings.

    Raises:
        NotImplementedError: No model architecture is defined yet.
    """

    raise NotImplementedError("Model definitions are not implemented in this scaffold.")
