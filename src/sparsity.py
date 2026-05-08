"""Sparsity utilities for future experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SparsityConfig:
    """Configuration for sparse model operations."""

    target_sparsity: float = 0.0


def apply_sparsity(model, config: SparsityConfig):
    """Apply a sparsity rule to a model.

    Args:
        model: Model to sparsify.
        config: Sparsity settings.

    Raises:
        NotImplementedError: Sparsity logic is not implemented yet.
    """

    raise NotImplementedError("Sparsity methods are not implemented in this scaffold.")
