"""Fixed unstructured sparsity utilities."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparsityConfig:
    """Configuration for sparse model operations."""

    target_sparsity: float = 0.0
    init_method: str = "random"
    seed: int = 0


def create_masks(model, config: SparsityConfig) -> dict[str, torch.Tensor]:
    """Create global unstructured binary masks for trainable weight tensors.

    Args:
        model: Model whose trainable weights will be masked.
        config: Target sparsity and initialization rule.

    Returns:
        A dictionary mapping parameter names to CPU binary masks.
    """

    _validate_config(config)
    named_weights = list(maskable_parameters(model))
    if not named_weights:
        return {}

    total_params = sum(param.numel() for _, param in named_weights)
    num_pruned = round(total_params * config.target_sparsity)
    num_active = total_params - num_pruned

    if num_active == total_params:
        return {
            name: torch.ones_like(param, dtype=param.dtype, device="cpu")
            for name, param in named_weights
        }
    if num_active == 0:
        return {
            name: torch.zeros_like(param, dtype=param.dtype, device="cpu")
            for name, param in named_weights
        }

    scores = _global_scores(named_weights, config)
    active_indices = torch.topk(scores, k=num_active, largest=True).indices
    flat_mask = torch.zeros(total_params, dtype=torch.float32)
    flat_mask[active_indices] = 1.0

    masks = {}
    offset = 0
    for name, param in named_weights:
        numel = param.numel()
        mask = flat_mask[offset : offset + numel].view_as(param)
        masks[name] = mask.to(dtype=param.dtype)
        offset += numel
    return masks


def apply_masks(model, masks: dict[str, torch.Tensor]) -> None:
    """Apply binary masks in-place to model parameters."""

    if not masks:
        return

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(device=param.device, dtype=param.dtype))


def apply_sparsity(model, config: SparsityConfig) -> dict[str, torch.Tensor]:
    """Create and apply fixed masks to a model."""

    masks = create_masks(model, config)
    apply_masks(model, masks)
    return masks


def sparsity_summary(model, masks: dict[str, torch.Tensor] | None = None) -> dict:
    """Compute total, layer-wise, and active-parameter sparsity statistics."""

    if masks:
        layer_sparsity = {
            name: _tensor_sparsity(mask)
            for name, mask in masks.items()
        }
        total_params = sum(mask.numel() for mask in masks.values())
        active_params = sum(int(mask.count_nonzero().item()) for mask in masks.values())
    else:
        layer_sparsity = {
            name: _tensor_sparsity(param.detach())
            for name, param in maskable_parameters(model)
        }
        total_params = sum(param.numel() for _, param in maskable_parameters(model))
        active_params = sum(
            int(param.detach().count_nonzero().item())
            for _, param in maskable_parameters(model)
        )

    total_sparsity = 0.0
    if total_params > 0:
        total_sparsity = 1.0 - (active_params / total_params)

    return {
        "total_sparsity": total_sparsity,
        "layer_sparsity": layer_sparsity,
        "active_params": active_params,
        "total_params": total_params,
    }


def layer_active_counts(masks: dict[str, torch.Tensor]) -> dict[str, int]:
    """Return the number of active weights in each sparse layer mask."""

    return {
        name: int(mask.detach().count_nonzero().item())
        for name, mask in masks.items()
    }


def format_layer_sparsity(layer_sparsity: dict[str, float]) -> str:
    """Format layer-wise sparsity for CSV logs."""

    return ";".join(
        f"{name}:{value:.6f}" for name, value in sorted(layer_sparsity.items())
    )


def maskable_parameters(model):
    """Yield trainable non-bias weight tensors that should be sparsified."""

    for name, param in model.named_parameters():
        if param.requires_grad and name.endswith(".weight") and param.ndim > 1:
            yield name, param


def _global_scores(named_weights, config: SparsityConfig) -> torch.Tensor:
    """Return one score per maskable parameter entry."""

    if config.init_method == "magnitude":
        return torch.cat(
            [param.detach().abs().cpu().flatten() for _, param in named_weights]
        )

    if config.init_method == "random":
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        return torch.cat(
            [
                torch.rand(param.numel(), generator=generator)
                for _, param in named_weights
            ]
        )

    raise ValueError("init_method must be either 'magnitude' or 'random'.")


def _tensor_sparsity(tensor: torch.Tensor) -> float:
    """Return the fraction of zeros in a tensor."""

    if tensor.numel() == 0:
        return 0.0
    active = tensor.detach().count_nonzero().item()
    return 1.0 - (active / tensor.numel())


def _validate_config(config: SparsityConfig) -> None:
    """Validate sparsity settings."""

    if not 0.0 <= config.target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0.0, 1.0).")
    if config.init_method not in {"magnitude", "random"}:
        raise ValueError("init_method must be either 'magnitude' or 'random'.")
