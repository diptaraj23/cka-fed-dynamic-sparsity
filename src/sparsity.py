"""Fixed unstructured sparsity utilities."""

from dataclasses import dataclass
import math

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


def cka_scores_to_layer_sparsities(
    masks: dict[str, torch.Tensor],
    cka_scores: dict[str, float],
    base_sparsity: float,
    strength: float = 0.5,
    min_sparsity: float = 0.0,
    max_sparsity: float = 0.99,
) -> dict[str, float]:
    """Convert CKA scores into layer-wise sparsity targets.

    Layers with higher CKA receive more active weights, which means lower
    sparsity. Only layers with measured CKA scores receive adaptive targets;
    other sparse layers keep their existing active-count budget in FedDST.
    """

    if not masks:
        return {}
    if not 0.0 <= base_sparsity < 1.0:
        raise ValueError("base_sparsity must be in [0.0, 1.0).")
    if strength < 0.0:
        raise ValueError("strength must be non-negative.")
    if not 0.0 <= min_sparsity <= max_sparsity < 1.0:
        raise ValueError("Expected 0.0 <= min_sparsity <= max_sparsity < 1.0.")

    names = [
        name for name in masks
        if _activation_name(name) in cka_scores
    ]
    if not names:
        return {}

    param_counts = {name: masks[name].numel() for name in names}
    total_params = sum(param_counts.values())
    total_active = round(total_params * (1.0 - base_sparsity))
    min_active = {
        name: math.ceil(param_counts[name] * (1.0 - max_sparsity))
        for name in names
    }
    max_active = {
        name: math.floor(param_counts[name] * (1.0 - min_sparsity))
        for name in names
    }
    total_active = max(
        sum(min_active.values()),
        min(total_active, sum(max_active.values())),
    )

    scores = {
        name: float(cka_scores.get(_activation_name(name), 0.5))
        for name in names
    }
    mean_score = sum(scores.values()) / len(scores)
    raw_weights = {}
    for name in names:
        score_shift = scores[name] - mean_score
        raw_weights[name] = param_counts[name] * max(0.05, 1.0 + strength * score_shift)

    raw_total = sum(raw_weights.values())
    active_float = {
        name: total_active * raw_weights[name] / raw_total
        for name in names
    }
    active_counts = _rounded_active_budget(
        active_float,
        param_counts,
        total_active,
        min_active=min_active,
        max_active=max_active,
    )

    return {
        name: 1.0 - (active_counts[name] / param_counts[name])
        for name in names
    }


def format_layer_sparsity(layer_sparsity: dict[str, float]) -> str:
    """Format layer-wise sparsity for CSV logs."""

    return ";".join(
        f"{name}:{value:.6f}" for name, value in sorted(layer_sparsity.items())
    )


def format_layer_values(values: dict[str, float]) -> str:
    """Format layer-wise floating-point values for compact CSV logging."""

    return ";".join(f"{name}:{value:.6f}" for name, value in sorted(values.items()))


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


def _activation_name(parameter_name: str) -> str:
    """Map a parameter name such as 'conv1.weight' to activation name 'conv1'."""

    return parameter_name.rsplit(".", maxsplit=1)[0]


def _rounded_active_budget(
    active_float: dict[str, float],
    param_counts: dict[str, int],
    total_active: int,
    min_active: dict[str, int] | None = None,
    max_active: dict[str, int] | None = None,
) -> dict[str, int]:
    """Round active counts while preserving the requested global budget."""

    if min_active is None:
        min_active = {name: 0 for name in active_float}
    if max_active is None:
        max_active = param_counts

    active_counts = {
        name: max(min_active[name], min(max_active[name], int(active_float[name])))
        for name in active_float
    }
    current_total = sum(active_counts.values())

    while current_total < total_active:
        candidates = [
            name for name in active_counts if active_counts[name] < max_active[name]
        ]
        if not candidates:
            break
        name = max(candidates, key=lambda item: active_float[item] - active_counts[item])
        active_counts[name] += 1
        current_total += 1

    while current_total > total_active:
        candidates = [
            name for name in active_counts if active_counts[name] > min_active[name]
        ]
        if not candidates:
            break
        name = min(candidates, key=lambda item: active_float[item] - active_counts[item])
        active_counts[name] -= 1
        current_total -= 1

    return active_counts


def _validate_config(config: SparsityConfig) -> None:
    """Validate sparsity settings."""

    if not 0.0 <= config.target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0.0, 1.0).")
    if config.init_method not in {"magnitude", "random"}:
        raise ValueError("init_method must be either 'magnitude' or 'random'.")
