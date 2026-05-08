"""Dynamic sparse training mask updates."""

from dataclasses import dataclass

import torch

from .sparsity import apply_masks, sparsity_summary


@dataclass(frozen=True)
class DSTConfig:
    """Configuration for FedDST/RigL-style mask updates.

    ``layer_sparsities`` can be supplied by adaptive methods such as
    CKA-guided FedDST to change the per-layer active-weight budget.
    """

    mask_update_interval: int = 1
    prune_fraction: float = 0.1
    layer_sparsities: dict[str, float] | None = None


def update_sparse_topology(
    model,
    masks: dict[str, torch.Tensor],
    gradient_scores: dict[str, torch.Tensor],
    config: DSTConfig,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Update masks with magnitude pruning and gradient-based regrowth.

    Args:
        model: Sparse model after aggregation.
        masks: Current binary masks keyed by parameter name.
        gradient_scores: Accumulated absolute gradient scores keyed by name.
        config: Dynamic sparse training settings.

    Returns:
        Updated masks and a dictionary of topology-change statistics.
    """

    _validate_config(config)
    parameters = dict(model.named_parameters())

    updated_masks = {}
    layer_stats = {}
    total_pruned = 0
    total_regrown = 0
    total_changed = 0

    for name, mask in masks.items():
        if name not in parameters:
            raise KeyError(f"Mask key '{name}' was not found in model parameters.")

        weight = parameters[name].detach().abs().cpu().flatten()
        gradient = gradient_scores.get(name)
        if gradient is None:
            gradient = torch.zeros_like(mask, dtype=torch.float32)
        gradient = gradient.detach().abs().cpu().flatten()

        old_mask = mask.detach().cpu().flatten().bool()
        target_active = _target_active_count(name, mask, config)
        new_mask, stats = _update_layer_mask(
            old_mask=old_mask,
            weight_scores=weight,
            gradient_scores=gradient,
            target_active=target_active,
            prune_fraction=config.prune_fraction,
        )

        updated_mask = new_mask.view_as(mask).to(dtype=mask.dtype)
        updated_masks[name] = updated_mask

        layer_stats[name] = stats
        total_pruned += stats["pruned"]
        total_regrown += stats["regrown"]
        total_changed += stats["changed"]

    apply_masks(model, updated_masks)
    summary = sparsity_summary(model, updated_masks)

    return updated_masks, {
        "pruned": total_pruned,
        "regrown": total_regrown,
        "mask_changes": total_changed,
        "layer_stats": layer_stats,
        "total_sparsity": summary["total_sparsity"],
        "layer_sparsity": summary["layer_sparsity"],
        "active_params": summary["active_params"],
        "total_params": summary["total_params"],
    }


def _update_layer_mask(
    old_mask: torch.Tensor,
    weight_scores: torch.Tensor,
    gradient_scores: torch.Tensor,
    target_active: int,
    prune_fraction: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Update one flattened layer mask while preserving target active count."""

    total_params = old_mask.numel()
    current_active = int(old_mask.sum().item())
    target_active = max(0, min(target_active, total_params))

    if target_active == 0:
        new_mask = torch.zeros_like(old_mask)
        changed = int((new_mask != old_mask).sum().item())
        return new_mask, {"pruned": current_active, "regrown": 0, "changed": changed}

    active_indices = torch.nonzero(old_mask, as_tuple=False).flatten()
    stable_prune = int(round(current_active * prune_fraction))
    if current_active > target_active:
        stable_prune = max(stable_prune, current_active - target_active)
    if current_active <= target_active:
        stable_prune = min(stable_prune, current_active)
    prune_count = min(stable_prune, current_active)

    new_mask = old_mask.clone()
    pruned_indices = torch.empty(0, dtype=torch.long)
    if prune_count > 0:
        prune_positions = torch.topk(
            weight_scores[active_indices],
            k=prune_count,
            largest=False,
        ).indices
        pruned_indices = active_indices[prune_positions]
        new_mask[pruned_indices] = False

    active_after_prune = int(new_mask.sum().item())
    regrow_count = max(0, target_active - active_after_prune)

    original_inactive = torch.nonzero(~old_mask, as_tuple=False).flatten()
    inactive_after_prune = torch.nonzero(~new_mask, as_tuple=False).flatten()
    regrow_candidates = original_inactive
    if regrow_candidates.numel() < regrow_count:
        regrow_candidates = inactive_after_prune
    regrow_count = min(regrow_count, regrow_candidates.numel())

    regrown_indices = torch.empty(0, dtype=torch.long)
    if regrow_count > 0:
        regrow_positions = torch.topk(
            gradient_scores[regrow_candidates],
            k=regrow_count,
            largest=True,
        ).indices
        regrown_indices = regrow_candidates[regrow_positions]
        new_mask[regrown_indices] = True

    if int(new_mask.sum().item()) != target_active:
        new_mask = _repair_active_count(new_mask, gradient_scores, target_active)

    changed = int((new_mask != old_mask).sum().item())
    return new_mask, {
        "pruned": int(pruned_indices.numel()),
        "regrown": int(regrown_indices.numel()),
        "changed": changed,
    }


def _repair_active_count(
    mask: torch.Tensor,
    gradient_scores: torch.Tensor,
    target_active: int,
) -> torch.Tensor:
    """Repair rare active-count drift caused by small or dense layers."""

    current_active = int(mask.sum().item())
    if current_active == target_active:
        return mask

    repaired = mask.clone()
    if current_active < target_active:
        needed = target_active - current_active
        inactive = torch.nonzero(~repaired, as_tuple=False).flatten()
        needed = min(needed, inactive.numel())
        if needed > 0:
            add_positions = torch.topk(
                gradient_scores[inactive],
                k=needed,
                largest=True,
            ).indices
            repaired[inactive[add_positions]] = True
    else:
        extra = current_active - target_active
        active = torch.nonzero(repaired, as_tuple=False).flatten()
        extra = min(extra, active.numel())
        if extra > 0:
            remove_positions = torch.topk(
                gradient_scores[active],
                k=extra,
                largest=False,
            ).indices
            repaired[active[remove_positions]] = False

    return repaired


def _target_active_count(name: str, mask: torch.Tensor, config: DSTConfig) -> int:
    """Return the active count this layer should keep after a mask update."""

    if config.layer_sparsities and name in config.layer_sparsities:
        layer_sparsity = config.layer_sparsities[name]
        if not 0.0 <= layer_sparsity < 1.0:
            raise ValueError(f"Layer sparsity for '{name}' must be in [0.0, 1.0).")
        return round(mask.numel() * (1.0 - layer_sparsity))

    return int(mask.detach().count_nonzero().item())


def _validate_config(config: DSTConfig) -> None:
    """Validate dynamic sparse training settings."""

    if config.mask_update_interval <= 0:
        raise ValueError("mask_update_interval must be positive.")
    if not 0.0 <= config.prune_fraction <= 1.0:
        raise ValueError("prune_fraction must be in [0.0, 1.0].")
