"""Evaluation helpers for experiments."""

import torch
from torch import nn


def evaluate_model(model, dataloader, device=None) -> dict[str, float]:
    """Evaluate a model on a dataloader.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation data loader.

    Returns:
        A dictionary with average cross-entropy loss and accuracy.
    """

    if device is None:
        device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    if was_training:
        model.train()

    if total_samples == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
