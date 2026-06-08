"""Evaluation: accuracy on a held-out split."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def accuracy(
    model: nn.Module, dataloader: DataLoader, device: torch.device | None = None
) -> float:
    device = device or next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)
