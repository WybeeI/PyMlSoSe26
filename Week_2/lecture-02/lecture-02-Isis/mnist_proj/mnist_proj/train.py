"""Training loop as a library function.

No CLI parsing here — that lives in scripts/train.py. The function takes
already-built objects (model, dataloader, optimizer) and runs the loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mnist_proj.eval import accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one epoch. Returns the average loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    for images, labels in tqdm(dataloader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    epochs: int = 1,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> nn.Module:
    """Convenience: build optimizer + loss, run ``epochs`` epochs.

    If ``val_dataloader`` is given, report validation accuracy after each
    epoch — that's how we monitor for overfitting without touching the
    held-out test split.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        avg = train_one_epoch(model, dataloader, optimizer, loss_fn, device)
        msg = f"epoch {epoch + 1}/{epochs} — loss {avg:.4f}"
        if val_dataloader is not None:
            msg += f" — val acc {accuracy(model, val_dataloader, device):.4f}"
        print(msg)
    return model
