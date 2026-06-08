"""Model architecture.

A 2-layer MLP. Decoupled from training: imports nothing from data, train,
or eval. You can instantiate it and call it on a random tensor with no
training infra at all.
"""
import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, image_size: int = 28, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
