"""Smoke tests for the model.

Fast — no data download, no training. The point is to catch silent
breakages: e.g. someone refactors the architecture and the output
shape changes without anyone noticing.
"""
import torch

from mnist_proj.model import SimpleClassifier


def test_forward_shape():
    model = SimpleClassifier()
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_logits_are_finite():
    model = SimpleClassifier()
    x = torch.randn(2, 1, 28, 28)
    logits = model(x)
    assert torch.isfinite(logits).all()
