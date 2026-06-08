# AGENTS.md

Guidance for AI agents working in this repo.

## What this is

A minimal MNIST classifier used as the running example in TU Berlin's
PyML L2 lecture. Code is intentionally simple — it shows project
structure, not state-of-the-art.

## Conventions

- **Library code** lives under `mnist_proj/` and has no side effects on
  import (no top-level `argparse`, no I/O).
- **Scripts** live under `scripts/` and are the only place
  `if __name__ == "__main__":` and CLI parsing belong.
- **Run artifacts** go to `runs/<timestamp>/` — checkpoint, config,
  metrics. Never overwrite a previous run.
- **Tests** in `tests/` are fast unit tests (forward pass shapes, etc.).
  No tests that need a trained model or download data.

## When editing

- The data pipeline downloads MNIST as JPG files from Kaggle
  (`kagglehub`), then loads them from disk via PIL. Don't switch back
  to `torchvision.datasets.MNIST` — the folder-of-images approach is
  intentional, so students can inspect the data on disk.
- The model is a 2-layer MLP. Don't replace with a CNN unless the
  lecture asks for it.
- If you add a new entry script, mirror the existing `scripts/train.py`
  structure: parse args → call into library → write artifacts.
