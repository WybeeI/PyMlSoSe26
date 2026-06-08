"""Run a trained model on a sample.

Loads the most recent run from runs/, picks a random test image,
prints the prediction.

Usage:
    python -m scripts.predict
    python -m scripts.predict --run runs/20260504-103000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mnist_proj.data import MNIST, to_tensor
from mnist_proj.model import SimpleClassifier


def latest_run(runs_dir: Path) -> Path:
    runs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no runs in {runs_dir}")
    return runs[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None, help="path to a run dir")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--data-path", type=str, default="data/mnist")
    p.add_argument("--index", type=int, default=0, help="test set index to predict")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run) if args.run else latest_run(Path(args.runs_dir))
    print(f"loading {run_dir}")

    model = SimpleClassifier()
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()

    test_ds = MNIST(args.data_path, split="test", transforms=[to_tensor])
    image, label = test_ds[args.index]
    with torch.no_grad():
        pred = model(image.unsqueeze(0)).argmax(dim=1).item()

    print(f"true: {label}  predicted: {pred}")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    print(f"run test accuracy: {metrics['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
