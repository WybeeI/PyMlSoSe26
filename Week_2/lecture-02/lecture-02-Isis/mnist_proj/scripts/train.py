"""Train an MNIST classifier.

Entry point. Parses CLI args, builds the project's library objects,
runs training, writes a run directory.

Usage:
    python -m scripts.train --epochs 2 --batch-size 128 --lr 1e-3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch

from mnist_proj.data import MNIST, build_dataloader, random_augment, to_tensor
from mnist_proj.eval import accuracy
from mnist_proj.model import SimpleClassifier
from mnist_proj.train import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-path", type=str, default="data/mnist")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Augmentation only for training; val/test use the plain transform.
    train_ds = MNIST(args.data_path, split="train", transforms=[random_augment, to_tensor])
    val_ds = MNIST(args.data_path, split="val", transforms=[to_tensor])
    test_ds = MNIST(args.data_path, split="test", transforms=[to_tensor])
    train_dl = build_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = build_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = build_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleClassifier()
    train(model, train_dl, val_dataloader=val_dl, epochs=args.epochs, lr=args.lr)

    acc = accuracy(model, test_dl)
    print(f"test accuracy: {acc:.4f}")

    # Run artifacts: checkpoint + config + metric, in one directory.
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.runs_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    (run_dir / "metrics.json").write_text(json.dumps({"test_accuracy": acc}, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
