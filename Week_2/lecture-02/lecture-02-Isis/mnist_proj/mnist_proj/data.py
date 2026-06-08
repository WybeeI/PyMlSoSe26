"""Dataset + transforms for MNIST.

Downloads MNIST as JPG files from Kaggle, then loads from disk.
This keeps the data layer transparent: students can open the folder
and look at real image files.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import kagglehub
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

KAGGLE_DATASET = "scolianni/mnistasjpg"
# The labelled set has ~42 000 images. Deterministic split (by sorted filename):
TRAIN_SIZE = 30_000
VAL_SIZE = 6_000
# test = whatever is left (~6 000)
SPLITS = ("train", "val", "test")
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL grayscale image → normalised float tensor of shape (1, 28, 28)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0)              # (H, W) → (1, H, W)
    return (t - MNIST_MEAN) / MNIST_STD


def random_augment(img: Image.Image) -> Image.Image:
    """Random ±15° rotation + ±2 px shift. PIL → PIL."""
    img = img.rotate(random.uniform(-15, 15), resample=Image.BILINEAR)
    dx, dy = random.randint(-2, 2), random.randint(-2, 2)
    # PIL affine: output[x,y] = input[a*x+b*y+c, d*x+e*y+f]
    return img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy))


def download_mnist(data_path: str | Path = "data/mnist") -> Path:
    """Fetch the MNIST JPGs to ``data_path`` (no-op if already present)."""
    data_path = Path(data_path)
    if data_path.exists() and any(data_path.iterdir()):
        return data_path
    cache = kagglehub.dataset_download(KAGGLE_DATASET)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(cache), str(data_path))
    return data_path


def get_mnist_jpg_paths(data_path: str | Path) -> tuple[list[Path], list[int]]:
    """Return ``(paths, labels)`` for every JPG in the dataset, sorted by filename.

    The sort is by filename so the train/val/test split is reproducible
    across machines (directory iteration order is not guaranteed).
    """
    base = Path(data_path) / "trainingSet" / "trainingSet"
    paths: list[Path] = []
    labels: list[int] = []
    for digit in range(10):
        for img in sorted((base / str(digit)).glob("*.jpg")):
            paths.append(img)
            labels.append(digit)
    order = sorted(range(len(paths)), key=lambda i: paths[i].name)
    return [paths[i] for i in order], [labels[i] for i in order]


class MNIST:
    """Loads MNIST from a folder of JPGs.

    A plain Python class — *not* a subclass of ``torch.utils.data.Dataset``.
    PyTorch's ``DataLoader`` only needs ``__len__`` and ``__getitem__``;
    it duck-types on the protocol. Inheriting from ``Dataset`` is a
    convention, not a requirement.

    Args:
        data_path: where the dataset lives on disk (downloaded if missing).
        split: one of ``"train"``, ``"val"``, ``"test"``. The split is
            deterministic — sizes are ``TRAIN_SIZE`` / ``VAL_SIZE`` / rest,
            taken in sorted-filename order.
        transforms: list of callables applied in order to each sample.
            ``None`` means no transforms — ``__getitem__`` returns a raw
            PIL image. For training, pass ``[random_augment, to_tensor]``;
            for val/test pass ``[to_tensor]``.
    """

    def __init__(
        self,
        data_path: str | Path = "data/mnist",
        split: str = "train",
        transforms: list | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.data_path = Path(data_path)
        self.split = split
        self.transforms = transforms or []

        self.download()
        paths, labels = get_mnist_jpg_paths(self.data_path)

        train_end = TRAIN_SIZE
        val_end = TRAIN_SIZE + VAL_SIZE
        if split == "train":
            self.paths = paths[:train_end]
            self.labels = labels[:train_end]
        elif split == "val":
            self.paths = paths[train_end:val_end]
            self.labels = labels[train_end:val_end]
        else:  # "test"
            self.paths = paths[val_end:]
            self.labels = labels[val_end:]

    def download(self) -> None:
        """Delegate to the module-level :func:`download_mnist`."""
        download_mnist(self.data_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        sample = Image.open(self.paths[idx])
        for t in self.transforms:
            sample = t(sample)
        return sample, self.labels[idx]


def build_dataloader(
    dataset: MNIST, batch_size: int = 128, shuffle: bool = True
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
