"""PyTorch Dataset and DataLoader utilities for T06 MNIST Summation (CNN+LSTM)."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class MnistSummationDataset(Dataset):
    """Sequence dataset where each sample sums 1..max_to_add MNIST digit images.

    Matches the Keras `MNIST_CNN_LSTM` preprocessing:
    - Images are scaled to float32 in [0, 1] without mean/std normalization so
      zero-padded time steps remain exact zeros.
    - Each sequence picks `num_to_add` random digits in [min_to_add, max_to_add],
      places them in the leading slots, pads remaining slots with zeros, and
      targets the float32 sum of their digit labels.
    """

    def __init__(
        self,
        images,
        targets,
        *,
        num_sequences=None,
        max_to_add=8,
        min_to_add=1,
        seed=2018,
    ):
        if max_to_add < 1 or min_to_add < 1 or min_to_add > max_to_add:
            raise ValueError("Require 1 <= min_to_add <= max_to_add")
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError(f"Expected images shape [N, 1, 28, 28], got {tuple(images.shape)}")
        if len(images) != len(targets) or len(images) == 0:
            raise ValueError("Images and targets must be non-empty and have matching length")

        self.images = images.contiguous()
        self.base_targets = targets
        self.max_to_add = int(max_to_add)
        self.min_to_add = int(min_to_add)
        self.num_sequences = int(len(images) if num_sequences is None else num_sequences)
        if self.num_sequences < 1:
            raise ValueError("num_sequences must be positive")

        rng = np.random.RandomState(seed)
        n_base = len(images)
        indices = np.full((self.num_sequences, self.max_to_add), -1, dtype=np.int64)
        lengths = np.empty(self.num_sequences, dtype=np.int64)
        sums = np.empty(self.num_sequences, dtype=np.float32)

        for i in range(self.num_sequences):
            if self.min_to_add == 1:
                num_to_add = int(np.ceil(rng.rand() * self.max_to_add))
            else:
                num_to_add = int(rng.randint(self.min_to_add, self.max_to_add + 1))
            picked = rng.choice(n_base, size=num_to_add, replace=True)
            indices[i, :num_to_add] = picked
            lengths[i] = num_to_add
            sums[i] = float(targets[picked].sum().item())

        self.indices = torch.from_numpy(indices)
        self.lengths = torch.from_numpy(lengths)
        self.targets = torch.from_numpy(sums)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        seq = torch.zeros((self.max_to_add, 1, 28, 28), dtype=torch.float32)
        n = int(self.lengths[idx].item())
        seq[:n] = self.images[self.indices[idx, :n]]
        return seq, self.targets[idx]


def _load_raw_mnist(data_dir, *, train):
    ds = datasets.MNIST(root=str(data_dir), train=train, download=True)
    images = ds.data.to(dtype=torch.float32).unsqueeze(1) / 255.0
    targets = ds.targets.to(dtype=torch.int64)
    return images, targets


def _stratified_split(images, targets, val_ratio, seed):
    if val_ratio <= 0.0:
        return images, targets, None, None
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0.0, 1.0)")

    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    targets_np = targets.numpy()
    for label in np.unique(targets_np):
        cls_idx = rng.permutation(np.flatnonzero(targets_np == label)).tolist()
        val_count = max(1, round(len(cls_idx) * val_ratio))
        val_idx.extend(cls_idx[:val_count])
        train_idx.extend(cls_idx[val_count:])

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    return images[train_idx], targets[train_idx], images[val_idx], targets[val_idx]


def make_mnist_summation_splits(
    data_dir="data/mnist",
    *,
    max_to_add=8,
    min_to_add=1,
    val_ratio=0.1,
    split_seed=42,
    sequence_seed=2018,
    train_sequences=None,
    val_sequences=None,
    test_sequences=None,
):
    """Create train, validation (optional), and test MnistSummationDataset splits.

    Splitting base MNIST images before sampling sequences guarantees zero image
    overlap across train, validation, and test sets.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_raw_x, train_raw_y = _load_raw_mnist(data_path, train=True)
    test_raw_x, test_raw_y = _load_raw_mnist(data_path, train=False)

    tr_x, tr_y, va_x, va_y = _stratified_split(train_raw_x, train_raw_y, val_ratio, split_seed)

    train_ds = MnistSummationDataset(
        tr_x,
        tr_y,
        num_sequences=train_sequences,
        max_to_add=max_to_add,
        min_to_add=min_to_add,
        seed=sequence_seed,
    )
    val_ds = None
    if va_x is not None:
        val_ds = MnistSummationDataset(
            va_x,
            va_y,
            num_sequences=val_sequences,
            max_to_add=max_to_add,
            min_to_add=min_to_add,
            seed=sequence_seed + 1,
        )
    test_ds = MnistSummationDataset(
        test_raw_x,
        test_raw_y,
        num_sequences=test_sequences,
        max_to_add=max_to_add,
        min_to_add=min_to_add,
        seed=sequence_seed + 2,
    )
    return train_ds, val_ds, test_ds


def make_mnist_summation_loaders(
    data_dir="data/mnist",
    batch_size=128,
    *,
    max_to_add=8,
    min_to_add=1,
    val_ratio=0.1,
    split_seed=42,
    sequence_seed=2018,
    pin_memory=False,
):
    """Create DataLoaders for T06 MNIST Summation."""
    train_ds, val_ds, test_ds = make_mnist_summation_splits(
        data_dir,
        max_to_add=max_to_add,
        min_to_add=min_to_add,
        val_ratio=val_ratio,
        split_seed=split_seed,
        sequence_seed=sequence_seed,
    )
    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
        ),
    }
    if val_ds is not None:
        loaders["validation"] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
        )
    return loaders
