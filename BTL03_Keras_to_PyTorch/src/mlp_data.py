"""T03 data adapters; the shared T02 datasets.py is left unchanged."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.datasets import make_mnist_loaders


def gaussian_grid(side):
    """Same isotropic bivariate Gaussian and inclusive [-3,3] grid as Keras."""
    x1, x2 = np.mgrid[-3.0:3.0:complex(side), -3.0:3.0:complex(side)]
    x = np.column_stack([x1.ravel(), x2.ravel()])
    y = np.exp(-0.5 * np.square(x).sum(axis=1)) / (2 * np.pi)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)


def make_mlp_mnist_loaders(data_dir="data/mnist", batch_size=128, seed=42):
    """Use T02's torchvision datasets, with raw pixel scaling from Keras.

    T02's image transform additionally standardizes by mean/std. A local tensor
    view of its raw uint8 data preserves the source notebook's [0,1] inputs and
    caches them once. No shared dataset, transform or function is modified.
    """
    shared = make_mnist_loaders(str(data_dir), batch_size=batch_size)
    loaders = {}
    for split in ["train", "test"]:
        original = shared[split].dataset
        images = original.data.unsqueeze(1).to(torch.float32).div(255)
        targets = original.targets.to(torch.long)
        loaders[split] = DataLoader(TensorDataset(images, targets), batch_size=batch_size,
                                    shuffle=split == "train", num_workers=0,
                                    generator=torch.Generator().manual_seed(seed))
    return loaders
