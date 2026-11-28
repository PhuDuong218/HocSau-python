"""T03: architectures converted directly from the supplied Keras notebooks."""
import torch
from torch import nn


def keras_dense_initialization(module):
    """Match Dense's Glorot-uniform weights and zero bias, not its random stream."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class GaussianMLP(nn.Module):
    """Dense(2 -> hidden, sigmoid) -> Dense(hidden -> 1, sigmoid)."""
    def __init__(self, hidden=16):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, hidden), nn.Sigmoid(),
                                     nn.Linear(hidden, 1), nn.Sigmoid())
        self.apply(keras_dense_initialization)

    def forward(self, x):
        return self.network(x)


class MNISTMLP(nn.Module):
    """784 -> 512 -> 512 -> 10, ReLU and dropout 0.2; returns logits."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 10))
        self.apply(keras_dense_initialization)

    def forward(self, x):
        return self.network(x)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
