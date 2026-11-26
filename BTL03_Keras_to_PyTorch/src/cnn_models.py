"""PyTorch CNNs for the three Module 2 classification tasks."""

from torch import nn


def build_model(dataset):
    size, classes, blocks, dense, dropout, head_dropout = {
        "mnist": (28, 10, ((64,), (128,), (256,)), (512, 512), .2, .2),
        "gender": (64, 2, ((16, 16), (32,)*3, (64,)*3, (128,)*3, (256,)*3), (1024, 1024), .2, .2),
        "fer2013": (48, 7, ((64, 64), (128, 128, 218, 218)), (1024,), .25, .5),
    }[dataset]
    layers = []
    in_channels = 1
    for block in blocks:
        for index, out_channels in enumerate(block):
            padding = 0 if dataset == "fer2013" and index % 2 else 1
            layers.extend([nn.Conv2d(in_channels, out_channels, 3, padding=padding), nn.ReLU()])
            size -= 2 * (1 - padding)
            in_channels = out_channels
        layers.extend([nn.MaxPool2d(2), nn.Dropout(dropout)])
        size //= 2
    layers.append(nn.Flatten())
    features = in_channels * size * size
    for units in dense:
        layers.extend([nn.Linear(features, units), nn.ReLU(), nn.Dropout(head_dropout)])
        features = units
    layers.append(nn.Linear(features, classes))
    return nn.Sequential(*layers)
