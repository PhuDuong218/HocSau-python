from __future__ import annotations

import copy
import random

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from torch import nn


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sklearn_model(kind: str = "random_forest"):
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            loss="huber",
            random_state=42,
        )
    return RandomForestRegressor(
        n_estimators=500,
        max_features=0.7,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )


class HouseMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 300,
    lr: float = 1e-3,
    patience: int = 35,
):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HouseMLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xva), yva).item()
        history.append((epoch, float(loss.item()), float(val_loss)))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, history, device


def predict_mlp(model, X: np.ndarray, device):
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(xt).cpu().numpy().reshape(-1)
    return pred
