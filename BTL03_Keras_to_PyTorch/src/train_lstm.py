"""Train and evaluate the T06 MNIST Summation CNN+LSTM model with L1Loss (MAE)."""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .lstm_dataset import make_mnist_summation_splits
from .lstm_models import MnistCnnLstm, count_parameters


def run_epoch(model, loader, device, optimizer=None):
    """Run one training or evaluation epoch and return MAE (L1Loss), RMSE, and rounded accuracies."""
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.L1Loss(reduction="sum")
    mae_sum = 0.0
    sq_sum = 0.0
    exact_matches = 0
    within_one = 0
    count = 0

    with torch.set_grad_enabled(is_train):
        for sequences, targets in loader:
            sequences = sequences.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda").view(-1, 1)

            preds = model(sequences)
            batch_size = targets.size(0)
            loss = loss_fn(preds, targets) / batch_size

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            diff = (preds.detach() - targets).abs()
            mae_sum += diff.sum().item()
            sq_sum += (diff * diff).sum().item()
            exact_matches += (preds.detach().round() == targets).sum().item()
            within_one += (diff <= 1.0).sum().item()
            count += batch_size

    return {
        "loss": mae_sum / count,
        "rmse": math.sqrt(sq_sum / count),
        "exact_accuracy": exact_matches / count,
        "within_1_accuracy": within_one / count,
    }


def _build_optimizer(name, params, lr, momentum):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def train_lstm(
    data_dir="data/mnist",
    output_dir="results/lstm_summation_seed42",
    *,
    max_to_add=8,
    min_to_add=1,
    val_ratio=0.1,
    epochs=30,
    batch_size=128,
    lr=0.01,
    optimizer_name="sgd",
    momentum=0.0,
    seed=42,
    sequence_seed=2018,
    device="auto",
    threads=4,
    patience=0,
    lr_patience=0,
    lr_factor=0.1,
    min_lr=1e-4,
    keras_init=True,
    overwrite=False,
):
    """Train MnistCnnLstm with nn.L1Loss, checkpointing best val_loss and evaluating on test."""
    if min(epochs, batch_size, threads, max_to_add, min_to_add) < 1 or lr <= 0:
        raise ValueError("epochs, batch_size, threads, sequence lengths and lr must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)

    dev = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    pin_memory = dev.type == "cuda"

    train_ds, val_ds, test_ds = make_mnist_summation_splits(
        data_dir,
        max_to_add=max_to_add,
        min_to_add=min_to_add,
        val_ratio=val_ratio,
        split_seed=seed,
        sequence_seed=sequence_seed,
    )
    eval_ds = val_ds if val_ds is not None else test_ds
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    model = MnistCnnLstm(keras_init=keras_init).to(dev)
    optimizer = _build_optimizer(optimizer_name, model.parameters(), lr=lr, momentum=momentum)

    output = Path(output_dir)
    if output.exists() and not overwrite and (output / "metrics.json").is_file():
        raise FileExistsError(f"Output directory already contains metrics.json: {output}")
    output.mkdir(parents=True, exist_ok=True)

    history = []
    best_loss, best_epoch = float("inf"), 0
    plateau_best, plateau_wait = float("inf"), 0

    for epoch in range(1, epochs + 1):
        start = perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        train_stats = run_epoch(model, train_loader, dev, optimizer)
        val_stats = run_epoch(model, val_loader, dev)

        if not all(math.isfinite(s["loss"]) for s in (train_stats, val_stats)):
            raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}")

        if lr_patience > 0:
            if val_stats["loss"] < plateau_best - 1e-4:
                plateau_best, plateau_wait = val_stats["loss"], 0
            else:
                plateau_wait += 1
                if plateau_wait >= lr_patience:
                    for group in optimizer.param_groups:
                        group["lr"] = max(min_lr, group["lr"] * lr_factor)
                    plateau_wait = 0

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "seconds": perf_counter() - start,
            "train_loss": train_stats["loss"],
            "train_rmse": train_stats["rmse"],
            "train_exact_accuracy": train_stats["exact_accuracy"],
            "val_loss": val_stats["loss"],
            "val_rmse": val_stats["rmse"],
            "val_exact_accuracy": val_stats["exact_accuracy"],
            "val_within_1_accuracy": val_stats["within_1_accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        with (output / "history.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

        if val_stats["loss"] < best_loss:
            best_loss, best_epoch = val_stats["loss"], epoch
            torch.save(model.state_dict(), output / "best.pth")
        elif patience > 0 and epoch - best_epoch >= patience:
            break

    model.load_state_dict(torch.load(output / "best.pth", map_location=dev, weights_only=True))
    test_stats = run_epoch(model, test_loader, dev)
    param_info = count_parameters(model)

    metrics = {
        "task": "T06_mnist_cnn_lstm",
        "loss_function": "L1Loss (MAE)",
        "max_to_add": max_to_add,
        "min_to_add": min_to_add,
        "val_ratio": val_ratio,
        "seed": seed,
        "sequence_seed": sequence_seed,
        "keras_init": keras_init,
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
        "batch_size": batch_size,
        "initial_lr": lr,
        "optimizer": optimizer_name,
        "momentum": momentum,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "parameters": param_info["pytorch_params"],
        "keras_equivalent_parameters": param_info["keras_equivalent_params"],
        "train_samples": len(train_ds),
        "validation_samples": len(eval_ds),
        "test_samples": len(test_ds),
        "validation": {
            "loss": best_loss,
            "rmse": history[best_epoch - 1]["val_rmse"],
            "exact_accuracy": history[best_epoch - 1]["val_exact_accuracy"],
            "within_1_accuracy": history[best_epoch - 1]["val_within_1_accuracy"],
        },
        "test": test_stats,
        "training_seconds": sum(row["seconds"] for row in history),
        "mean_epoch_seconds": sum(row["seconds"] for row in history) / len(history),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/lstm_summation_seed42"))
    parser.add_argument("--max-to-add", type=int, default=8)
    parser.add_argument("--min-to-add", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", dest="optimizer_name", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-seed", type=int, default=2018)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--lr-patience", type=int, default=0)
    parser.add_argument("--lr-factor", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--no-keras-init", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = vars(parser.parse_args())
    args["keras_init"] = not args.pop("no_keras_init")
    train_lstm(**args)


if __name__ == "__main__":
    main()
