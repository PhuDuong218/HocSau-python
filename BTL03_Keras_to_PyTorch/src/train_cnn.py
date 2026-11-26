"""Train and evaluate Module 2 CNNs with validation-based model selection."""

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .cnn_models import build_model
from .datasets import make_fer2013_loaders, make_gender_loaders, make_mnist_loaders


class FerOneHotCsv(Dataset):
    classes = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

    def __init__(self, path):
        rows = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
        if rows.shape[1] != 2311 or not np.isfinite(rows).all():
            raise ValueError(f"Expected 2304 pixels and seven finite label columns in {path}")
        pixels, labels = rows[:, :2304], rows[:, 2304:]
        if pixels.min() < 0 or pixels.max() > 255 or not (
            np.isin(labels, [0, 1]).all() and (labels.sum(axis=1) == 1).all()
        ):
            raise ValueError(f"Invalid pixel range or one-hot labels in {path}")
        scale = 255 if pixels.max() <= 1 else 1
        self.images = np.rint(pixels * scale).astype(np.uint8).reshape(-1, 48, 48)
        self.targets = labels.argmax(axis=1)
        self.transform = None
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), int(self.targets[index])


def make_splits(dataset, root, seed, manifest):
    factory = {"mnist": make_mnist_loaders, "gender": make_gender_loaders,
               "fer2013": make_fer2013_loaders}[dataset]
    csv_train = root / "fer2013_training_onehot.csv"
    csv_test = root / "fer2013_publictest_onehot.csv"
    if dataset == "fer2013" and (csv_train.is_file() or csv_test.is_file()):
        if manifest:
            raise ValueError("The split manifest requires FER2013 image folders")
        train, test = FerOneHotCsv(csv_train), FerOneHotCsv(csv_test)
    else:
        loaders = factory(root)
        train = loaders["train"].dataset
        if dataset == "gender":
            validation = loaders["validation"].dataset
            for split in (train, validation):
                split.x = torch.rot90(split.x, k=3, dims=(-2, -1)).contiguous()
            return train, validation, None
        test = loaders["validation" if dataset == "fer2013" else "test"].dataset
    if dataset == "mnist":
        train.transform = test.transform = transforms.ToTensor()
    else:
        if train.class_to_idx != test.class_to_idx or len(train.classes) != 7:
            raise ValueError("FER2013 requires the same seven class folders in train and test")
        base = [transforms.Grayscale(1), transforms.Resize((48, 48))]
        train.transform = transforms.Compose([
            *base, transforms.Pad(48, padding_mode="edge"),
            # The padded canvas is three times wider than the original.
            transforms.RandomAffine(45, translate=(0.2 / 3, 0.2 / 3),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(48), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        ])
        test.transform = transforms.Compose([*base, transforms.ToTensor()])
    validation = copy.copy(train)
    validation.transform = test.transform
    if manifest:
        with Path(manifest).open(encoding="utf-8", newline="") as file:
            rows = {row["path"]: row for row in csv.DictReader(file)}
        indices = {"train": [], "validation": []}
        for index, (filename, label) in enumerate(train.samples):
            row = rows[Path(filename).relative_to(root).as_posix()]
            if int(row["label"]) != label:
                raise ValueError(f"Manifest label differs for {filename}")
            if row["assigned_split"] in indices:
                indices[row["assigned_split"]].append(index)
        return Subset(train, indices["train"]), Subset(validation, indices["validation"]), test
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    targets = np.asarray(train.targets)
    for label in np.unique(targets):
        indices = rng.permutation(np.flatnonzero(targets == label)).tolist()
        count = max(1, round(len(indices) * 0.1))
        val_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    return Subset(train, train_indices), Subset(validation, val_indices), test


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_fn = nn.CrossEntropyLoss()
    loss_total = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count += labels.numel()
            loss_total += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
    return {"loss": loss_total / count, "accuracy": correct / count}


def train_cnn(dataset, data_dir, output_dir, *, epochs=12, batch_size=128,
              lr=None, seed=42, device="auto", threads=4, patience=5,
              split_manifest=None, optimizer_name=None):
    lr = {"mnist": .01, "gender": .001, "fer2013": .0001}[dataset] if lr is None else lr
    if min(epochs, batch_size, threads, patience) < 1 or lr <= 0:
        raise ValueError("Epochs, batch size, threads, patience and learning rate must be positive")
    if split_manifest and dataset != "fer2013":
        raise ValueError("The split manifest is supported only for FER2013")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    train, validation, test = make_splits(dataset, Path(data_dir), seed, split_manifest)
    if min(len(train), len(validation)) == 0:
        raise ValueError("Train and validation must both contain samples")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=batch_size)
    model = build_model(dataset).to(device, memory_format=torch.channels_last)
    optimizer_name = optimizer_name or ("sgd" if dataset == "mnist" else "adam")
    optimizer = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[optimizer_name](model.parameters(), lr=lr)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    history = []
    best_loss, best_epoch = float("inf"), 0
    plateau_best, plateau_wait = float("inf"), 0
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        rate = optimizer.param_groups[0]["lr"]
        training = run_epoch(model, train_loader, device, optimizer)
        valid = run_epoch(model, val_loader, device)
        if not all(math.isfinite(result["loss"]) for result in (training, valid)):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")
        if valid["loss"] < plateau_best - 1e-4:
            plateau_best, plateau_wait = valid["loss"], 0
        else:
            plateau_wait += 1
            if plateau_wait >= 5:
                for group in optimizer.param_groups:
                    group["lr"] = min(group["lr"], max(0.0001, group["lr"] * 0.1))
                plateau_wait = 0
        row = {"epoch": epoch, "learning_rate": rate, "seconds": perf_counter() - start}
        row.update({f"train_{key}": value for key, value in training.items()})
        row.update({f"val_{key}": value for key, value in valid.items()})
        history.append(row)
        print(json.dumps(row), flush=True)
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if valid["loss"] < best_loss:
            best_loss, best_epoch = valid["loss"], epoch
            torch.save(model.state_dict(), output / "best.pth")
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(torch.load(output / "best.pth", map_location=device, weights_only=True))
    metrics = {
        "dataset": dataset, "seed": seed, "device": str(device), "batch_size": batch_size,
        "initial_lr": lr, "optimizer": optimizer_name,
        "epochs_requested": epochs, "epochs_completed": len(history),
        "best_epoch": best_epoch, "parameters": sum(p.numel() for p in model.parameters()),
        "train_samples": len(train), "validation_samples": len(validation),
        "test_samples": len(test) if test is not None else 0,
        "split_manifest": str(split_manifest) if split_manifest else None,
        "classes": list(getattr(train.dataset if isinstance(train, Subset) else train, "classes",
                                ("female", "male"))),
        "validation": {"loss": best_loss, "accuracy": history[best_epoch - 1]["val_accuracy"]},
        "test": run_epoch(model, DataLoader(test, batch_size=batch_size), device) if test is not None else None,
        "training_seconds": sum(row["seconds"] for row in history),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["mnist", "gender", "fer2013"])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optimizer", dest="optimizer_name", choices=["sgd", "adam"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--split-manifest", type=Path)
    args = vars(parser.parse_args())
    args["data_dir"] = args["data_dir"] or Path("data") / args["dataset"]
    train_cnn(**args)


if __name__ == "__main__":
    main()
