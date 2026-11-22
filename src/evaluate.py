"""Evaluate the existing baseline checkpoint; never trains a model."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from .models import BaselineCNN
except ImportError:
    from models import BaselineCNN

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_checkpoint(path, device):
    """Support plain state_dict and conventional checkpoint dictionaries."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint phai chua state_dict, khong phai model pickle.")
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise ValueError("Khong tim thay state_dict trong checkpoint.")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = BaselineCNN(num_classes=7)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload


def save_case_grid(dataset, indices, targets, predictions, confidences, path, title):
    count = min(32, len(indices))
    if not count:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No samples in this category", ha="center", va="center")
        ax.axis("off")
    else:
        # Fixed seed ensures repeated reports select the same examples.
        selected = np.random.default_rng(42).choice(indices, count, replace=False)
        cols = min(8, count)
        rows = math.ceil(count / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, selected):
            image, _ = dataset[int(index)]
            ax.imshow((image[0].numpy() * 0.5 + 0.5).clip(0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"True: {dataset.classes[targets[index]]}\n"
                         f"Pred: {dataset.classes[predictions[index]]} ({confidences[index]:.1%})", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_filters(model, path):
    # conv3 weights are [128 output channels, 64 input channels, 3, 3].
    # Show signed mean across input channels for 16 output filters.
    kernels = model.conv3.weight.detach().cpu().mean(dim=1)[:16].numpy()
    limit = float(np.abs(kernels).max()) or 1.0
    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for i, (ax, kernel) in enumerate(zip(axes.flat, kernels)):
        ax.imshow(kernel, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(f"conv3 filter {i}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Final convolution weights: mean across 64 input channels")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate(checkpoint, data_dir, output_dir, *, device="cpu", batch_size=128,
             split_name="test", selection_used=True):
    checkpoint, data_dir, output_dir = map(Path, (checkpoint, data_dir, output_dir))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Thieu checkpoint: {checkpoint}. Can file baseline_best.pth tu Nhi.")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Thieu thu muc anh: {data_dir}. Can cac thu muc con 7 cam xuc.")
    if batch_size <= 0:
        raise ValueError("batch_size phai lon hon 0")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("May khong co CUDA; hay dung --device cpu.")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if dataset.classes != CLASSES:
        raise ValueError(f"Thu muc nhan khong khop baseline ImageFolder: {dataset.classes}; can {CLASSES}")
    model, payload = load_checkpoint(checkpoint, device)
    if "class_to_idx" in payload and payload["class_to_idx"] != dataset.class_to_idx:
        raise ValueError("class_to_idx trong checkpoint khong khop du lieu.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    targets, predictions, confidences = [], [], []
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            if images.shape[1:] != (1, 48, 48):
                raise ValueError(f"Anh phai co kich thuoc 48x48: nhan batch {tuple(images.shape)}")
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if not torch.isfinite(logits).all():
                raise ValueError("Model tra ve NaN/Inf; kiem tra checkpoint va du lieu.")
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            confidence, prediction = logits.softmax(dim=1).max(dim=1)
            targets.extend(labels.cpu().tolist())
            predictions.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    confidences = np.asarray(confidences)
    report = classification_report(targets, predictions, labels=list(range(7)),
                                   target_names=dataset.classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(targets, predictions, labels=list(range(7)))
    warning = ("Tap nay da duoc dung de chon checkpoint trong notebook baseline; "
               "ket qua khong phai danh gia tren test doc lap." if selection_used else
               "Chi xem la test doc lap neu tap nay chua tung duoc dung de chon model/hyperparameters.")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metrics = {
        "split": split_name, "samples": len(dataset), "loss": total_loss / len(dataset),
        "accuracy": accuracy_score(targets, predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "class_to_idx": dataset.class_to_idx,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest.hexdigest(),
        "data_dir": str(data_dir.resolve()), "device": str(device),
        "selection_used": selection_used, "evaluation_note": warning,
        "preprocessing": "grayscale(1), ToTensor, Normalize(mean=0.5, std=0.5), no augmentation",
        "filter_note": "conv3 weights, first 16 output filters, mean across 64 input channels; not feature maps or Grad-CAM",
        "classification_report": report,
    }
    # Create outputs only after inference has completed successfully.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    text_report = classification_report(targets, predictions, labels=list(range(7)),
                                        target_names=dataset.classes, digits=4, zero_division=0)
    (output_dir / "classification_report.txt").write_text(warning + "\n\n" + text_report, encoding="utf-8")
    with (output_dir / "classification_report.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for name in dataset.classes + ["macro avg", "weighted avg"]:
            row = report[name]
            writer.writerow([name] + [row[k] for k in ("precision", "recall", "f1-score", "support")])
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "true", "predicted", "confidence", "correct"])
        for (path, _), actual, predicted, confidence in zip(dataset.samples, targets, predictions, confidences):
            writer.writerow([str(Path(path).relative_to(data_dir)), dataset.classes[actual],
                             dataset.classes[predicted], float(confidence), bool(actual == predicted)])
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true / predicted"] + dataset.classes)
        for label, row in zip(dataset.classes, matrix):
            writer.writerow([label] + row.tolist())
    for normalized, filename in [(False, "confusion_matrix.png"), (True, "confusion_matrix_normalized.png")]:
        values = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1) if normalized else matrix
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(values, annot=True, fmt=".1%" if normalized else "d", cmap="Blues",
                    xticklabels=dataset.classes, yticklabels=dataset.classes, ax=ax)
        ax.set(xlabel="Predicted", ylabel="True", title=f"Baseline CNN - {split_name} (n={len(dataset)})")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)
    save_case_grid(dataset, np.flatnonzero(targets == predictions), targets, predictions, confidences,
                   output_dir / "true_predict_images.png", "Correct predictions (up to 32 samples)")
    save_case_grid(dataset, np.flatnonzero(targets != predictions), targets, predictions, confidences,
                   output_dir / "wrong_predict_images.png", "Wrong predictions (up to 32 samples)")
    save_filters(model, output_dir / "final_conv_filters.png")
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs = sorted(((int(off_diagonal[i, j]), dataset.classes[i], dataset.classes[j])
                    for i in range(7) for j in range(7) if off_diagonal[i, j]), reverse=True)[:5]
    summary = [f"# Baseline evaluation: {split_name}", "", warning, "",
               f"- Samples: {len(dataset)}", f"- Accuracy: {metrics['accuracy']:.2%}",
               f"- Macro-F1: {metrics['macro_f1']:.4f}", f"- Weighted-F1: {metrics['weighted_f1']:.4f}",
               f"- Cross-entropy: {metrics['loss']:.4f}", "", "## Most frequent errors", ""]
    summary += [f"- {actual} -> {predicted}: {count}" for count, actual, predicted in pairs]
    summary += ["", "Filter visualization shows mean signed conv3 weights across input channels.",
                "It does not show activation maps or establish which facial regions the model uses.",
                "Use predictions.csv to inspect examples before explaining causes of errors."]
    (output_dir / "evaluation_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(warning)
    print(text_report)
    print(f"Accuracy: {metrics['accuracy']:.2%}; Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Saved to: {output_dir.resolve()}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/baseline_best.pth")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluation/baseline")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--independent-holdout", action="store_true",
                        help="Only use when the supplied split was never used for model selection.")
    args = parser.parse_args()
    try:
        evaluate(args.checkpoint, args.data_dir, args.output_dir, device=args.device,
                 batch_size=args.batch_size, split_name=args.split_name,
                 selection_used=not args.independent_holdout)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
