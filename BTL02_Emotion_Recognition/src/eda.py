"""Recreate FER2013 class distribution and sample-face figures."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


def run_eda(data_dir: Path = ROOT / "data/raw", output_dir: Path = ROOT / "results/plots",
            audit_path: Path = ROOT / "results/data_audit/dataset_audit.json", seed: int = 42):
    """Use verified audit counts; use raw images for the reproducible sample grid."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    counts = audit["original_counts"]
    for split in ("train", "test"):
        if set(counts[split]) != set(CLASSES):
            raise ValueError(f"Unexpected classes in {split} audit counts")
    output_dir.mkdir(parents=True, exist_ok=True)

    x = list(range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([v - 0.2 for v in x], [counts["train"][c] for c in CLASSES],
           width=0.4, label="Train")
    ax.bar([v + 0.2 for v in x], [counts["test"][c] for c in CLASSES],
           width=0.4, label="Test")
    ax.set_xticks(x, CLASSES, rotation=30)
    ax.set_ylabel("Number of images")
    ax.set_title("FER2013 class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_distribution.png", dpi=200)
    plt.close(fig)

    sample_path = output_dir / "sample_faces.png"
    if (data_dir / "train").is_dir():
        rng = random.Random(seed)
        fig, axes = plt.subplots(len(CLASSES), 5, figsize=(10, 12))
        for row, label in enumerate(CLASSES):
            files = sorted(p for p in (data_dir / "train" / label).iterdir()
                           if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            if len(files) < 5:
                raise ValueError(f"Need at least five train images for {label}")
            for col, path in enumerate(rng.sample(files, 5)):
                with Image.open(path) as image:
                    if image.size != (48, 48):
                        raise ValueError(f"Unexpected image size: {path}")
                    axes[row, col].imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
                axes[row, col].axis("off")
                if col == 0:
                    axes[row, col].text(-0.08, 0.5, label, transform=axes[row, col].transAxes,
                                        ha="right", va="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(sample_path, dpi=200)
        plt.close(fig)
    else:
        sample_path = None

    return counts, sample_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/plots")
    args = parser.parse_args()
    counts, sample = run_eda(args.data_dir, args.output_dir)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"Saved distribution plot to {args.output_dir / 'dataset_distribution.png'}")
    print(f"Sample grid: {sample if sample else 'raw images unavailable; not regenerated'}")
