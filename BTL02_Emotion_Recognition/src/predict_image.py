"""Predict FER2013 emotion probabilities for one cropped face."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models import BaselineCNN

ROOT = Path(__file__).resolve().parents[1]


def predict(image_path, checkpoint, metadata):
    image_path, checkpoint, metadata = map(Path, (image_path, checkpoint, metadata))
    info = json.loads(metadata.read_text(encoding="utf-8"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if info.get("checkpoint_sha256") != digest:
        raise ValueError("Checkpoint SHA-256 does not match metadata")
    mapping = info.get("class_to_idx", {})
    if sorted(mapping.values()) != list(range(7)):
        raise ValueError("Metadata must contain seven unique class indices 0-6")
    labels = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    model = BaselineCNN(num_classes=7)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    with Image.open(image_path) as image:
        image = image.convert("L").resize((48, 48))
        tensor = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(image))
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
    return sorted(zip(labels, probabilities.tolist()), key=lambda pair: pair[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a cropped face image")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/baseline_best.pth")
    parser.add_argument("--metadata", type=Path, default=ROOT / "checkpoints/baseline_best.metadata.json")
    args = parser.parse_args()
    try:
        result = predict(args.image, args.checkpoint, args.metadata)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")
    for label, probability in result:
        print(f"{label:10s} {probability:.2%}")


if __name__ == "__main__":
    main()
