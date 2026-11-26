"""Convert the original FER2013 CSV file into ImageFolder directories.

The resulting layout works with ``torchvision.datasets.ImageFolder``:

data/raw/
  train/<emotion>/*.png
  test/<emotion>/*.png

The official PublicTest and PrivateTest partitions are combined into ``test``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


LABELS = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral",
}

SPLITS = {
    "Training": "train",
    "PublicTest": "test",
    "PrivateTest": "test",
}


def convert(csv_path: Path, output_dir: Path, overwrite: bool) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    targets = [output_dir / "train", output_dir / "test"]
    present = [path for path in targets if path.exists()]
    if present and not overwrite:
        names = ", ".join(str(path) for path in present)
        raise FileExistsError(
            f"Thư mục đích đã tồn tại: {names}. "
            "Dùng --overwrite nếu bạn muốn tạo lại toàn bộ ảnh."
        )
    if overwrite:
        for path in present:
            shutil.rmtree(path)

    for split in ("train", "test"):
        for label in LABELS.values():
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        expected_columns = {"emotion", "pixels", "Usage"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected_columns:
            raise ValueError(
                "CSV phải có đúng ba cột: emotion, pixels, Usage. "
                f"Nhưng file hiện có: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                split = SPLITS[row["Usage"]]
                label = LABELS[row["emotion"]]
            except KeyError as error:
                raise ValueError(
                    f"Nhãn hoặc tập dữ liệu không hợp lệ ở dòng {row_number}: {error}"
                ) from error

            pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
            if pixels.size != 48 * 48:
                raise ValueError(
                    f"Dòng {row_number} có {pixels.size} pixel, cần đúng 2.304 pixel."
                )

            index = counts[split]
            image = Image.fromarray(pixels.reshape(48, 48), mode="L")
            image.save(output_dir / split / label / f"{index:05d}.png")
            counts[split] += 1

    print(f"Đã tạo {counts['train']:,} ảnh train và {counts['test']:,} ảnh test.")
    print(f"Đầu ra: {output_dir.resolve()}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "raw" / "fer2013.csv",
        help="Đường dẫn tới fer2013.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "raw",
        help="Thư mục sẽ chứa train/ và test/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Xóa train/ và test/ hiện có trước khi chuyển đổi.",
    )
    args = parser.parse_args()
    convert(args.input.resolve(), args.output.resolve(), args.overwrite)


if __name__ == "__main__":
    main()
