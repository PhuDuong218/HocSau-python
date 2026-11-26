import argparse
import sys
import os

# Thêm thư mục gốc vào đường dẫn hệ thống
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.datasets import (
    make_mnist_loaders,
    make_gender_loaders,
    make_fer2013_loaders,
    make_hand_depth_loaders,
)

def check_dataset(name, batch_size):
    print(f"\n================ ĐANG KIỂM TRA DATASET: {name.upper()} ================")
    try:
        if name == "mnist":
            loaders = make_mnist_loaders(batch_size=batch_size)
            split_keys = ["train", "test"]
        elif name == "gender":
            loaders = make_gender_loaders(batch_size=batch_size)
            split_keys = ["train", "validation"]
        elif name == "fer2013":
            loaders = make_fer2013_loaders(batch_size=batch_size)
            split_keys = ["train", "validation"]
        elif name == "hand":
            loaders = make_hand_depth_loaders(batch_size=batch_size)
            split_keys = ["train", "test"]
        else:
            print(f"Lỗi: Dataset {name} không hợp lệ!")
            return

        for split in split_keys:
            loader = loaders[split]
            images, labels = next(iter(loader))
            print(f"-> Split: {split.upper()}")
            print(f"   + Tổng số batch: {len(loader)} | Batch size thực tế: {images.size(0)}")
            print(f"   + Image Tensor Shape: {images.shape} | Dtype: {images.dtype}")
            print(f"   + Label/Mask Shape: {labels.shape} | Dtype: {labels.dtype}")
            print(f"   + Min-Max Image Value: [{images.min():.3f}, {images.max():.3f}]")
            print(f"   + Min-Max Label Index: [{labels.min()}, {labels.max()}]")
        print(f"==> KIỂM TRA {name.upper()} THÀNH CÔNG! HỢP LỆ VỚI PYTORCH.")

    except FileNotFoundError as e:
        print(f"==> [THIẾU FILE DỮ LIỆU]: {e}")
    except Exception as e:
        print(f"==> [LỖI TRUY VẤN DỮ LIỆU]: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "gender", "fer2013", "hand"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    check_dataset(args.dataset, args.batch_size)