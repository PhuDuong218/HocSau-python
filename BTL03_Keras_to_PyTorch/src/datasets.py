import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# ==========================================
# 1. DATASET: MNIST (Chữ số viết tay 28x28)
# ==========================================
def make_mnist_loaders(data_dir="data/mnist", batch_size=128):
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Chuẩn hóa mean, std chuẩn của MNIST
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return {"train": train_loader, "test": test_loader}

# ==========================================
# 2. DATASET: HUMAN GENDER (Ảnh NPY 64x64)
# ==========================================
class NpyGenderDataset(Dataset):
    def __init__(self, x_path, y_path):
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Không tìm thấy file: {x_path} hoặc {y_path}")

        # Đọc mảng numpy từ file
        raw_x = np.load(x_path) # Shape Keras: [N, 64, 64, 1] hoặc [N, 64, 64]
        raw_y = np.load(y_path) # Shape nhãn: One-hot [N, 2] hoặc [N]

        # Chuẩn hóa về định dạng PyTorch [N, 1, 64, 64]
        if raw_x.ndim == 3:
            raw_x = np.expand_dims(raw_x, axis=1)
        elif raw_x.ndim == 4 and raw_x.shape[-1] == 1:
            raw_x = np.transpose(raw_x, (0, 3, 1, 2))

        # Scale giá trị pixel về float32 trong khoảng [0, 1]
        if raw_x.max() > 1.0:
            raw_x = raw_x.astype(np.float32) / 255.0
        else:
            raw_x = raw_x.astype(np.float32)

        # Chuyển nhãn One-hot -> Class Index (0: Female, 1: Male)
        if raw_y.ndim == 2:
            raw_y = np.argmax(raw_y, axis=1)

        self.x = torch.from_numpy(raw_x).float()
        self.y = torch.from_numpy(raw_y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def make_gender_loaders(data_dir="data/gender", batch_size=128):
    train_x = os.path.join(data_dir, "64_64_11938_4098_train_x_onehot.npy")
    train_y = os.path.join(data_dir, "64_64_11938_4098_train_y_onehot.npy")
    val_x = os.path.join(data_dir, "64_64_5968_4098_val_x_onehot.npy")
    val_y = os.path.join(data_dir, "64_64_5968_4098_val_y_onehot.npy")

    train_ds = NpyGenderDataset(train_x, train_y)
    val_ds = NpyGenderDataset(val_x, val_y)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    }

# ==========================================
# 3. DATASET: FER2013 (Đọc từ thư mục ảnh train/test)
# ==========================================
def make_fer2013_loaders(data_dir="data/fer2013", batch_size=128):
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục: '{train_path}' hoặc '{test_path}'")

    # Data Augmentation cho tập Train tương đương ImageDataGenerator của Keras
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Pipeline cho tập Validation/Test (không augment ngẫu nhiên)
    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # ImageFolder tự động ánh xạ 7 folder con thành nhãn số nguyên [0..6] dạng torch.long
    train_ds = datasets.ImageFolder(root=train_path, transform=transform_train)
    val_ds = datasets.ImageFolder(root=test_path, transform=transform_val)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    }

# ==========================================
# 4. DATASET: HAND DEPTH SEGMENTATION (96x96)
# ==========================================
class HandDepthSegmentationDataset(Dataset):
    # 7 mốc giá trị pixel gốc trong ảnh mask phân đoạn
    RAW_VALUES = [0, 43, 85, 128, 170, 212, 255]

    def __init__(self, root_dir, split="train", target_size=(96, 96)):
        self.depth_dir = os.path.join(root_dir, split, "depth")
        self.labels_dir = os.path.join(root_dir, split, "labels")
        self.target_size = target_size

        self.depth_files = sorted(glob.glob(os.path.join(self.depth_dir, "*.png")))
        if len(self.depth_files) == 0:
            raise FileNotFoundError(f"Không tìm thấy ảnh depth tại: {self.depth_dir}")

        # Bảng Lookup Table (LUT) 256 phần tử để ánh xạ trực tiếp sang class index 0..6
        self.lut = np.zeros(256, dtype=np.int64)
        for class_idx, pixel_val in enumerate(self.RAW_VALUES):
            self.lut[pixel_val] = class_idx

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth_path = self.depth_files[idx]
        label_filename = os.path.basename(depth_path).replace("_depth_", "_labels_")
        label_path = os.path.join(self.labels_dir, label_filename)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Không tìm thấy file nhãn tương ứng: {label_path}")

        # 1. Đọc ảnh depth -> grayscale -> resize (96, 96) -> chuẩn hóa [0, 1]
        depth_img = Image.open(depth_path).convert("L")
        depth_img = depth_img.resize(self.target_size, Image.BILINEAR)
        depth_arr = np.array(depth_img, dtype=np.float32) / 255.0
        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0).float() # [1, 96, 96]

        # 2. Đọc ảnh label mask -> resize (96, 96) bằng NEAREST
        label_img = Image.open(label_path)
        label_img = label_img.resize(self.target_size, Image.NEAREST)
        label_arr = np.array(label_img)
        if label_arr.ndim == 3:
            label_arr = label_arr[:, :, 0]

        # Ánh xạ pixel gốc về class index 0..6
        label_mapped = self.lut[label_arr]
        mask_tensor = torch.from_numpy(label_mapped).long() # [96, 96]

        return depth_tensor, mask_tensor

def make_hand_depth_loaders(data_dir="data/segnet", batch_size=8):
    train_ds = HandDepthSegmentationDataset(data_dir, split="train")
    test_ds = HandDepthSegmentationDataset(data_dir, split="test")

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    }
