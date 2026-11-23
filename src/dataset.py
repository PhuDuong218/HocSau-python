import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms():
    """
    Trả về transforms cho tập train (có Augmentation) và tập test (chỉ chuẩn hóa)
    Dữ liệu FER2013 là ảnh mức xám (Grayscale 1 kênh), kích thước 48x48.
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, test_transform

def get_dataloaders(train_dir="../data/raw/train", test_dir="../data/raw/test", batch_size=128, num_workers=0):
    """
    Khởi tạo DataLoader cho cả train và test.
    """
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes
