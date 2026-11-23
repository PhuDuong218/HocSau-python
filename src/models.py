"""
Định nghĩa kiến trúc các mô hình mạng nơ-ron tích chập (CNN) cho bài toán FER2013:
- BaselineCNN: Kiến trúc cơ sở 3 khối Conv-BN-ReLU-Pool + FC
- SmallCNN: Thử nghiệm kiến trúc nhỏ gọn
- Medium01CNN: Thử nghiệm mạng sâu hơn
- VGG16_FailCase: Thử nghiệm mô hình nặng chuyển giao sang dữ liệu 48x48
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BaselineCNN(nn.Module):
    """
    Kiến trúc Baseline CNN (3 khối Conv2D-BatchNorm-ReLU-MaxPool + FC 256).
    Phù hợp kích thước ảnh 48x48, bảo toàn cấu trúc feature map ở kích thước 6x6.
    Giữ nguyên tên layer và trọng số để tương thích hoàn toàn state_dict.
    """
    def __init__(self, num_classes=7):
        super(BaselineCNN, self).__init__()
        # Khối 1: 48x48 -> 24x24
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # Khối 2: 24x24 -> 12x12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Khối 3: 12x12 -> 6x6
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully Connected Layer
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SmallCNN(nn.Module):
    """
    Kiến trúc Small CNN: 2 Conv block 1, 3 Conv block 2.
    """
    def __init__(self, num_classes=7):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 12 * 12, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class Medium01CNN(nn.Module):
    """
    Kiến trúc Medium-01 CNN: 3 khối tích chập sâu hơn.
    """
    def __init__(self, num_classes=7):
        super(Medium01CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 48x48 -> 24x24
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.2),

            # Block 2: 24x24 -> 12x12
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.2),

            # Block 3: 12x12 -> 6x6
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class VGG16_FailCase(nn.Module):
    """
    Mô hình VGG16 thực nghiệm Fail Case:
    13 tầng Conv + 5 tầng MaxPool làm teo feature map từ 48x48 xuống 1x1 pixel,
    dẫn đến mất hoàn toàn cấu trúc không gian của khuôn mặt và quá khớp do >130M tham số.
    """
    def __init__(self, num_classes=7):
        super(VGG16_FailCase, self).__init__()
        base_vgg = models.vgg16(weights=None)
        self.features = base_vgg.features
        # Thay đổi đầu vào từ 3 kênh RGB thành 1 kênh Grayscale
        self.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Sau 5 tầng MaxPool: 48 -> 24 -> 12 -> 6 -> 3 -> 1x1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
