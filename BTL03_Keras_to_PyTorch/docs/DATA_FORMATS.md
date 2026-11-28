# Định dạng dữ liệu đầu vào T02

## MNIST
Không cần chuẩn bị thủ công. `torchvision.datasets.MNIST(root, train=True/False, download=True)` tải dữ liệu khi được yêu cầu.

## Human Gender
Đặt 4 file NPY sau vào `data/gender/`:
- `64_64_11938_4098_train_x_onehot.npy`
- `64_64_11938_4098_train_y_onehot.npy`
- `64_64_5968_4098_val_x_onehot.npy`
- `64_64_5968_4098_val_y_onehot.npy`

Theo notebook Keras, ảnh có dạng NHWC sau khi đọc và được `np.rot90(..., k=3, axes=(1, 2))`. Dataset chuyển sang CHW. Nhãn one-hot được chuyển thành class index. Quy ước notebook: 0=female, 1=male.

## FER2013
Đặt:
- `fer2013_training_onehot.csv`
- `fer2013_publictest_onehot.csv`

Mỗi dòng phải có 2304 giá trị pixel + 7 giá trị one-hot. Pixel được reshape thành 48x48. Mapping lấy đúng theo hàm `get_emotion` trong notebook:
0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral.

## Hand Depth segmentation
Cấu trúc:
```text
data/segnet/
├── train/
│   ├── depth/
│   └── labels/
└── test/
    ├── depth/
    └── labels/
```
Ảnh depth và mask có cùng tên file. Mask dùng mã màu lớp trong notebook:
- 255: palm -> class 0
- 43: finger 1 -> class 1
- 85: finger 2 -> class 2
- 128: finger 3 -> class 3
- 170: finger 4 -> class 4
- 212: finger 5 -> class 5
- 0: background -> class 6

Ảnh/mask được resize 96x96 bằng nearest-neighbor để không làm thay đổi giá trị nhãn. Mẫu lỗi hoặc thiếu mask sẽ báo lỗi, không âm thầm bỏ qua.
