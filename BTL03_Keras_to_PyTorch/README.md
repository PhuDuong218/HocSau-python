# BTL03 – Chuyển code từ Keras sang PyTorch

> Repository này dùng chung cho nhóm thực hiện BTL03 trên repo **[PhuDuong218/HocSau-python](https://github.com/PhuDuong218/HocSau-python)** (nhánh làm việc: `btl03`). Mục tiêu là chuyển các ví dụ Deep Learning từ Keras sang PyTorch, tổ chức mã nguồn thống nhất, kiểm thử dữ liệu và ghi lại kết quả có thể tái lập.

## 1. Mục tiêu và phạm vi

Theo yêu cầu bài tập trên Google Classroom, nhóm cần chuyển code Keras sang PyTorch và đính kèm báo cáo, mã nguồn đã chuyển đổi cùng các tệp liên quan.

Các nhóm bài toán được nêu trong tài liệu:

1. **MLP** – MNIST và bài toán xấp xỉ hàm.
2. **CNN** – MNIST, dự đoán giới tính từ ảnh, nhận diện cảm xúc FER2013.
3. **RNN/LSTM** – MNIST Summation, chuỗi gồm 7 ảnh chữ số và dự đoán tổng.
4. **Image Segmentation** – phân đoạn ảnh độ sâu bàn tay (Hand Depth, 96×96).

Repository có nền tảng **T01 (repo/môi trường)** và **T02 (Dataset/DataLoader)**. Các notebook CNN của **T04 và T05** cùng lệnh huấn luyện được mô tả ở mục CNN.

## 2. Thành viên và nguyên tắc phối hợp

Phân công công việc theo bảng theo dõi nhóm:

| Thành viên | GitHub username | Nhiệm vụ chính | Nhiệm vụ hỗ trợ | Nhánh làm việc |
|---|---|---|---|---|
| **Tuyết Nhi**  | `@TuyetNhi-Tech` | **T01, T02:** Quản lý Repo & Data Pipeline.<br>**T07:** Module 4 - SegNet Bàn tay.<br>**R01, R05:** Báo cáo Tổng quan & SegNet. | Review code toàn nhóm, tổng hợp báo cáo | `feature/T07-segnet` |
| **Xuân Anh** | `@PhuDuong218` | **T03:** Module 1 - MLP (Xấp xỉ hàm & MNIST).<br>**R02:** Báo cáo kỹ thuật Module MLP. | Hỗ trợ kiểm thử pipeline | `feature/T03-mlp` |
| **Phú Dương** | `@XuanAnh` | **T04, T05:** Module 2 - CNN (MNIST, Gender, FER2013).<br>**R03:** Báo cáo kỹ thuật Module CNN. | Tối ưu hóa siêu tham số | `feature/T04-T05-cnn` |
| **Quốc Việt** | `@Viet` | **T06:** Module 3 - RNN/LSTM (MNIST Summation 7 ảnh).<br>**R04:** Báo cáo kỹ thuật Module LSTM. | Kiểm thử tính toán chuỗi | `feature/T06-lstm` |

**Nguyên tắc làm việc**
- Mỗi task có một người phụ trách chính (Owner); người hỗ trợ (Support) chỉ phối hợp trong phạm vi được giao.
- Trước khi bắt đầu task, đọc README, kiểm tra issue/task board và đầu vào được bàn giao.
- Không tự ý sửa file thuộc phần người khác đang phụ trách. Nếu cần thay đổi giao diện dữ liệu hoặc API, tạo issue/trao đổi trước.
- Không đánh dấu hoàn thành nếu chưa có code, kiểm thử và bằng chứng tương ứng.
- Không tạo số liệu giả. Mọi Accuracy, Loss, thời gian chạy, hình minh họa và kết luận phải lấy từ lần chạy thực tế.
- Khi task phụ thuộc dữ liệu hoặc kết quả của thành viên khác mà chưa được bàn giao, ghi rõ trạng thái **Blocked/Đang chờ**, không tự làm thay.

## 3. Cấu trúc repository

```text
BTL03_Keras_to_PyTorch/
├── README.md                  # Hướng dẫn chung cho cả nhóm
├── requirements.txt           # Các thư viện Python
├── .gitignore                 # Loại trừ môi trường, dữ liệu lớn, checkpoint
├── data/
│   ├── README.md              # Hướng dẫn đặt dữ liệu
│   ├── mnist/                  # MNIST tải qua Torchvision
│   ├── gender/                 # Các file NPY dự đoán giới tính
│   ├── fer2013/                # Thư mục ảnh train/ và test/ (7 lớp)
│   └── segnet/
│       ├── train/depth/
│       ├── train/labels/
│       ├── test/depth/
│       └── test/labels/
├── docs/
│   └── DATA_FORMATS.md        # Định dạng và mapping nhãn
├── notebooks/                 # Notebook thử nghiệm/khám phá
├── src/
│   ├── __init__.py
│   └── datasets.py             # Dataset và hàm tạo DataLoader
├── scripts/
│   └── check_data.py           # Script kiểm tra DataLoader
├── models/                     # Kiến trúc model (bổ sung theo task)
├── checkpoints/                # Trọng số model (không commit mặc định)
├── results/
│   ├── metrics/
│   ├── plots/
│   └── logs/
├── reports/                    # Báo cáo và hình dùng trong báo cáo
└── tests/                      # Unit test/kiểm thử tích hợp
```

Một số thư mục có thể chưa xuất hiện cho đến khi task tương ứng được triển khai. Chỉ tạo thêm khi có nhu cầu thực tế.

## 4. Cài đặt môi trường

### 4.1. Yêu cầu

- Python 3.10, 3.11 hoặc 3.12 được khuyến nghị.
- Git.
- VS Code hoặc IDE Python tương đương.
- PyTorch và Torchvision. Nếu cần GPU NVIDIA, cài đúng bản PyTorch tương thích CUDA theo hướng dẫn chính thức của PyTorch; không tự chọn phiên bản CUDA theo phỏng đoán.

### 4.2. Clone repository và chuyển nhánh

```bash
git clone https://github.com/PhuDuong218/HocSau-python.git
cd HocSau-python
git checkout btl03
```

### 4.3. Tạo môi trường ảo

Trên nhánh `Duong`, dự án nằm trong thư mục `BTL03_Keras_to_PyTorch`: chạy `cd BTL03_Keras_to_PyTorch` từ gốc repository trước các lệnh cài đặt và chạy bên dưới. Trên nhánh `btl03` gốc, các file dự án hiện nằm ở gốc repository.

**Windows PowerShell:**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn kích hoạt script, có thể dùng Command Prompt:
```bat
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.4. Cài thư viện

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra PyTorch:
```bash
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Nếu `CUDA available: False`, chương trình vẫn có thể chạy bằng CPU; tốc độ sẽ phụ thuộc cấu hình máy.

## 5. Quy ước dữ liệu và tensor

PyTorch sử dụng định dạng ảnh **Channels First**:

- Một ảnh: `[C, H, W]`.
- Một batch: `[B, C, H, W]`.
- Ảnh grayscale: `C = 1`.
- Nhãn phân loại: tensor số nguyên `torch.long`.
- Segmentation mask: `[B, H, W]`, mỗi pixel là class index.

| Dataset | Đầu vào mỗi ảnh | Số lớp/đầu ra | Định dạng nguồn trong pipeline |
|---|---|---|---|
| MNIST | `[1, 28, 28]` | 10 chữ số | Torchvision MNIST |
| Gender | `[1, 64, 64]` | 2 lớp | NPY ảnh và nhãn one-hot |
| FER2013 | `[1, 48, 48]` | 7 cảm xúc | Thư mục ảnh `train/` và `test/` qua `ImageFolder` |
| Hand Depth | `[1, 96, 96]` | Mask 7 lớp | Ảnh depth và ảnh labels ghép cặp (ánh xạ LUT) |

Mapping FER2013 được pipeline hiện tại sử dụng:
`0 angry, 1 disgust, 2 fear, 3 happy, 4 neutral, 5 sad, 6 surprise`.

Mapping mask Hand Depth được ánh xạ từ 7 giá trị pixel gốc `[0, 43, 85, 128, 170, 212, 255]` về class index `0..6` qua bảng tra cứu LUT tại `src/datasets.py`. Không tự đổi thứ tự nhãn nếu chưa đối chiếu notebook Keras gốc.

## 6. Chuẩn bị dữ liệu

### 6.1. MNIST

MNIST được tải tự động qua Torchvision khi gọi loader với `download=True`. Máy cần có Internet ở lần tải đầu.

### 6.2. Human Gender

Tải [bốn file NPY từ thư mục của giảng viên](https://drive.google.com/drive/folders/0BxINLo5jshCRYW8xODhNSlkyLTQ?resourcekey=0-RNJvvSSE67Q1bMqQi3A1RQ) và đặt trong `data/gender/` với các tên sau:

```text
64_64_11938_4098_train_x_onehot.npy
64_64_11938_4098_train_y_onehot.npy
64_64_5968_4098_val_x_onehot.npy
64_64_5968_4098_val_y_onehot.npy
```

### 6.3. FER2013

Cấu trúc thư mục ảnh tại `data/fer2013/`:

```text
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Dữ liệu được nạp qua `torchvision.datasets.ImageFolder`. DataLoader dùng chung chuẩn hóa về $[-1, 1]$. Trainer CNN dùng cấu hình riêng được mô tả trong [tài liệu đối chiếu](docs/CNN_CONVERSION.md).

### 6.4. Hand Depth

Giải nén `5. Others/depth_hand.zip` trong gói `examples.zip` của giảng viên để có cấu trúc thư mục sau:

```text
data/segnet/
├── train/
│   ├── depth/
│   └── labels/
└── test/
    ├── depth/
    └── labels/
```

Mỗi ảnh depth phải có ảnh mask tương ứng. Tên file mask trong bộ dữ liệu đi kèm có phần `_labels_` tương ứng với `_depth_` trong tên ảnh depth. Pipeline tự động resize về $(96, 96)$ (ảnh depth dùng nội suy song tuyến tính, ảnh mask dùng phép lấy mẫu lân cận gần nhất NEAREST để giữ nguyên nhãn nguyên).

**Lưu ý quản lý dữ liệu:** Không commit dataset lớn, dữ liệu có bản quyền hoặc dữ liệu cá nhân lên GitHub. Pipeline đã thiết lập `.gitignore` loại trừ thư mục `data/`.

Chi tiết định dạng: xem [`docs/DATA_FORMATS.md`](docs/DATA_FORMATS.md).

**Lưu ý:** `docs/DATA_FORMATS.md` hiện còn mô tả FER2013 dạng CSV, cách xoay ảnh gender và mapping mask khác với `src/datasets.py`. Cần đối chiếu dữ liệu và notebook gốc trước khi dùng các mô tả đó cho thực nghiệm; code hiện dùng thư mục ảnh FER2013 và LUT trong `src/datasets.py`.

## 7. Sử dụng Dataset và DataLoader

Mở terminal tại thư mục gốc repository.

```python
from src.datasets import (
    make_mnist_loaders,
    make_gender_loaders,
    make_fer2013_loaders,
    make_hand_depth_loaders,
)

mnist_loaders = make_mnist_loaders("data/mnist", batch_size=128)
gender_loaders = make_gender_loaders("data/gender", batch_size=128)
fer_loaders = make_fer2013_loaders("data/fer2013", batch_size=128)
hand_loaders = make_hand_depth_loaders("data/segnet", batch_size=8)
```

Mỗi hàm trả về dictionary các split:
- MNIST: `train`, `test`.
- Gender: `train`, `validation`.
- FER2013: `train`, `validation`.
- Hand Depth: `train`, `test`.

Ví dụ lấy một batch:
```python
images, labels = next(iter(mnist_loaders["train"]))
print(images.shape, images.dtype)
print(labels.shape, labels.dtype)
```

Với Hand Depth:
```python
images, masks = next(iter(hand_loaders["train"]))
print(images.shape)  # [B, 1, 96, 96]
print(masks.shape)   # [B, 96, 96]
```

**Quan trọng:** Không dùng `CrossEntropyLoss` segmentation với mask one-hot hoặc mask RGB. Dataset hiện chuyển mask thành class index dạng `torch.long`, phù hợp với `nn.CrossEntropyLoss` khi output model có dạng `[B, 7, H, W]`.

## 8. Chạy kiểm tra dữ liệu

Sau khi đặt đủ dữ liệu, chạy từng lệnh:

```bash
python scripts/check_data.py --dataset mnist
python scripts/check_data.py --dataset gender
python scripts/check_data.py --dataset fer2013
python scripts/check_data.py --dataset hand
```

Có thể thay batch size:
```bash
python scripts/check_data.py --dataset hand --batch-size 4
```

Kết quả kiểm tra cần xem:
- Loader có tạo được hay không.
- Kích thước batch có đúng không.
- Kiểu dữ liệu ảnh/nhãn có đúng không.
- Giá trị ảnh có nằm trong khoảng hợp lý không.
- Nhãn có nằm trong khoảng class index hợp lệ không.
- Số lượng ảnh và mask có khớp không.

Nếu gặp lỗi `FileNotFoundError`, kiểm tra vị trí/tên file dữ liệu. Nếu lỗi shape, không sửa bằng cách reshape tùy tiện; hãy đối chiếu `docs/DATA_FORMATS.md` và notebook nguồn.

## CNN của Dương (T04 và T05)

- [T04 — MNIST và Gender](notebooks/02_cnn_gender.ipynb)
- [T05 — FER2013](notebooks/02_cnn_fer2013.ipynb)
- [Đối chiếu với Keras và kết quả thực nghiệm](docs/CNN_CONVERSION.md)

Để mở notebook, cài thêm `jupyterlab` trong môi trường dự án và chạy `python -m jupyterlab` từ thư mục BTL03.

Chạy từ thư mục BTL03. Các lệnh dưới đây tạo thư mục kết quả mới, gồm CSV theo epoch, JSON tổng hợp và trọng số được chọn theo validation loss.

```powershell
python -m src.train_cnn mnist --epochs 4 --optimizer adam --lr 0.001 --output-dir results/mnist_run
python -m src.train_cnn gender --epochs 30 --threads 8 --output-dir results/gender_run
python -m src.train_cnn fer2013 --epochs 12 --output-dir results/fer2013_run
```

Gender đọc bốn file NPY trong `data/gender/`. Với FER2013, đặt `fer2013_training_onehot.csv` và `fer2013_publictest_onehot.csv` từ [thư mục của giảng viên](https://drive.google.com/drive/folders/0BxINLo5jshCRbXpRcnl2SjJkdlk?resourcekey=0-7qr7Ro7hgkmFknsN1yParQ) trong `data/fer2013/` hoặc dùng hai thư mục ảnh `train/` và `test/`. Khi dùng lại ảnh và manifest của BTL02, truyền thêm hai tham số sau:

```powershell
python -m src.train_cnn fer2013 --data-dir ../BTL02_Emotion_Recognition/data/raw --split-manifest ../BTL02_Emotion_Recognition/results/training/baseline_seed42/split_manifest.csv --epochs 12 --output-dir results/fer2013_image_run
```

## 9. Quy trình làm việc cho mỗi task

Mỗi thành viên thực hiện theo quy trình thống nhất:

1. **Nhận task:** xác nhận Owner, Support, đầu vào, đầu ra và deadline trên bảng theo dõi.
2. **Kiểm tra phụ thuộc:** xác nhận dữ liệu, file hoặc module từ người khác đã được bàn giao.
3. **Tạo branch:** tạo branch riêng cho task từ nhánh `btl03`.
4. **Thực hiện:** chỉ sửa các file thuộc phạm vi task; không ghi đè thay đổi của người khác.
5. **Kiểm thử:** chạy lệnh kiểm tra liên quan và lưu log/kết quả.
6. **Commit:** commit nhỏ, có nội dung rõ ràng.
7. **Pull Request:** tạo PR về nhánh `btl03`, ghi task ID, mô tả thay đổi, cách test và ảnh/log minh chứng.
8. **Review:** nhóm trưởng Tuyết Nhi hoặc ít nhất một thành viên khác xem xét trước khi merge.
9. **Cập nhật tiến độ:** cập nhật Task Tracking, Work Log và link commit/PR.
10. **Bàn giao:** ghi rõ file đã tạo/sửa, cách chạy, kết quả test, hạn chế còn tồn tại.

### Quy tắc branch và commit

Tên branch phân công:
```text
feature/T03-mlp
feature/T04-T05-cnn
feature/T06-lstm
feature/T07-segnet
```

Commit message quy ước:
```text
feat(T03): implement MLP function approximation model and training loop
feat(T04): implement CNN architecture for MNIST and gender classification
feat(T05): convert emotion CNN with callbacks to PyTorch
feat(T06): build LSTM sequence network for MNIST summation
feat(T07): build SegNet encoder-decoder for hand depth segmentation
```

Các lệnh Git thường dùng:
```bash
git status
git pull origin btl03
git switch -c feature/<TenBranch>
git add <files>
git commit -m "feat: mo ta thay doi"
git push -u origin feature/<TenBranch>
```

Không dùng `git add .` một cách máy móc nếu thư mục có dataset, checkpoint hoặc file không thuộc task. Luôn xem `git status` trước khi commit.

## 10. Quy ước code

- Dùng `pathlib.Path` để xử lý đường dẫn.
- Không hard-code đường dẫn máy cá nhân như `C:\Users\...`.
- Tách Dataset, model, training, evaluation thành module phù hợp; không gom toàn bộ dự án vào một notebook.
- Hàm/class cần tên rõ nghĩa; thêm docstring khi cần.
- Giữ mapping nhãn thống nhất giữa Dataset, model, evaluation và demo.
- Đặt random seed và ghi cấu hình trong các thí nghiệm để tăng khả năng tái lập.
- Không áp dụng augmentation ngẫu nhiên cho validation/test.
- Không đưa Softmax vào output khi dùng `nn.CrossEntropyLoss` cho bài toán phân loại; model trả logits.
- Với segmentation, output logits có dạng `[B, num_classes, H, W]`, target class index `[B, H, W]`.

## 11. Lưu kết quả và bằng chứng

Các kết quả cần lưu theo task, ví dụ:

```text
results/
├── logs/
│   └── <experiment_id>_train.log
├── metrics/
│   └── <experiment_id>_metrics.csv
└── plots/
    ├── loss_curve.png
    ├── accuracy_curve.png
    └── confusion_matrix.png
```

Với mỗi lần chạy, ghi:
- Experiment ID.
- Dataset và split.
- Model/phiên bản code (commit hash).
- Batch size, optimizer, learning rate, epoch, seed.
- Train/validation loss và metrics.
- Thời gian chạy, thiết bị CPU/GPU.
- Đường dẫn checkpoint và hình kết quả.

Chỉ ghi kết quả đo được từ chương trình. Nếu chưa chạy, ghi `Not run` hoặc để trống; không ước lượng số liệu.

## 12. Xử lý lỗi và trao đổi nhóm

Khi gặp lỗi:
1. Lưu toàn bộ traceback.
2. Ghi câu lệnh đã chạy và thư mục làm việc.
3. Ghi phiên bản Python, PyTorch, Torchvision.
4. Tạo issue hoặc nhắn nhóm kèm file/log liên quan.
5. Không sửa dữ liệu gốc hoặc đổi mapping nhãn để né lỗi.

Khi một task bị chặn:
- Đặt trạng thái `Blocked/Đang chờ`.
- Ghi rõ đang chờ ai, cần file/kết quả gì.
- Không tự làm thay task của Owner khác nếu chưa được nhóm phân công lại.

## 13. Checklist trước khi merge và nộp bài

### Trước khi merge
- [ ] Chỉ thay đổi file thuộc phạm vi task.
- [ ] Code chạy được trên môi trường đã ghi trong README.
- [ ] Đã chạy test/lệnh kiểm tra liên quan.
- [ ] Không commit dataset, checkpoint lớn, token hoặc mật khẩu.
- [ ] Mapping nhãn và shape tensor được giữ nhất quán.
- [ ] PR có mô tả và bằng chứng.

### Trước khi nộp
- [ ] Có source code Keras-to-PyTorch theo đủ phạm vi đề yêu cầu.
- [ ] Có hướng dẫn cài đặt và chạy.
- [ ] Có báo cáo và các hình/bảng lấy từ thực nghiệm.
- [ ] Có file liên quan mà giảng viên yêu cầu.
- [ ] Đã kiểm tra lại đường dẫn và cấu trúc file nộp.
- [ ] Xác nhận tất cả thành viên đã bàn giao phần việc.

## 14. Trạng thái và giới hạn hiện tại

README này mô tả cách sử dụng repository và quy trình làm việc; nó không xác nhận tất cả task BTL03 đã hoàn thành.

Tại thời điểm hiện tại:
- **T01 & T02 (Hoàn thành):** Đã hoàn thiện cấu trúc dự án, môi trường, và Data Pipeline cho toàn bộ 4 bộ dữ liệu (MNIST, Gender, FER2013 dạng ImageFolder, SegNet ánh xạ LUT 7 lớp qua `scripts/check_data.py`).
- **T04 & T05:** Đã chuyển đổi MNIST CNN, Gender CNN và FER2013 CNN. Xem [kết quả thực nghiệm](docs/CNN_CONVERSION.md) để đối chiếu các mốc nghiệm thu.

## 15. Tài liệu tham khảo trong repository

- `docs/DATA_FORMATS.md`: chi tiết tên file, cấu trúc dữ liệu và mapping mask.
- `src/datasets.py`: implementation Dataset/DataLoader hiện tại.
- `scripts/check_data.py`: kiểm tra nhanh các DataLoader.
- Tài liệu bài giảng và notebook Keras do giảng viên cung cấp: dùng làm nguồn đối chiếu khi chuyển đổi; không tự thay đổi hành vi của ví dụ gốc mà không ghi chú.
