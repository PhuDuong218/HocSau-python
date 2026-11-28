# BTL02 Emotion Recognition

Ứng dụng nhận diện bảy cảm xúc từ ảnh khuôn mặt bằng PyTorch trên bộ FER2013. Ảnh đầu vào được chuyển sang grayscale 48×48; mô hình trả xác suất cho Angry, Disgust, Fear, Happy, Neutral, Sad và Surprise.

## Cấu trúc dự án

- src/: chuẩn bị dữ liệu, EDA, mô hình, huấn luyện, đánh giá và dự đoán một ảnh.
- notebooks/: notebook EDA và các thực nghiệm của nhóm.
- checkpoints/: trọng số và metadata đi kèm.
- results/: audit dữ liệu, log huấn luyện, chỉ số đánh giá và biểu đồ.
- docs/: tài liệu và báo cáo của nhóm.
- data/raw/: ảnh train/test; dữ liệu gốc không đưa lên Git.

## Cài đặt trên Windows

Dùng Python 3.12. Mở PowerShell trong thư mục repository rồi chạy:

```powershell
cd BTL02_Emotion_Recognition
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Cài bản CPU để chạy trên máy không cần GPU NVIDIA:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.15.0+cpu torchvision==0.29.2+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt pandas jupyterlab
```

Nếu dùng GPU NVIDIA tương thích CUDA 13.0, cài cấu hình của nhóm bằng requirements.txt thay cho các lệnh cài CPU:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kiểm tra môi trường:

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

Nếu đã có môi trường Python hoặc Conda với các thư viện trên, dùng Python của môi trường đó thay cho đường dẫn .venv. Mở thư mục BTL02_Emotion_Recognition trong VS Code và chọn đúng Python interpreter/kernel.

## Chuẩn bị dữ liệu

Tải FER2013 từ [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). Đặt file nén ở data/downloads/fer2013.zip rồi chạy để kiểm tra archive, giải mã ảnh và tạo manifest:

```powershell
.\.venv\Scripts\python.exe src/dataset.py
```

Kết quả ảnh có cấu trúc:

```text
data/raw/
  train/angry, disgust, fear, happy, neutral, sad, surprise
  test/angry, disgust, fear, happy, neutral, sad, surprise
```

Bộ gốc gồm 28.709 ảnh train và 7.178 ảnh test. Nếu có FER2013 CSV thay cho ZIP ảnh, dùng src/prepare_fer2013.py với tham số --input và --output để chuyển sang thư mục ảnh; quy trình audit/train hiện tại của nhóm dùng ZIP ảnh Kaggle nói trên.

Tạo biểu đồ phân phối và lưới 35 ảnh mẫu:

```powershell
.\.venv\Scripts\python.exe src/eda.py
```

EDA dùng results/data_audit/dataset_audit.json và đối chiếu số lượng từng lớp với ảnh cục bộ khi có dữ liệu. Biểu đồ được lưu vào results/plots/.

## Dự đoán một ảnh

Đặt cặp baseline_best.pth và baseline_best.metadata.json trong checkpoints/. Dùng ảnh đã cắt khuôn mặt:

```powershell
.\.venv\Scripts\python.exe src/predict_image.py "C:\duong_dan\khuon_mat.jpg"
```

Chương trình tự chuyển grayscale, resize 48×48, chuẩn hóa và in xác suất bảy lớp. SHA-256 của checkpoint phải khớp metadata. Có thể truyền --checkpoint và --metadata để chọn cặp tệp khác. Ảnh đầu vào cần được cắt khuôn mặt trước; lệnh này không phát hiện vị trí khuôn mặt.

## Đánh giá mô hình

```powershell
.\.venv\Scripts\python.exe src/evaluate.py --output-dir results/evaluation/baseline
```

Kết quả gồm các chỉ số, classification report, confusion matrix, dự đoán từng ảnh, ví dụ đúng/sai và hình bộ lọc cuối. Chỉ dùng --independent-holdout khi tập đánh giá chưa được dùng để chọn mô hình hoặc siêu tham số. Metadata checkpoint được đối chiếu với dữ liệu và cách đánh giá.

## Huấn luyện và notebook

Sau khi chạy audit dữ liệu:

```powershell
.\.venv\Scripts\python.exe src/train.py --epochs 3 --seed 42 --threads 4
.\.venv\Scripts\python.exe -m jupyterlab
```

Trainer tách validation theo nhóm pixel với seed cố định, loại trùng pixel với test và chọn checkpoint theo validation loss. Chương trình từ chối ghi đè checkpoint hoặc kết quả đã tồn tại; sao lưu lần chạy cũ trước khi tạo thí nghiệm mới. Xem docs/TRAINING_BASELINE.md để biết chi tiết quy trình.

Mở notebooks/01_eda.ipynb để chạy EDA. Hai notebook 02 và 03 lưu các thực nghiệm của nhóm; những cell dùng đường dẫn ../data hoặc ../checkpoints cần chạy với thư mục làm việc là notebooks. Kết quả baseline cũ trong notebook và kết quả baseline_retrained là các lần chạy khác nhau, cần dùng đúng checkpoint và log tương ứng.
