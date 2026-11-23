# BÀI TẬP LỚN 02: FACIAL EMOTION RECOGNITION (FER2013)
**Môn học:** Học sâu (Deep Learning)
**Framework:** PyTorch
**Môi trường thực thi:** Conda environment (`fer_env`), Python 3.10+
**Kho mã nguồn:** [GitHub PhuDuong218/HocSau-python](https://github.com/PhuDuong218/HocSau-python)
**Bảng điều phối dự án (Task Tracking & Work Log):** [Google Sheets - Emotion Recognition Project](https://docs.google.com/spreadsheets/d/1VQ-_o1qQzwpZ_4W0habN7_9UEAgdagkDPqxXh7H4YrU/edit?gid=634460577#gid=634460577)

---

## 1. Giới thiệu đề tài
Dự án xây dựng mô hình Deep Learning nhận diện cảm xúc khuôn mặt từ ảnh mức xám (grayscale) kích thước $48 \times 48$ pixel dựa trên tập dữ liệu chuẩn quốc tế **FER2013**.
Hệ thống giải quyết bài toán phân loại đa lớp (Multi-class Classification) với 7 biểu cảm cảm xúc cơ bản của con người:
* `0: Angry` (Tức giận)
* `1: Disgust` (Ghê tởm)
* `2: Fear` (Sợ hãi)
* `3: Happy` (Hạnh phúc / Vui vẻ)
* `4: Sad` (Buồn bã)
* `5: Surprise` (Ngạc nhiên)
* `6: Neutral` (Trung tính / Bình thường)

**Nguyên tắc số liệu:** Kết quả 48,34% và 44,30% dưới đây là output của các notebook cũ. Checkpoint hiện có trong `checkpoints/` thuộc lần huấn luyện lại, có metadata và kết quả riêng trong `results/training/` và `results/evaluation/baseline_retrained/`. Không gộp số liệu của hai lần chạy.

---

## 2. Thông tin Dataset & Hiện trạng thực nghiệm

### 2.1. Thống kê tập dữ liệu FER2013
* **Tổng số mẫu:** 35.887 ảnh ($48 \times 48$, Grayscale).
* **Tập Huấn luyện (Train):** 28.709 ảnh.
* **Tập Kiểm thử (Test):** 7.178 ảnh (hợp nhất từ Public Test và Private Test).
* **Đặc tính dữ liệu:**
  * **Class Imbalance nặng:** Lớp `Happy` áp đảo với 8.989 ảnh (25.05%), trong khi lớp `Disgust` chỉ có 547 ảnh (1.52%, tập test chỉ có 111 ảnh).
  * **Thách thức thị giác:** Ảnh độ phân giải thấp, nhiều góc nghiêng ($30^\circ - 45^\circ$), một số mẫu bị che khuất một phần bởi tay hoặc kính mắt.

### 2.2. Kết quả thực nghiệm cốt lõi
| Tiêu chí | Mô hình Baseline CNN (Nhóm đề xuất) | Mô hình VGG16 (Nghiên cứu Fail Case) |
| :--- | :--- | :--- |
| **Kiến trúc** | 3 khối Conv2D-BatchNorm-ReLU-MaxPool + FC 256 | 5 khối Conv2D liên tiếp (13 Conv + 5 MaxPool) |
| **Số lượng tham số** | Khoảng 1,27 triệu tham số | Khoảng 15,25 triệu tham số với classifier đã thay trong notebook |
| **Epoch 1** | Train Loss: 1.7818 \| Test Acc: 39.73% | Train Loss: 1.7959 \| Test Acc: 26.83% |
| **Epoch 2** | Train Loss: 1.5566 \| Test Acc: 43.22% | Train Loss: 1.6669 \| Test Acc: 42.02% |
| **Epoch 3** | Train Loss: 1.4623 \| **Test Acc: 48.34%** | Train Loss: 1.4882 \| **Test Acc: 44.30%** |
| **Kích thước Feature Map cuối** | $6 \times 6$ pixel (Bảo toàn cấu trúc khuôn mặt) | $1 \times 1$ pixel (Bị suy thoái không gian) |
| **Giới hạn kết luận** | Kết quả cũ sau 3 epoch, chưa tách validation riêng | Feature map còn $1 \times 1$ sau 5 lần pooling; kết quả 3 epoch chưa đủ để khẳng định nguyên nhân giảm độ chính xác hay overfitting |

---

## 3. Phân công trách nhiệm 4 thành viên

Theo bảng phân công trên [Google Sheet Quản lý](https://docs.google.com/spreadsheets/d/1VQ-_o1qQzwpZ_4W0habN7_9UEAgdagkDPqxXh7H4YrU/edit?gid=634460577#gid=634460577):

| Thành viên | Vai trò chuyên môn | Nhiệm vụ kỹ thuật & Code | Nhiệm vụ Báo cáo & Tài liệu |
| :--- | :--- | :--- | :--- |
| **Tuyết Nhi** | Model Lead & Experiment Analyst | - Xây dựng kiến trúc Baseline CNN (3 Conv blocks).<br>- Huấn luyện baseline, lưu checkpoint `baseline_best.pth`.<br>- Thực nghiệm phân tích Fail Case VGG16. | - Soạn thảo Mục 2 (Kiến trúc & Training Baseline).<br>- Soạn thảo Mục 4 (Phân tích thất bại của VGG16).<br>- Trích xuất và giải thích bộ lọc tích chập (Filter Visualization). |
| **Phú Dương** | Data Lead & Repository Manager | - Khám phá dữ liệu (EDA), phân tích phân bố lớp.<br>- Xây dựng pipeline Data Augmentation & DataLoader.<br>- Quản lý repository GitHub và file `.gitignore`. | - Soạn thảo Mục 1 (Data Processing, phân tích mất cân bằng lớp Happy/Disgust).<br>- Viết tài liệu hướng dẫn cài đặt trong `README.md`. |
| **Xuân Anh** | Evaluation Lead & Report Editor | - Thiết lập Evaluation Pipeline.<br>- Tính toán Accuracy, Precision, Recall, Macro-F1.<br>- Phân tích Confusion Matrix và True/Wrong Cases. | - Soạn thảo Mục 3 (Evaluations & Phân tích lỗi).<br>- Ghép nối các phần báo cáo, dàn trang, chèn 7 hình minh chứng.<br>- Xuất bản file `BT02_Report.pdf` nộp bài. |
| **Quốc Việt** | UI/Demo Lead, Slide & Log Manager | - Cập nhật dữ liệu vào Experiment Log và Work Log.<br>- Thiết kế ứng dụng Demo nhận diện cảm xúc (Streamlit/Webcam).<br>- Chuẩn bị video demo dự phòng. | - Thiết kế bộ Slide thuyết trình chuẩn 12–15 slide.<br>- Soạn thảo Mục 5 (So sánh mô hình) & Mục 6 (Kết luận & Hướng phát triển). |

---

## 4. Cấu trúc thư mục dự án

```text
BT02_Emotion_Recognition/
├── README.md                                    # Hướng dẫn chi tiết dự án & quy trình làm việc
├── requirements.txt                             # Danh sách thư viện Python cần thiết
├── .gitignore                                   # Loại trừ dữ liệu thô, checkpoint nặng và cache
├── data/
│   ├── raw/                                     # Dữ liệu gốc FER2013 (train/, test/) - Không đẩy lên Git
│   └── processed/                               # Dữ liệu sau xử lý (nếu có)
├── checkpoints/
│   └── baseline_best.pth                        # Checkpoint của lần huấn luyện lại; xem metadata đi kèm
├── notebooks/
│   ├── 01_eda.ipynb                             # Khám phá dữ liệu, vẽ biểu đồ phân bố và ảnh mẫu
│   ├── 02_baseline.ipynb                        # Huấn luyện Baseline CNN và lưu đồ thị suy giảm Loss
│   └── 03_experiments_and_evaluation.ipynb      # Đánh giá chi tiết, phân loại sai, Filter Viz & VGG16
├── results/
│   └── plots/                                   # 7 hình ảnh minh chứng phục vụ báo cáo:
│       ├── dataset_distribution.png             # Hình 1.1: Biểu đồ cột phân bố 7 nhãn
│       ├── sample_faces.png                     # Hình 1.2: Lưới ảnh mẫu 48x48
│       ├── baseline_learning_curve.png          # Hình 2.1: Đường cong Loss và Acc của Baseline
│       ├── baseline_confusion_matrix.png        # Hình 3.1: Ma trận nhầm lẫn 7 lớp
│       ├── final_conv_filters.png               # Hình 3.2: 16 bộ lọc của tầng Conv cuối
│       ├── true_predict_images.png              # Hình 3.3: Lưới 32 ảnh dự đoán đúng
│       ├── wrong_predict_images.png             # Hình 3.4: Lưới 32 ảnh dự đoán sai
│       └── vgg16_fail_curve.png                 # Hình 4.1: Đường cong học tập của VGG16
└── docs/
    ├── BT02_Report_XuanAnh_v2.docx              # Bản Word hiện có
    ├── BT02_Report.pdf                          # File báo cáo PDF chính thức
    └── TRAINING_BASELINE.md                     # Hướng dẫn lần huấn luyện lại
```

---

## 5. Hướng dẫn thiết lập môi trường & Chạy thử nghiệm

### Bước 1: Vào thư mục dự án và kích hoạt môi trường
Chạy các lệnh sau từ thư mục gốc của repository:
```bash
cd BTL02_Emotion_Recognition
conda activate fer_env
```
Nếu chưa có môi trường, tạo môi trường Python trước khi kích hoạt. Cài các thư viện theo tệp của dự án:
```bash
conda create -n fer_env python=3.11
conda activate fer_env
python -m pip install -r requirements.txt
```
`requirements.txt` hiện ghim bản PyTorch cho CUDA 13.0; kiểm tra khả năng tương thích với GPU và driver trên máy trước khi cài. Nếu chỉ cần chạy phần đánh giá trong môi trường PyTorch đã có, xem `requirements-evaluation.txt`.

### Bước 2: Chuẩn bị dữ liệu
Tải tập dữ liệu FER2013 và giải nén theo đúng cấu trúc:
```text
BT02_Emotion_Recognition/data/raw/
├── train/
│   ├── angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/
└── test/
    ├── angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/
```

### Bước 3: Kiểm tra dữ liệu và chạy EDA
1. Chuẩn bị dữ liệu trong `data/raw/train/` và `data/raw/test/`. Nếu có bản CSV gốc, chạy `python src/prepare_fer2013.py` sau khi đặt `fer2013.csv` vào `data/raw/`.
2. Chạy `python src/eda.py` hoặc `notebooks/01_eda.ipynb` để tạo lại biểu đồ phân phối từ `results/data_audit/dataset_audit.json`. Lưới ảnh mẫu chỉ được tạo lại khi có ảnh gốc trong `data/raw/train/`.
3. Xem `docs/TRAINING_BASELINE.md` để biết cách kiểm định dữ liệu, huấn luyện lại và đánh giá checkpoint hiện có. Không chạy lại `src/train.py` khi chưa kiểm tra checkpoint và kết quả cũ vì chương trình từ chối ghi đè.
4. `notebooks/02_baseline.ipynb` ghi lại thực nghiệm baseline cũ. `notebooks/03_experiments_and_evaluation.ipynb` dành cho phân tích và đánh giá; xem thêm `docs/EVALUATION_GUIDE.md` và bản thảo phần Dương trong `docs/DUONG_HANDOFF.md`.

---

## 6. Quy chuẩn phối hợp Git & Nhánh làm việc

1. **Tuyệt đối không push file nặng lên Git:**
   * File `.gitignore` đã chặn dữ liệu trong `data/raw/`, các checkpoint `checkpoints/*.pth` và thư mục cache `__pycache__/`, `.ipynb_checkpoints/`. Tệp checkpoint đã được Git theo dõi trước đó vẫn có thể xuất hiện trong commit; kiểm tra `git status` trước khi push.
2. **Quy ước nhánh (Branching):**
   * Không commit trực tiếp vào `main`.
   * Mỗi thành viên tạo nhánh riêng tương ứng với task:
     * `Nhi`: Đẩy code huấn luyện Baseline, Fail Case VGG16.
     * `Duong`: Đẩy code EDA, DataLoader, Augmentation.
     * `XuanAnh`: Đẩy phần evaluation và hoàn thiện tài liệu báo cáo.
     * `Viet`: Đẩy code Demo và Slide thuyết trình.
3. **Cú pháp commit rõ ràng:**
   * `feat(T05): hoan thanh pipeline tien xu ly va Data Augmentation`
   * `feat(T08): huan luyen xong baseline CNN dat 48.34% acc`
   * `docs(T16): cap nhat bao cao phan tich fail case VGG16`

---

## 7. Checklist nghiệm thu và bàn giao

- [x] Dương: khôi phục notebook EDA và bổ sung code tạo lại biểu đồ, ảnh mẫu khi có dữ liệu gốc.
- [x] Dương: chuẩn bị bản thảo W02 và W06 có nguồn số liệu, giới hạn kết luận trong `docs/DUONG_HANDOFF.md`.
- [ ] Dương: chạy EDA trên ảnh thật, đối chiếu hình mới và xác nhận phần bàn giao.
- [ ] Dương: cập nhật Work Log/Report Tracking bằng ngày, thời gian và link minh chứng thực tế.
- [ ] Dương: kiểm tra hướng dẫn cài đặt trên máy dùng để nộp và hoàn tất T19.
- [ ] Nhóm: rà soát báo cáo, demo, slide, các số liệu giữa những lần chạy và sản phẩm nộp.
