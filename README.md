# BÀI TẬP LỚN 02: FACIAL EMOTION RECOGNITION (FER2013)

## Cập nhật phần Evaluation của Xuân Anh

- Kết quả checkpoint mới: Accuracy **48,41%**, Macro-F1 **0,3793** trên 7.178 ảnh test; chọn epoch bằng validation loss.
- Đánh giá từ checkpoint đã bàn giao: `python src/evaluate.py --output-dir results/evaluation/baseline_retrained` (cần tải dữ liệu ảnh).
- `python src/analyze_saved_results.py` chỉ phân tích kết quả cũ của Nhi, không phải lần chạy mới.
- Hướng dẫn: [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md).
- Danh sách đã làm và còn thiếu: [docs/XUANANH_TASK_STATUS.md](docs/XUANANH_TASK_STATUS.md).
- Báo cáo biên tập: [Word](docs/BT02_Report_XuanAnh_v2.docx), [PDF](docs/BT02_Report.pdf).

Phần bàn giao mới gồm code, checkpoint và metadata, log, manifest chia tập, kết quả đánh giá cùng Word/PDF có 7 hình minh chứng. Mapping là `angry, disgust, fear, happy, neutral, sad, surprise`. Notebook cũ của Nhi dùng test để chọn checkpoint; lần train lại của Xuân Anh dùng validation riêng và xử lý trùng pixel giữa các tập. Các bảng số liệu và checklist cũ bên dưới được giữ làm tài liệu nhóm ban đầu; kết quả mới lấy theo báo cáo v2 và `results/evaluation/baseline_retrained/`. VGG16 hiện chỉ là số liệu tham khảo của nhóm, chưa có log để tái kiểm chứng.

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

**Nguyên tắc số liệu:** Toàn bộ số liệu trong báo cáo, slide và README được trích xuất từ các lần chạy thực tế trong repository (`results/plots/`, `checkpoints/baseline_best.pth`), không dùng dữ liệu mô phỏng.

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
| **Số lượng tham số** | Gọn nhẹ (~1.25 triệu tham số) | Rất nặng (>130 triệu tham số) |
| **Epoch 1** | Train Loss: 1.7818 \| Test Acc: 39.73% | Train Loss: 1.7959 \| Test Acc: 26.83% |
| **Epoch 2** | Train Loss: 1.5566 \| Test Acc: 43.22% | Train Loss: 1.6669 \| Test Acc: 42.02% |
| **Epoch 3** | Train Loss: 1.4623 \| **Test Acc: 48.34%** | Train Loss: 1.4882 \| **Test Acc: 44.30%** |
| **Kích thước Feature Map cuối** | $6 \times 6$ pixel (Bảo toàn cấu trúc khuôn mặt) | $1 \times 1$ pixel (Bị suy thoái không gian) |
| **Kết luận kỹ thuật** | Phù hợp với ảnh $48 \times 48$, hội tụ nhanh | Thất bại do 5 tầng MaxPool làm teo feature map về $1 \times 1$ và overfit do dư thừa tham số |

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
│   └── baseline_best.pth                        # Trọng số tốt nhất của Baseline CNN (Acc 48.34%)
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
├── demo/
│   └── app.py                                   # Mã nguồn ứng dụng demo nhận diện cảm xúc
└── docs/
    ├── Bao_Cao_Nhan_Dien_Cam_Xuc_FER2013.docx   # File báo cáo Word hoàn chỉnh
    ├── BT02_Report.pdf                          # File báo cáo PDF chính thức
    └── BT02_Presentation.pptx                   # Slide thuyết trình nhóm
```

---

## 5. Hướng dẫn thiết lập môi trường & Chạy thử nghiệm

### Bước 1: Khởi tạo môi trường Conda
Mở terminal và kích hoạt môi trường đã cấu hình:
```bash
conda activate fer_env
```
*(Nếu cài trên máy mới chưa có thư viện)*:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib seaborn pandas scikit-learn pillow streamlit
```

### Bước 2: Chuẩn bị dữ liệu
Tải tập dữ liệu FER2013 và giải nén theo đúng cấu trúc:
```text
BT02_Emotion_Recognition/data/raw/
├── train/
│   ├── angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/
└── test/
    ├── angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/
```

### Bước 3: Chạy các Notebook theo thứ tự
1. Chạy `notebooks/01_eda.ipynb`: Kiểm tra dữ liệu và tự động xuất ảnh `dataset_distribution.png`, `sample_faces.png`.
2. Chạy `notebooks/02_baseline.ipynb`: Huấn luyện mạng Baseline 3 epoch, lưu trọng số vào `checkpoints/baseline_best.pth` và xuất `baseline_learning_curve.png`.
3. Chạy `notebooks/03_experiments_and_evaluation.ipynb`: Nạp checkpoint, tính Classification Report, xuất ma trận nhầm lẫn, trực quan hóa 16 kernel tích chập, xuất lưới ảnh True/Wrong Cases và thực nghiệm đối chứng Fail Case với VGG16.

---

## 6. Quy chuẩn phối hợp Git & Nhánh làm việc

1. **Tuyệt đối không push file nặng lên Git:**
   * File `.gitignore` đã chặn các file trong `data/raw/`, `checkpoints/*.pth` lớn hơn 100MB và thư mục cache `__pycache__/`, `.ipynb_checkpoints/`.
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

- [x] T01 - T03: Cấu trúc dự án, thu thập dữ liệu và phân tích EDA.
- [x] T05 - T06: Xây dựng pipeline tiền xử lý, Data Augmentation và tối ưu DataLoader.
- [x] T07 - T08: Thiết kế kiến trúc Baseline CNN, train đạt 48.34% và lưu checkpoint.
- [x] T09 - T10: Đánh giá mô hình Baseline qua Classification Report và Confusion Matrix.
- [x] T11 - T12: Thực nghiệm đối chứng Fail Case VGG16 (đạt 44.30%).
- [x] T14: Trực quan hóa 16 kernel tích chập, trích xuất lưới 32 ảnh đúng và 32 ảnh sai.
- [ ] T15: Hoàn thiện giao diện demo nhận diện cảm xúc.
- [ ] T16 - T18: Ghép nối nội dung báo cáo Word, chèn đủ 7 biểu đồ và xuất bản PDF.
- [ ] T17: Hoàn thiện 12–15 slide thuyết trình.
- [ ] T19 - T20: Rà soát rubric chấm điểm, dọn dẹp repo và đóng gói file nén nộp bài.
