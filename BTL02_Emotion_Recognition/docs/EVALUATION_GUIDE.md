# Hướng dẫn phần Evaluation — Xuân Anh

## Checkpoint mới đã được huấn luyện

Lần chạy mới tạo `checkpoints/baseline_best.pth` và metadata đi kèm, chọn theo validation loss. Xem `docs/TRAINING_BASELINE.md` để biết nguồn dữ liệu, xử lý ảnh trùng, cách chia tập và log. Đây là checkpoint mới của Xuân Anh, không phải file trọng số gốc của Nhi.

`evaluate.py` tự đọc metadata, kiểm tra hash/mapping và dùng đúng mô tả chọn mô hình. Phần hướng dẫn về tập test đã tham gia chọn checkpoint bên dưới áp dụng cho notebook baseline cũ của Nhi, không áp dụng cho lần train lại dùng validation riêng.

## Khi chưa có checkpoint

Chạy `python src/analyze_saved_results.py` từ gốc repo hoặc phần A trong notebook đánh giá. Kết quả ghi ở `results/evaluation/saved_baseline/`, được tính từ số đếm trên ảnh confusion matrix đã có và đối chiếu output notebook. Đây là phân tích bằng chứng đã lưu, không phải lần test mới và không xuất được danh tính từng ảnh đúng/sai.

## Đầu vào

- `checkpoints/baseline_best.pth`: checkpoint thật do Nhi gửi, đúng kiến trúc trong `02_baseline.ipynb`.
- `data/raw/test/<tên cảm xúc>/*`: đúng bộ ảnh Nhi đã dùng.
- Nhãn theo `ImageFolder`: angry, disgust, fear, happy, neutral, sad, surprise. Đây là thứ tự trong output notebook, khác danh sách số nhãn FER2013 gốc ở README.
- Ảnh 48 × 48, grayscale; chuẩn hóa mean=0.5/std=0.5; không augmentation khi đánh giá.

Checkpoint và ảnh thô không được lưu trên Git; bản clone mới vẫn cần tải dữ liệu và nhận checkpoint, hoặc huấn luyện lại. Không thay bằng checkpoint ngẫu nhiên để lấy số liệu báo cáo.

## Chạy

Từ thư mục gốc `HocSau-python`:

```bat
python src/evaluate.py
```

Hoặc mở `notebooks/03_experiments_and_evaluation.ipynb`, chọn kernel Python đã cài torch/torchvision và chạy lần lượt. Notebook không huấn luyện lại.

Đường dẫn tùy chọn:

```bat
python src/evaluate.py --checkpoint "D:\models\baseline_best.pth" --data-dir "D:\fer2013\test"
```

## Sản phẩm

Kết quả nằm riêng tại `results/evaluation/baseline/`, không ghi đè các hình cũ của Nhi:

- `metrics.json`: số liệu, mapping nhãn, nguồn checkpoint và SHA-256.
- `classification_report.txt`, `.csv`: precision/recall/F1/support từng lớp và trung bình.
- `predictions.csv`: từng ảnh, nhãn thật, nhãn dự đoán, xác suất và đúng/sai.
- `confusion_matrix.csv`, `.png`, `confusion_matrix_normalized.png`.
- `true_predict_images.png`, `wrong_predict_images.png`: tối đa 32 mẫu mỗi nhóm, lấy mẫu không lặp với seed cố định.
- `final_conv_filters.png`: 16 bộ lọc conv3 đầu tiên, lấy trung bình trọng số có dấu qua 64 kênh đầu vào. Đây là trọng số filter, không phải feature map hay Grad-CAM.
- `evaluation_summary.md`: chỉ số và các cặp nhãn nhầm nhiều nhất từ kết quả chạy thật.

## Cách diễn giải trong báo cáo

Notebook baseline dùng `test_loader` sau mỗi epoch để chọn checkpoint. Vì vậy số liệu trên cùng tập đó không phải kết quả test độc lập; giữ ghi chú này trong báo cáo. Nếu nhóm muốn test độc lập, phải chốt lại cách tách train/validation/test và huấn luyện/chọn model bằng validation.

Chỉ dùng `--independent-holdout` khi bộ dữ liệu cung cấp thực sự chưa được dùng để chọn model hoặc tham số. Đổi tên tập không khiến kết quả trở thành test độc lập.

Khi viết Mục 3: trình bày chỉ số tổng thể, phân tích precision/recall theo lớp, giải thích confusion matrix, rồi xem ảnh đúng/sai trước khi nêu giả thuyết nguyên nhân. Không kết luận overfitting hoặc nguyên nhân thất bại chỉ từ ba epoch hay một con số accuracy.
