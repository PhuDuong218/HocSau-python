# Tạo lại baseline checkpoint từ FER2013

Đây là lần huấn luyện mới của Xuân Anh theo kiến trúc và siêu tham số baseline của Nhi. Không phải checkpoint gốc của Nhi và không được gán mặc định Accuracy 48,34%.

## Nguồn dữ liệu và kiểm tra

- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
- Endpoint tải: https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013
- Archive: `data/downloads/fer2013.zip`, không đưa lên Git.
- Đối chiếu với slide Emotion_Recognition.pdf: 28.709 train, 7.178 test (3.589 PublicTest và 3.589 PrivateTest), 7 lớp, ảnh grayscale 48 × 48.
- `python src/dataset.py` kiểm tra CRC của ZIP, giải mã toàn bộ ảnh, đối chiếu số lượng từng lớp, tính SHA-256 file và pixel, báo cáo ảnh trùng hoặc có nhãn mâu thuẫn.
- Manifest và kết quả kiểm tra nằm ở `results/data_audit/`.

## Phương pháp huấn luyện

- BaselineCNN lấy từ `02_baseline.ipynb`, 1.274.823 tham số học được; key và shape trọng số được giữ nguyên.
- Adam, learning rate 0,001, batch size 128, CrossEntropyLoss, 3 epoch, CPU 4 threads, seed 42.
- Train: grayscale, đổi uint8 thành float /255, horizontal flip độc lập p=0,5 mỗi ảnh, Normalize mean=0,5/std=0,5. Dữ liệu được cache dạng tensor uint8 để tránh đọc JPEG lại mỗi epoch.
- Validation: cùng chuẩn hóa, không tăng cường ngẫu nhiên.
- Khác notebook cũ: chọn checkpoint theo validation loss thấp nhất, không dùng điểm test để chọn mô hình.
- Validation được lấy từ 15% nhóm ảnh train có pixel khác nhau, stratify theo nhãn với seed 42. Mọi bản sao cùng pixel nằm trong cùng một tập. Số ảnh thực tế có thể không đúng 15% vì nhóm có kích thước khác nhau.
- Loại khỏi train/validation các ảnh có pixel trùng với test và các nhóm train có nhãn mâu thuẫn. Không xóa ảnh gốc. Các trường hợp loại được ghi trong split_manifest.csv.
- Kiểm tra trùng chỉ phát hiện ảnh có pixel giải mã giống hoàn toàn, không loại được toàn bộ ảnh gần giống hoặc xác minh danh tính người trong ảnh.

Chạy từ thư mục gốc repo:

```bat
python src/dataset.py
python -u src/train.py --epochs 3 --seed 42 --threads 4
```

Trainer từ chối ghi đè checkpoint hoặc lịch sử chạy đã tồn tại.

## Sản phẩm và xác minh

- `checkpoints/baseline_best.pth`: state_dict thật từ epoch có validation loss thấp nhất.
- `checkpoints/baseline_best.metadata.json`: mapping nhãn, seed, hash archive, cách chia tập, epoch được chọn, metric validation và SHA-256 của checkpoint.
- `results/training/baseline_seed42/history.csv`: loss/accuracy theo epoch và Macro-F1 validation.
- `results/training/baseline_seed42/split_manifest.csv`: từng file ảnh, hash, nhãn và tập được gán, kể cả ảnh bị loại khỏi train/validation.
- `results/training/baseline_seed42/run_metadata.json` và `learning_curve.png`.
- Trainer kiểm tra loss/gradient hữu hạn, sau khi train nạp lại checkpoint bằng strict=True.

Sau khi train hoàn tất, đánh giá một lần trên test:

```bat
python src/evaluate.py --output-dir results/evaluation/baseline_retrained --independent-holdout
```

Evaluator xác minh hash checkpoint và mapping nhãn với metadata. Số liệu lần train mới phải được lưu riêng với các kết quả baseline cũ của Nhi; không trộn hai lần chạy trong báo cáo.

## Bàn giao

Git đang bỏ qua checkpoint `.pth` và ảnh thô. Khi cần chuyển checkpoint cho nhóm, gửi cả `baseline_best.pth` và `baseline_best.metadata.json` cùng link nguồn dữ liệu. Có thể push source code, manifest, metadata và kết quả đánh giá lên nhánh cá nhân. Không tự động push dữ liệu hoặc checkpoint.
