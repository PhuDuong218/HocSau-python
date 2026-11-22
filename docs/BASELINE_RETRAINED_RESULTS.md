# Kết quả baseline huấn luyện lại của Xuân Anh

Đây là lần chạy mới, không phải checkpoint gốc của Nhi.

## Kết quả đã chạy thật

- Epoch được chọn: 3 / 3 theo validation loss thấp nhất.
- Validation accuracy: 47.10%
- Test accuracy: 48.41%
- Test Macro-F1: 0.3793
- Test Weighted-F1: 0.4613
- Test cross-entropy: 1.3401
- Số ảnh test: 7178

## Kiểm tra và phương pháp

Toàn bộ 35.887 ảnh Kaggle đã được giải mã và kiểm tra grayscale 48 × 48. Đã loại khỏi phần học 703 ảnh train trùng pixel với test và 73 ảnh thuộc nhóm train có nhãn mâu thuẫn. Còn 23.748 train và 4.185 validation; test giữ nguyên 7.178 ảnh. Không có pixel hash giao nhau giữa ba tập. Không dùng điểm test để chọn checkpoint trong lần chạy này.

Kiến trúc khớp BaselineCNN của Nhi, 1.274.823 tham số học được; Adam lr=0,001, batch 128, seed 42, CPU 4 threads, 3 epoch. Nạp lại checkpoint strict=True thành công; mọi trọng số hữu hạn; đã đối chiếu log, hash và tính lại Accuracy/Macro-F1 từ predictions.csv.

## Bàn giao

- checkpoints/baseline_best.pth
- checkpoints/baseline_best.metadata.json
- results/training/baseline_seed42/history.csv
- results/training/baseline_seed42/checkpoint_verification.json
- results/evaluation/baseline_retrained/metrics.json
- results/evaluation/baseline_retrained/predictions.csv
- results/evaluation/baseline_retrained/confusion_matrix.png
- results/evaluation/baseline_retrained/true_predict_images.png
- results/evaluation/baseline_retrained/wrong_predict_images.png
- results/evaluation/baseline_retrained/final_conv_filters.png

## Giới hạn

Baseline mới train 3 epoch, chưa phải mô hình tối ưu. Disgust chưa được dự đoán thành công trong lần đánh giá này; không suy diễn nguyên nhân chỉ từ một lần chạy. Bộ test vẫn có thể có ảnh gần giống/nhãn mơ hồ; kiểm tra hash chỉ loại trùng pixel chính xác giữa các tập. Báo cáo hiện đã đồng bộ số liệu mới tại `docs/BT02_Report_XuanAnh_v2.docx` và `docs/BT02_Report.pdf`. Các bản báo cáo cũ giữ làm lịch sử; các con số cũ và mới không phải cùng một run. Phần VGG16 trong báo cáo hiện được gắn rõ là số liệu tham khảo do nhóm cung cấp.

SHA-256 checkpoint: 5697e8d1c69502a93e540762aa0b954458d5b5c5b5fdb6b7e74bffef97058297
