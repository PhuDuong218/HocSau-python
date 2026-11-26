# Đối chiếu hình và bàn giao BTL02

Nguồn đối chiếu: Report_Emotion Recognition.docx trong Downloads (25/11/2026), results/plots, nhật ký và output/BT02_Phan_Duong.docx. Bản Word nhóm chưa được chỉnh sửa.

- Hình 1.1 và 1.2 trong Word hiện khác file PNG mới trong results/plots theo SHA-256. Nếu dùng hình mới, thay cả hình và chú thích tương ứng.
- sample_faces.png gồm 35 ảnh: 7 lớp × 5 ảnh; báo cáo đang ghi 32 ảnh. Một ảnh nhãn Happy có chữ thay vì khuôn mặt rõ, không tự xóa khỏi dữ liệu.
- Bảng 1.1: Fear tổng 5.121 (4.097 + 1.024); Sad train 4.830, không phải 1.830. Sửa FER3012 thành FER2013.
- output/BT02_Phan_Duong.docx có W02 và W06, gồm hình minh họa augmentation và đường cong VGG16. Bản Word nhóm hiện để trống 2.1 và chưa chứa mục 4; cần Dương/nhóm ghép và duyệt.
- Đồ thị VGG16 là kết quả đã lưu 3 epoch; 44,30% không chứng minh nguyên nhân overfitting. VGG16 đã sửa có 15.246.023 tham số.
- Baseline trong Word là thử nghiệm cũ 48,34%, còn metadata checkpoint hiện tại thuộc lần train lại; cần nhóm thống nhất trước khi dùng các đoạn mô tả cùng nhau.
- 26/11/2026: Python 3.12.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu; DataLoader đọc 28.709 train và 7.178 test, batch (8,1,48,48). Lệnh predict_image.py chạy thành công trên một ảnh test có sẵn với checkpoint và metadata. Chưa đánh giá độ chính xác bằng ví dụ đơn.
- Conda chưa được nhận diện trên PATH. pip --dry-run -r requirements.txt giải quyết được các gói cho Python 3.12 trên máy này; chưa thực hiện cài đặt đầy đủ requirements CUDA hoặc kiểm thử GPU.
- Nhật ký cập nhật tại output/BT02_Nhat_ky_Duong_cap_nhat.xlsx là bản sao; số giờ, link bản Word khi chia sẻ và bước ghép báo cáo vẫn cần Dương/nhóm xác nhận.