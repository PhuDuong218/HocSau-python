# ỨNG DỤNG DEMO NHẬN DIỆN CẢM XÚC KHUÔN MẶT (FER-2013)

Ứng dụng web tương tác thời gian thực được xây dựng bằng **Streamlit**, phục vụ cho việc trình diễn bài báo cáo và kiểm thử trực quan mô hình Học sâu nhận diện 7 biểu cảm khuôn mặt.

---

## 1. Tính năng chính

- 📷 **Chụp ảnh trực tiếp từ Webcam:** Cho phép người dùng chụp ảnh khuôn mặt qua webcam laptop và nhận kết quả tức thì.
- 📁 **Tải ảnh chân dung từ máy tính:** Hỗ trợ mọi định dạng ảnh phổ biến (`JPG`, `PNG`, `WEBP`, `BMP`).
- 🎯 **Phát hiện khuôn mặt hiện đại (YuNet):** Sử dụng mô hình Deep Learning ONNX của OpenCV phát hiện chính xác khuôn mặt nhiều góc nghiêng.
- 📊 **Phân tích xác suất 7 biểu cảm:**
  - Nhãn cảm xúc dự đoán cao nhất kèm Emoji sinh động và độ tin cậy (% Confidence).
  - Biểu đồ thanh ngang trực quan hóa phân bố xác suất của cả 7 trạng thái cảm xúc.
- ⚡ **Thiết kế Modular (Sẵn sàng mở rộng):** Tự động nhận diện thêm các mô hình thử nghiệm khác nếu có bổ sung checkpoint vào thư mục `checkpoints/`.

---

## 2. Hướng dẫn khởi chạy ứng dụng

Mở terminal tại thư mục gốc của dự án (`DoAnHocSau/`) và thực thi lệnh:

```bash
# Kích hoạt môi trường ảo (nếu chưa kích hoạt và streamlit)
.\.venv\Scripts\python.exe -m streamlit run demo/app.py
```

Sau khi chạy lệnh trên, trình duyệt web sẽ tự động mở địa chỉ:
👉 **`http://localhost:8501`**

---

## 3. Cấu trúc thư mục

```text
demo/
├── app.py                     # Mã nguồn chính của ứng dụng Streamlit
├── face_detection_yunet.onnx  # Trọng số mô hình phát hiện khuôn mặt YuNet (227 KB)
└── README_DEMOAPP.md                  # Hướng dẫn sử dụng và kiểm thử
```

---

## 4. Gợi ý chuẩn bị Video Demo dự phòng (Phần việc của Quốc Việt)

1. Mở ứng dụng theo lệnh ở mục 2.
2. Dùng công cụ quay màn hình (như Windows Game Bar `Win + G` hoặc OBS):
   - Thao tác thử 1: Chụp ảnh webcam với các biểu cảm khác nhau (Cười tươi `Happy`, ngạc nhiên `Surprise`, nghiêm túc `Neutral`).
   - Thao tác thử 2: Tải lên một vài bức ảnh mẫu từ máy tính để xem khung nhận diện và biểu đồ cột.
3. Xuất file video định dạng `.mp4` (độ dài 1–2 phút) lưu vào máy để sẵn sàng phát khi thuyết trình phòng trường hợp kết nối webcam bị lỗi.
