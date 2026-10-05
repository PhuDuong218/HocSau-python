# LAB03 - House Prices Advanced Regression Techniques

## Mục tiêu
- Đọc paper Dean De Cock về bộ dữ liệu Ames Housing.
- Xây dựng mô hình ML truyền thống bằng scikit-learn.
- Xây dựng MLP bằng PyTorch.
- Thực nghiệm với các nhóm đặc trưng để kiểm tra tác động của feature engineering.
- Sinh file dự đoán cho Kaggle.

## Dữ liệu
Đặt các file của cuộc thi Kaggle vào thư mục `data/`:

```text
data/
  train.csv
  test.csv
  sample_submission.csv   (không bắt buộc, code sẽ tự tạo nếu thiếu)
```

Kaggle: House Prices - Advanced Regression Techniques.

## Cài thư viện

```bash
pip install -r requirements.txt
```

## Chạy toàn bộ

```bash
python src/run_all.py
```

Các kết quả được lưu trong `results/` và mô hình trong `models/`.

## Các thực nghiệm
1. BASELINE: giữ dữ liệu gốc, xử lý thiếu và one-hot encoding.
2. ENGINEERED: thêm đặc trưng tổng diện tích, phòng tắm, hiên, tuổi nhà, tuổi garage và các tương tác đơn giản.
3. ENGINEERED + LOG FEATURES: log1p các biến số lệch phải để hỗ trợ MLP.
4. SELECTED: dùng nhóm đặc trưng có ý nghĩa thực tế cùng các đặc trưng tổng hợp.

Metric dùng trong validation: RMSLE, cùng cách tính theo log mà Kaggle sử dụng.

## Kiểm tra
- `src/smoke_test.py`: kiểm tra pipeline với dữ liệu giả lập nhỏ.
- Khi đã có `train.csv` và `test.csv`, chạy `src/run_all.py` để sinh toàn bộ kết quả.
- File `results/manual_check.txt` chứa phép kiểm tra dự đoán cho quan sát đầu tiên.

## Lưu ý
Trong gói này không nhúng dữ liệu Kaggle vì file CSV không nằm trong các file đính kèm của bài. Source code được viết để chạy trực tiếp sau khi đặt `train.csv` và `test.csv` vào `data/`.
