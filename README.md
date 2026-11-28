# Học sâu Python

Repository nhóm được chia theo bài tập để mã nguồn, kết quả và báo cáo không bị lẫn.

```text
HocSau-python/
├── BTL02_Emotion_Recognition/
├── BTL03_Keras_to_PyTorch/
├── .gitignore
├── GIT_FILES_XUAN_ANH.txt
└── README.md
```

## BTL02

Mã và tài liệu hiện có được giữ trong [BTL02_Emotion_Recognition](BTL02_Emotion_Recognition/README.md). Phần tổ chức lần này không thay đổi nội dung BTL02.

## BTL03

Xem [hướng dẫn chung của nhóm](BTL03_Keras_to_PyTorch/README.md). Mỗi lệnh chạy BTL03 được thực hiện sau khi chuyển vào thư mục `BTL03_Keras_to_PyTorch`.

Phần của Nguyễn Xuân Anh:

- **T03:** [notebook MLP](BTL03_Keras_to_PyTorch/notebooks/01_mlp_pytorch.ipynb), model, adapter dữ liệu, train/evaluation và kiểm thử.
- **EXP_01 / EXP_02:** [số liệu điền nhật ký](BTL03_Keras_to_PyTorch/docs/DAN_VAO_BANG_EXP01_EXP02.txt), kết quả thực tế trong `results/EXP_01` và `results/T03_mnist`.
- **R03:** [bản thảo Mục 3](BTL03_Keras_to_PyTorch/reports/R03_LSTM_SegNet_XuanAnh.docx), viết theo các mục 3.1–3.4 của đề cương; sơ đồ PNG/SVG trong `reports/R03_figures`.
- **Hỗ trợ R05/T05:** [ghi chú kỹ thuật](BTL03_Keras_to_PyTorch/reports/HO_TRO_R05_T05_XuanAnh.txt).

Các phần CNN và SegNet đã có của thành viên khác được giữ nguyên. Phần LSTM trong R03 là phân tích thiết kế từ Keras theo đề cương 7 ảnh; không kèm số đo thực nghiệm LSTM chưa có nguồn xác nhận.

## Kiểm tra phần MLP

```powershell
cd BTL03_Keras_to_PyTorch
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_exp01.py
python -m src.train_mlp --task mnist
python scripts/verify_mlp.py
```

`verify_mlp.py` kiểm tra thêm checkpoint Gaussian của khảo sát nguồn. Chạy `python -m src.train_mlp --task function` trước nếu chưa có checkpoint đó. Dataset MNIST được tải ở lần chạy đầu. Checkpoint dùng để kiểm chứng được bàn giao riêng, không thêm vào Git.

## Quy tắc Git

Commit code, notebook, tài liệu, báo cáo và kết quả nhỏ có thể đối chiếu. Không thêm môi trường ảo, dataset, checkpoint mới, ZIP hoặc ảnh render tạm. Danh sách cụ thể và cách push nhánh được ghi trong [GIT_FILES_XUAN_ANH.txt](GIT_FILES_XUAN_ANH.txt).
