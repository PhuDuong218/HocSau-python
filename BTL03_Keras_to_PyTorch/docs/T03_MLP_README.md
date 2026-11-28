# T03 MLP và tư liệu kỹ thuật của Nguyễn Xuân Anh

Bảng phân công cuối xác định báo cáo chính của Xuân Anh là **R03 LSTM/SegNet**. Báo cáo MLP R02 có sẵn là tư liệu gửi Phú Dương. Xem `docs/XUAN_ANH_NOP_GI.txt` để biết đúng phần bàn giao.

Phạm vi code: chuyển `Func_apxm_MLP.ipynb` và `MNIST_MLP.ipynb` từ Keras sang PyTorch; báo cáo kỹ thuật Module MLP. Không sửa Dataset/DataLoader chung của T02 hoặc code CNN/LSTM/SegNet.

## Chạy

Mở terminal tại gốc repo (thư mục chứa `src/`):

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train_mlp --download-only
python -m src.train_mlp --task function
python -m src.train_mlp --task mnist
python scripts/verify_mlp.py
```

Mặc định: CPU, seed 42, MNIST 30 epoch. Lần tải MNIST đầu cần Internet; sau đó chạy offline. Dùng `--device cuda` nếu đã có PyTorch CUDA phù hợp. `--epochs` chỉ điều khiển MNIST; các số epoch khảo sát hàm được giữ theo notebook gốc. `--task all` chạy cả hai bài.

`requirements-mlp-tested.txt` ghi môi trường thực tế đã chạy; các thư viện trong `requirements.txt` của nhóm vẫn được giữ nguyên. Kết quả có thể khác nhỏ giữa phiên bản và phần cứng.

## Kiến trúc và chuyển đổi

| Nội dung | Keras gốc | PyTorch |
| --- | --- | --- |
| Xấp xỉ hàm | Dense(2,B,sigmoid), Dense(B,1,sigmoid) | Linear(2,B), Sigmoid, Linear(B,1), Sigmoid |
| Hàm mục tiêu | Gaussian 2 chiều, mean=(0,0), covariance=I | exp(-(x1²+x2²)/2)/(2π) |
| MNIST | Flatten, 512 ReLU, Dropout .2, 512 ReLU, Dropout .2, 10 Softmax | Cùng lớp ẩn/dropout, đầu ra 10 logits |
| Loss MNIST | categorical_crossentropy + one-hot | CrossEntropyLoss + class index long |
| Optimizer | SGD, lr=.01, batch=128 | SGD, lr=.01, batch=128 |
| Khởi tạo Dense | Glorot uniform, bias zero | xavier_uniform_, bias zero |

`CrossEntropyLoss` nhận logits, vì vậy không gắn Softmax trước loss. Chỉ dùng `softmax` khi cần xác suất để trình diễn. Trọng số Dense của Keras có chiều `[in,out]`, còn `nn.Linear` dùng `[out,in]`; kiểm thử NumPy xác nhận phép chuyển vị cho kết quả forward tương đương.

## MNIST và pipeline T02

`make_mlp_mnist_loaders()` gọi `make_mnist_loaders()` chung để lấy đúng dataset Torchvision. Sau đó tạo tensor view cục bộ từ pixel gốc và chia 255 để giữ đầu vào `[0,1]` của notebook Keras. Cách chuẩn hóa mean=.1307/std=.3081 của loader chung được giữ nguyên trong T02 nhưng không áp dụng cho thí nghiệm đối chiếu này. Adapter không thay đổi dữ liệu, transform hay API chung.

Input `[B,1,28,28]` -> Flatten `[B,784]` -> logits `[B,10]`. Nhãn `[B]`, `torch.long`. Dùng đủ 60.000 mẫu train và 10.000 test. Giữ 30 epoch cố định và checkpoint cuối cùng, không chọn model tốt nhất theo test. Để bám notebook, test được ghi theo mỗi epoch; không dùng để điều chỉnh siêu tham số. Đây không phải validation độc lập; khi tối ưu thêm phải tách validation từ train.

## Ba khảo sát hàm Gaussian

- A: 100, 200, ..., 1000 mẫu; B=16, C=100.
- B: 8, 16, 32, 64, 128, 256, 512; A=800, C=100.
- C: 10, 20, ..., 190 epoch; A=800, B=128.

Tổng 36 cấu hình; mỗi cấu hình khởi tạo lại model. Lưới train 32×32, lưới test 10×10 trên [-3,3]². Giữ pool chỉ số 1..1000 như notebook gốc, dù lưới có 1024 điểm. Train-grid MSE tính trên toàn bộ 1024 điểm, bao gồm cả điểm không được chọn; nó không phải loss chỉ trên tập con fit.

Khác biệt có chủ đích: cố định seed và giữ cùng tập mẫu cho các khảo sát B/C để dễ đối chiếu. Không cam kết khớp bit với Keras vì RNG và thứ tự minibatch khác. Checkpoint minh họa chọn trước A=800, B=128, C=100, không chọn theo test. Loss giảm không bảo đảm học đúng hình dạng Gaussian; xem cả hình bề mặt và sai số so với baseline hằng.

## Các file của T03

- `src/mlp_models.py`: hai kiến trúc và khởi tạo.
- `src/mlp_data.py`: lưới Gaussian và adapter MNIST.
- `src/train_mlp.py`: train/evaluate, 36 khảo sát, log, biểu đồ, checkpoint.
- `tests/test_mlp.py`: kiểm thử forward, loss, gradient, reload và tích hợp adapter.
- `scripts/verify_mlp.py`: nạp lại checkpoint, đối chiếu dự đoán và metrics.
- `notebooks/01_mlp_function.ipynb`, `notebooks/01_mlp_mnist.ipynb`: notebook điều khiển.
- `results/T03_function`, `results/T03_mnist`: kết quả thật và cấu hình.
- `checkpoints/T03_function.pth`, `checkpoints/T03_mnist.pth`: trọng số, không commit mặc định.
- `reports/R03_LSTM_SegNet_XuanAnh.docx`: báo cáo để nhóm trưởng ghép.
- `references/keras_mlp`: notebook nguồn không sửa và SHA256.

Mỗi thí nghiệm ghi seed, thiết bị, phiên bản, thời gian, base commit và hash mã nguồn. Base commit là nền của worktree; hash ghi chính xác nội dung file thực nghiệm, kể cả khi chưa commit.

Để tạo lại báo cáo từ kết quả đã có: cài `python -m pip install -r requirements-mlp-report.txt`, sau đó chạy `python scripts/build_r03_report.py`. Chạy lại sẽ tạo lại ô MSSV trống, vì vậy điền thông tin cá nhân sau lần tạo báo cáo cuối.

## Giới hạn benchmark

MNIST Keras có loss 0.1142885, accuracy 96.48% trong output notebook được cung cấp. Đây là output tham khảo có sẵn, không phải Keras được chạy lại trong cùng môi trường. Không kết luận framework nào nhanh hơn/tốt hơn từ hai lần chạy khác thiết bị, RNG và phiên bản. Notebook hàm không có bảng số benchmark Keras đã lưu; không tự điền số.

## Bàn giao và review

Nhánh hiện tại: `XuanAnh`, tích hợp cây mã có riêng hai thư mục BTL02/BTL03 từ `Nhi`. Đối chiếu `docs/XUAN_ANH_NOP_GI.txt` để xem kết quả và trạng thái review. Không đánh dấu merge/review đã hoàn thành khi chưa có người khác duyệt. Dataset và checkpoint không đưa lên GitHub; gói bàn giao local có checkpoint để kiểm chứng.

## Notebook và EXP theo bảng nhóm mới nhất

Notebook chính: `notebooks/01_mlp_pytorch.ipynb`. EXP_01 chạy bằng `python scripts/run_exp01.py`: input random [-1,1], SGD .01, batch32, 50 epoch. EXP_02 dùng MNIST30 epoch như trên. Số liệu dán vào bảng nằm trong `docs/DAN_VAO_BANG_EXP01_EXP02.txt`. Các khảo sát grid gốc vẫn được giữ làm tư liệu bổ sung, không dùng nhầm để điền EXP_01 mới.
