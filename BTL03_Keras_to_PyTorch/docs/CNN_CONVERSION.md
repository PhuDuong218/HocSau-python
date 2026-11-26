# Chuyển đổi CNN từ Keras sang PyTorch

T04 dùng `1_MNIST_CNN.ipynb` và `4. Training Utilities/2_Human_Gender_CNN.ipynb` trong `examples.zip`. T05 dùng `4. Training Utilities/3_Emotion_Reg_CNN.ipynb`. Slide `02_KerasExamples.pdf` mô tả kiến trúc CNN, callback và tăng cường dữ liệu. Slide giới thiệu và README của giảng viên cung cấp bối cảnh môi trường Keras của các ví dụ.

| Mô hình | Thân mạng | Đầu phân loại |
| --- | --- | --- |
| MNIST | Conv 64, 128, 256; mỗi tầng theo sau bởi max pooling và dropout 0,2 | Dense 512 → 512 → 10 |
| Gender | 5 khối Conv 16, 32, 64, 128, 256; lần lượt 2, 3, 3, 3, 3 lớp | Dense 1024 → 1024 → 2 |
| FER2013 | Hai khối Conv 64–64 và 128–128–218–218, xen kẽ padding same/valid | Dense 1024 → 7 |

PyTorch dùng tensor NCHW và nhãn chỉ số nguyên. Mô hình xuất logits cho `CrossEntropyLoss`. Trainer lưu checkpoint có validation loss thấp nhất và nạp lại trước khi đo test. EarlyStopping dừng sau 5 epoch không tốt hơn theo mặc định. ReduceLROnPlateau theo dõi validation loss, nhân learning rate với 0,1 sau 5 epoch chững, với ngưỡng dưới 0,0001. Tăng `--patience` nếu muốn để mô hình chạy thêm sau khi giảm learning rate.

MNIST dùng ảnh trong [0,1] như notebook gốc. Lượt chạy nhanh dùng Adam với learning rate 0,001, còn SGD 0,01 là cấu hình mặc định theo notebook Keras. Gender xoay NPY 90° theo chiều kim đồng hồ, giữ cường độ gốc [0,255], khởi tạo Glorot uniform với bias bằng 0 và dùng tập validation được cung cấp. FER2013 mô phỏng phép xoay ±45°, tịnh tiến tối đa 20% và lật ngang của ImageDataGenerator qua torchvision. Padding theo biên và nội suy song tuyến tính là cách triển khai tương ứng; thứ tự phép biến đổi và quy tắc làm tròn có thể sai khác nhỏ. Validation và test dùng phép biến đổi xác định.

FER2013 hỗ trợ cả CSV one-hot gốc và dữ liệu ảnh. Với CSV, thứ tự nhãn theo notebook (Sad, Surprise, Neutral ở cuối). Với thư mục ảnh, thứ tự theo ImageFolder (Neutral, Sad, Surprise ở cuối). Xem `classes` trong JSON của mỗi lượt chạy. Lượt chạy trên ảnh BTL02 dùng manifest đã lọc trùng pixel và tập test gộp Public/Private, vì vậy không dùng số đo đó để suy ra chênh lệch trực tiếp với kết quả Keras trên CSV PublicTest.

Notebook in số tham số, kích thước tập, kết quả và vẽ đường cong từ CSV. Mốc 98,8% của MNIST và 87,7% của Gender là mốc tham chiếu trong bảng công việc. Chỉ dùng JSON và CSV của các lượt chạy thật khi báo cáo kết quả PyTorch.

## Kết quả trên CPU

Các lượt PyTorch chạy với seed 42 và batch 128. Số tham số được đếm trực tiếp trong PyTorch, khớp với model summary của cả ba notebook Keras. Bảng ghi accuracy của checkpoint có validation loss thấp nhất.

| Mô hình | Tham số | Epoch PyTorch | Validation accuracy | Test accuracy | Thời gian train và validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | 1.817.610 | 4 | 98,92% | 99,24% | 709,39 giây |
| Gender | 4.063.570 | 30 | 87,60% | — | 3.183,22 giây |
| FER2013 | 19.028.287 | 1 | 25,59% | 24,81% | 1.302,71 giây |

- [Log MNIST](../results/mnist_cpu_seed42/history.csv) và [metrics MNIST](../results/mnist_cpu_seed42/metrics.json).
- [Log Gender](../results/gender_full_seed42/history.csv), [metrics Gender](../results/gender_full_seed42/metrics.json) và [biểu đồ Gender](../results/gender_full_seed42/learning_curve.png). [Lượt Gender 6 epoch](../results/gender_keras_input_seed42/metrics.json) là phép kiểm tra ban đầu.
- [Log FER2013](../results/fer2013_cpu_seed42/history.csv) và [metrics FER2013](../results/fer2013_cpu_seed42/metrics.json).

MNIST đạt mốc tham chiếu 98,8%. Gender lần đầu vượt mốc tham chiếu 87,7% ở epoch 21 (87,72%) và cao nhất ở epoch 23 (88,04%). Checkpoint lưu từ epoch 25 có validation loss thấp nhất (0,3258) và accuracy 87,60% trên 5.968 ảnh. Bộ Gender của giảng viên cung cấp tập train và validation. FER2013 đã chạy hết quy trình huấn luyện, lưu checkpoint, nạp lại và đánh giá. Sau một epoch, test accuracy xấp xỉ tỷ lệ lớp Happy trong tập test (24,71%). Lượt này xác minh pipeline chạy được. Để đánh giá chất lượng mô hình, cần huấn luyện thêm và so sánh trên cùng bộ chia dữ liệu.

Log Keras trong ZIP ghi validation accuracy cao nhất là 98,85% (MNIST), 88,17% (Gender có callback) và 61,83% (FER2013 có augmentation). Các giá trị này lấy từ log có sẵn của giảng viên. MNIST và FER2013 sử dụng dữ liệu validation khác giữa hai lần chạy, và ngân sách epoch cũng khác nhau. Đối chiếu thời gian giữa CPU và GTX 1080 cần giữ thông tin thiết bị đi kèm.
