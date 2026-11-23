# Bàn giao phần việc của Xuân Anh

## Bản báo cáo đã đồng bộ sau cùng

- File Word hiện dùng: `docs/BT02_Report_XuanAnh_v2.docx` (sửa trực tiếp, giữ nguyên tên).
- PDF xuất từ cùng file Word: `docs/BT02_Report.pdf`, 10 trang, đã xem đủ từng trang.
- Số liệu trong phần baseline được đồng bộ với lần train mới: Accuracy 48,41%, Macro-F1 0,3793, Weighted-F1 0,4613.
- Đã cập nhật xử lý ảnh trùng, train/validation/test, tiền xử lý, kiến trúc, đường cong học tập, bảng metric, confusion matrix, ảnh đúng/sai, filter conv3, kết luận và nguồn số liệu.
- Có 7 hình minh chứng thực: phân bố dữ liệu, ảnh mẫu, learning curve mới, confusion matrix mới, mẫu đúng, mẫu sai và filter cuối.
- Phần VGG16 được ghi rõ là số liệu tham khảo của nhóm chưa tái kiểm chứng; không trình bày như một thí nghiệm mới đã chạy. Nhóm cần người phụ trách xác nhận cấu hình/log nếu muốn kết luận so sánh mô hình cuối cùng.
- Phần đánh giá và biên tập báo cáo của Xuân Anh đã hoàn thiện theo bằng chứng hiện có. Toàn bài nhóm vẫn cần các thực nghiệm nhiều cấu hình, slide/demo và kiểm tra đóng gói theo phân công.
- Không thực hiện commit/push; người dùng tự làm theo yêu cầu.

### Nội dung nhật ký có thể bàn giao cho nhóm

| Ngày | Công việc đã thực hiện với Codex hỗ trợ | Minh chứng |
|---|---|---|
| 22/11/2026 | Kiểm tra FER2013, xử lý trùng ảnh và huấn luyện lại baseline có validation riêng | `results/data_audit/`, `results/training/baseline_seed42/`, checkpoint và metadata |
| 22/11/2026 | Đánh giá 7.178 ảnh, kiểm tra lại metric và xuất hình phân tích | `results/evaluation/baseline_retrained/` |
| 22/11/2026 | Biên tập và đồng bộ báo cáo, bổ sung 7 hình, xuất PDF và kiểm tra 10 trang | Word v2 và `docs/BT02_Report.pdf` |

Không tự điền số giờ, ngày họp hoặc phần việc của các thành viên khác. Các ghi chú bên dưới là lịch sử bàn giao trước khi bản báo cáo được hoàn thiện.

Đối chiếu phân công tại mục 3 của README: Evaluation Lead và Report Editor.

## Cập nhật sau khi tạo checkpoint mới

Đã tải và kiểm tra đầy đủ dữ liệu Kaggle, huấn luyện lại baseline của Xuân Anh trong 3 epoch. Checkpoint `checkpoints/baseline_best.pth` và `baseline_best.metadata.json` đã được tạo từ lần chạy thật, chọn theo validation loss. Đây không phải checkpoint gốc của Nhi.

Kiểm tra dữ liệu phát hiện 703 ảnh train trùng pixel với test; đã loại khỏi phần học. Loại thêm 73 ảnh train thuộc nhóm pixel có nhãn mâu thuẫn. Tập mới có 23.748 train, 4.185 validation và giữ 7.178 test. Chia tập theo nhóm pixel để không có trùng pixel train/validation/test.

Hướng dẫn và bằng chứng mới: `docs/TRAINING_BASELINE.md`, `results/data_audit/`, `results/training/baseline_seed42/` và `results/evaluation/baseline_retrained/`. Báo cáo Word/PDF đã tạo trước đó phân tích kết quả cũ của Nhi; chưa được thay bằng các số liệu của lần chạy mới.

Các mục phía dưới mô tả lần bàn giao trước khi tải dữ liệu và train lại; các ghi chú thiếu checkpoint/dữ liệu ở đó đã được giải quyết bằng lần chạy mới.

## Đã hoàn thành

1. `src/models.py`: đưa BaselineCNN từ notebook gốc thành module; giữ nguyên tên và kích thước tham số để nạp checkpoint. Đã đối chiếu đầu ra bằng cùng trọng số và tensor đầu vào với class gốc, khớp hoàn toàn.
2. `src/evaluate.py`: nạp checkpoint bằng weights_only, kiểm tra kiến trúc và mapping nhãn, đánh giá không augmentation, xuất Accuracy, loss, Macro Precision/Recall/F1, Weighted-F1 và chỉ số từng lớp.
3. Pipeline xuất confusion matrix dạng số đếm và chuẩn hóa theo hàng; CSV dự đoán từng ảnh; tối đa 32 ảnh đúng và 32 ảnh sai; hình 16 filter conv3 và tóm tắt lỗi. Đây là khả năng của code, chưa phải các file ảnh thật đã xuất từ checkpoint của nhóm.
4. `notebooks/03_experiments_and_evaluation.ipynb`: notebook hướng dẫn phân tích kết quả đã lưu và chạy đánh giá khi có đầu vào; không tự huấn luyện lại.
5. `src/analyze_saved_results.py`: tính lại metric từ 49 ô đã đọc trực tiếp trên ảnh confusion matrix có trong repo. Kiểm tra SHA-256 ảnh, tổng từng hàng và precision/recall/F1 làm tròn khớp output notebook gốc. Không khôi phục được danh tính ảnh từ ma trận tổng hợp.
6. `results/evaluation/saved_baseline/`: bảng metric có nguồn, classification report CSV, confusion matrix CSV và output notebook đã lưu. Accuracy 0,483421566; 3.470 đúng / 7.178; Macro-F1 khoảng 0,3604. Đây là phân tích kết quả đã lưu của Nhi, không phải lần test mới.
7. `docs/BT02_Report_XuanAnh.docx` và bản PDF cùng tên: biên tập bản báo cáo nhóm đã cung cấp, bổ sung Mục 3, chèn hai hình thực sự có, chỉnh mapping nhãn, augmentation và các kết luận nguyên nhân chưa đủ bằng chứng. Bản gốc ở Downloads được giữ nguyên.
8. `docs/EVALUATION_GUIDE.md`, `requirements-evaluation.txt`: hướng dẫn chạy và đầu vào cần thiết.

## Kiểm tra đã thực hiện

- Pipeline được chạy trọn vẹn trên fixture tổng hợp trong thư mục tạm để đối chiếu Accuracy, F1, cross-entropy, số dòng CSV và tính hợp lệ của các ảnh xuất. Fixture bị xóa sau kiểm tra, không dùng làm số liệu thực nghiệm.
- Thử checkpoint state_dict trực tiếp, checkpoint dictionary và prefix module.; trường hợp thiếu checkpoint dừng rõ ràng; danh sách ảnh đúng hoặc sai rỗng vẫn tạo hình hợp lệ.
- Tất cả cell code của notebook biên dịch thành công.
- Chạy lại phân tích ma trận đã lưu; 7 tổng hàng khớp support trong notebook, tất cả chỉ số từng lớp làm tròn hai chữ số khớp.
- Số tham số học được kiểm tra trực tiếp từ BaselineCNN: 1.274.823.
- Phần A của notebook được thực thi thành công khi không có checkpoint.
- Báo cáo Word được xuất sang PDF qua bộ máy Word-compatible có trên máy; đã render và xem đủ 6 trang, kiểm tra bảng, chữ tiếng Việt và hai hình không bị cắt/đè.

## Các điểm còn thiếu ở lần bàn giao trước

- Chạy inference trên checkpoint thật: thiếu `checkpoints/baseline_best.pth` và bộ ảnh FER2013 đã dùng.
- Tạo lại predictions.csv và hình True/Wrong Cases từ ảnh thật; xác minh nguyên nhân thị giác từng lỗi.
- Tạo filter conv3 từ checkpoint thật và đối chiếu phần trực quan hóa do Nhi phụ trách.
- Tái kiểm chứng số liệu VGG16: repo hiện không có notebook/log tương ứng. Báo cáo giữ số liệu đó với ghi rõ nguồn từ báo cáo nhóm, không khẳng định nguyên nhân thất bại đã được chứng minh.
- Chèn đủ bộ hình bàn giao: README ghi 7 hình nhưng liệt kê 8 tên; hiện chỉ có learning curve và confusion matrix. Còn thiếu dataset_distribution, sample_faces, final_conv_filters, true_predict_images, wrong_predict_images, vgg16_fail_curve.
- Đánh giá test độc lập: notebook baseline đã dùng cùng tập test để chọn checkpoint. Cần nhóm quyết định chia validation/test và train lại nếu cần kết quả test độc lập.

## Chạy ngay

Tại gốc repo:

```bat
python src/analyze_saved_results.py
```

Khi nhận đủ dữ liệu và checkpoint:

```bat
python src/evaluate.py
```

Chưa commit hoặc push các thay đổi này lên GitHub. Báo cáo hiện chưa đủ minh chứng để coi là bản nộp cuối.
