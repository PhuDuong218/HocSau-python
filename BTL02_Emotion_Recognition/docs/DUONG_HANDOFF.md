# Bàn giao phần dữ liệu và phân tích của Dương

Tài liệu này là bản thảo để nhóm ghép vào báo cáo và đối chiếu Work Log. Không ghi thay ngày, số giờ hoặc tác giả thực hiện của các thành viên khác.

## T02–T03: FER2013 và EDA

`results/data_audit/dataset_audit.json` ghi nhận 35.887 ảnh grayscale 48×48, gồm 28.709 train và 7.178 test (3.589 PublicTest, 3.589 PrivateTest). Bảy lớp theo `ImageFolder` là angry, disgust, fear, happy, neutral, sad, surprise. Happy có 8.989 ảnh, Disgust có 547 ảnh; đây là chênh lệch lớn cần lưu ý khi đánh giá theo từng lớp. Audit ghi CRC archive đạt và toàn bộ ảnh giải mã được. Nó cũng ghi 1.516 nhóm ảnh trùng pixel, trong đó 531 nhóm có mặt ở cả train và test, và 57 nhóm có nhãn mâu thuẫn. Định nghĩa trùng ở đây là hash SHA-256 của pixel giải mã giống hoàn toàn, không phải ảnh gần giống.

`src/eda.py` và `notebooks/01_eda.ipynb` tạo lại biểu đồ phân phối từ audit. Lưới ảnh mẫu chỉ tạo được khi có `data/raw/train/`; dữ liệu gốc không nằm trong Git. Kết quả audit và biểu đồ đã có không chứng minh ai là người chạy, nên cần gắn commit hoặc link bàn giao của Dương trong Work Log.

## T05–T06 và W02: Tiền xử lý, tăng cường dữ liệu, DataLoader

`src/dataset.py` có ba cấu hình train: `baseline` dùng horizontal flip xác suất 0,5; `extended` thêm rotation tối đa 10 độ và translation tối đa 10% kích thước ảnh; `none` không tăng cường. Mọi cấu hình chuyển ảnh thành một kênh grayscale, `ToTensor()` đưa giá trị về [0,1], sau đó `Normalize(0.5, 0.5)` đưa về khoảng [-1,1]. Test không có biến đổi ngẫu nhiên. `DataLoader` shuffle train và giữ thứ tự test; mặc định batch 128, `num_workers=0`. Hàm kiểm tra thứ tự lớp giữa train và test trước khi trả kết quả.

Checkpoint `baseline_best.pth` hiện tại thuộc lần huấn luyện lại ghi trong `docs/TRAINING_BASELINE.md`; lần đó dùng horizontal flip, không dùng rotation/translation. Vì vậy `extended` mới là cấu hình sẵn để thử nghiệm tiếp, chưa có metric chứng minh cải thiện. Báo cáo nên ghi rõ cấu hình nào đã chạy thật và tách kết quả theo Run ID. Không gọi DataLoader hiện tại là đã tối ưu prefetch khi chưa đo hoặc bật tính năng đó.

## W06: Phân tích thử nghiệm VGG16

Output đã lưu trong `notebooks/03_experiments_and_evaluation.ipynb` ghi VGG16 không pretrain, dùng ảnh 48×48 grayscale, Adam learning rate 0,0001, 3 epoch; độ chính xác trên tập test sau từng epoch là 26,83%, 42,02%, 44,30%. Kiến trúc có năm lần MaxPool, nên kích thước không gian giảm 48 → 24 → 12 → 6 → 3 → 1. Bộ phân loại của notebook đã thay bằng Linear(512,1024) và Linear(1024,7); tổng số tham số theo các tầng này là khoảng 15,25 triệu, không phải hơn 130 triệu của VGG16 chuẩn.

Kết quả 44,30% là output lưu trong notebook, chưa có checkpoint/log độc lập để chạy lại. Notebook dùng test trong mỗi epoch, nên số này không phải đánh giá test độc lập sau khi chọn mô hình. Ba epoch và train loss giảm chưa đủ để kết luận overfitting hoặc khẳng định riêng việc giảm feature map gây ra chênh lệch với baseline. Trước khi báo cáo kết luận cuối, cần chạy lại với validation riêng, cùng quy trình chia dữ liệu, seed, augmentation và ngân sách huấn luyện, rồi đánh giá test một lần.

## T19: README và bàn giao

Chạy các lệnh từ thư mục `BTL02_Emotion_Recognition`. Xem `README.md` để cài môi trường và `docs/TRAINING_BASELINE.md` cho quy trình huấn luyện lại. Dữ liệu gốc được Git bỏ qua; muốn tạo lưới ảnh mẫu hoặc chạy lại train cần đặt ảnh đúng cấu trúc trong `data/raw/`. Kiểm tra `git status` trước khi commit để tránh đưa dữ liệu hoặc checkpoint không dự định lên remote.

Phần còn cần xác nhận thủ công: chạy `src/eda.py` trên ảnh FER2013 thật; đối chiếu hình mới với báo cáo Word/PDF; điền ngày, số giờ, link commit và link bản thảo này vào Work Log/Report Tracking. Chưa có cơ sở để tự ghi các trường đó thay Dương.
