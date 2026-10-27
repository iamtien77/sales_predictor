Bản v4. Phân Chia Chuỗi Thời Gian Theo Tuần Lịch (Temporal Split 80/10/10)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là phân chia tập dữ liệu nghiêm ngặt theo dòng thời gian tuần lịch, loại bỏ hoàn toàn nguy cơ rò rỉ dữ liệu tương lai vào quá khứ.

Bước 1. Thu thập danh sách các tuần lịch duy nhất
Trích xuất toàn bộ các mốc tuần target duy nhất trong bảng weekly panel và sắp xếp theo thứ tự tăng dần thời gian.
Xác định tổng số tuần quan sát trong toàn bộ lịch sử Olist.

Bước 2. Tính toán điểm cắt phân chia theo tỷ lệ 80% / 10% / 10%
Chia danh sách tuần theo tỷ lệ số tuần target:
- 80% số tuần đầu tiên dành cho tập Huấn luyện (Train Set).
- 10% số tuần tiếp theo dành cho tập Tinh chỉnh / Kiểm định (Validation Set).
- 10% số tuần cuối cùng dành cho tập Kiểm tra độc lập (Test Set).

Bước 3. Phân bổ các bản ghi sản phẩm vào từng tập dữ liệu
Lọc toàn bộ các bản ghi có tuần target thuộc phân vùng tương ứng:
- Tập Train (80%): 655.066 dòng.
- Tập Validation (10%): 249.399 dòng.
- Tập Test (10%): 293.717 dòng.
Tổng cộng: 655.066 + 249.399 + 293.717 = 1.198.182 dòng (khớp 100%).

Bước 4. Xác thực tính độc lập thời gian (Anti-Leakage Validation)
Kiểm tra mốc thời gian: max(Train week) < min(Val week) và max(Val week) < min(Test week).
Xác nhận toàn bộ sản phẩm trong cùng một tuần lịch đều nằm trọn vẹn ở một phân vùng dữ liệu duy nhất.
Loại bỏ hoàn toàn khả năng rò rỉ dữ liệu từ tập kiểm tra vào tập huấn luyện.

Bước 5. Xuất biểu đồ kiểm chứng phân chia chuỗi thời gian
Gọi hàm plot_temporal_split_distribution() trong src/evaluation/visualization.py.
Lưu biểu đồ phân bố mẫu ra file results/figures/eda_temporal_split_distribution.png.
