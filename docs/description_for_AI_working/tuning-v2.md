Bản v2. Dò Tìm Siêu Tham Số Cho Random Forest

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là quét không gian tham số của Random Forest để tìm cấu hình cân bằng giữa độ sâu cây và số lượng cây con.

Bước 1. Thiết lập không gian lưới tham số
Cấu hình param_grid cho Random Forest:
- n_estimators: [10, 20] cây.
- max_depth: [5, 8].
- min_samples_split: [10, 20].
Tổng cộng 8 cấu hình độc lập.

Bước 2. Thực hiện quét lưới trên Validation
Chạy grid search trên 249.399 dòng Validation.
Ghi nhận kết quả: khi nâng max_depth từ 5 lên 8, Validation RMSE giảm nhẹ từ 61.8 BRL xuống khoảng 60.5 BRL.
Số lượng cây n_estimators = 20 cho độ ổn định cao hơn 10 cây.

Bước 3. Phân tích hiện tượng làm mịn quá mức (Over-smoothing)
Nhận thấy rằng việc lấy trung bình cộng của 20 cây con vô tình làm phẳng các đỉnh doanh thu thực tế trên tập dữ liệu có 90% số 0.
Mặc dù Random Forest có MAE rất thấp nhưng khả năng bắt các đơn hàng lớn bị suy giảm nhẹ so với cây quyết định đơn lẻ có độ sâu vừa phải.

Bước 4. Lưu lại cấu hình Random Forest tốt nhất
Best params: n_estimators = 20, max_depth = 8, min_samples_split = 10 với Validation RMSE BRL ≈ 60.52.

Bước 5. Đóng gói kết quả vào cv_results
Ghi nhận toàn bộ thông số của 8 lượt thử vào danh sách lịch sử phục vụ xuất file tuning_history.csv.
