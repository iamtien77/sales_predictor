Bản v3. Dò Tìm Siêu Tham Số Cho Decision Tree

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là tinh chỉnh độ sâu và ràng buộc lá của Decision Tree để tìm ra mô hình có năng lực phân nhánh sắc bén nhất.

Bước 1. Thiết lập không gian lưới tham số cho Decision Tree
Cấu hình param_grid:
- max_depth: [3, 5, 8]
- min_samples_leaf: [10, 30]
Tổng cộng 6 cấu hình độc lập.

Bước 2. Thực hiện quét lưới trên Validation Set
Chạy grid search trên 249.399 dòng Validation:
- max_depth = 3, min_samples_leaf = 10: Val RMSE = 62.41 BRL (Cây quá nông, underfitting).
- max_depth = 5, min_samples_leaf = 10: Val RMSE = 61.17 BRL (Bản mặc định).
- max_depth = 8, min_samples_leaf = 10: Val RMSE = 59.23 BRL (Tối ưu nhất).
- max_depth = 8, min_samples_leaf = 30: Val RMSE = 59.39 BRL (Ràng buộc lá quá chặt).

Bước 3. Xác định cấu hình chiến thắng (Winner Selection)
Cấu hình max_depth = 8, min_samples_leaf = 10 đạt Validation RMSE thấp nhất toàn bộ các lượt thử (59.23 BRL).
Độ sâu 8 cho phép cây tạo ra tới 256 vùng phân hoạch không gian đặc trưng, đủ độ chi tiết để tách biệt các tuần bán chạy khỏi 90% số 0.

Bước 4. Phân tích nguyên nhân Decision Tree đánh bại Random Forest
Do tín hiệu dự báo doanh thu tập trung cực mạnh vào một số ít đặc trưng chủ lực (last_week_sales, sales_mean_4_weeks, price).
Kỹ thuật subsampling đặc trưng ngẫu nhiên của Random Forest vô tình làm mất các đặc trưng này ở nhiều cây con, trong khi Decision Tree luôn chọn được biến tốt nhất ở nút gốc.

Bước 5. Lưu lại mô hình Tuned Decision Tree
Lưu cấu hình tối ưu làm ứng viên số 1 cho danh hiệu Selected Winner Model của hệ thống.
