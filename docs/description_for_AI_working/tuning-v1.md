Bản v1. Chiến Lược Tinh Chỉnh Siêu Tham Số Trên Validation 10%

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là thiết lập chiến lược kiểm định Holdout Validation theo thời gian độc lập, tránh rò rỉ dữ liệu của k-fold CV truyền thống.

Bước 1. Xác định hạn chế của k-fold Cross-Validation trong chuỗi thời gian
Trong phân tích chuỗi thời gian bán lẻ, việc chia k-fold ngẫu nhiên sẽ lấy dữ liệu của tuần tương lai đem huấn luyện rồi dự báo cho tuần quá khứ, vi phạm nghiêm trọng luật nhân quả và gây lạc quan ảo về sai số.
Bắt buộc phải áp dụng chiến lược Holdout Validation độc lập theo dòng thời gian tuần lịch.

Bước 2. Cấu hình tập Validation (249.399 dòng)
Sử dụng 10% số tuần lịch sử nằm giữa tập Train và tập Test làm tập kiểm định duy nhất.
Tập Validation hoàn toàn không được tham gia vào quá trình fit trọng số của các mô hình.

Bước 3. Xác định thang đo tối ưu hóa (Scoring Metric)
Quy định rõ ràng: tiêu chí lựa chọn mô hình chiến thắng là **Validation RMSE trên thang đo tiền tệ thực tế BRL** (sau khi nghịch đảo expm1), không dùng RMSE trên không gian log để chọn mô hình.

Bước 4. Cài đặt hàm grid_search_from_scratch()
Xây dựng hàm dò tìm lưới siêu tham số trong src/models/tuning.py:
Duyệt qua toàn bộ tích Descartes của các giá trị tham số trong param_grid.
Với mỗi tổ hợp, fit mô hình trên Train, predict trên Val, tính Validation RMSE BRL và ghi nhận lịch sử.

Bước 5. Hỗ trợ chế độ tăng tốc --fast-mode
Nếu bật cờ --fast-mode, tự động lấy mẫu ngẫu nhiên tái lập 20.000 dòng Train để tinh chỉnh nhanh trong < 30 giây phục vụ kiểm thử tích hợp liên tục (CI/CD).
