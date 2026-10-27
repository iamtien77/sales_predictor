Bản v2. Làm Sạch Nguồn Item Trước Aggregation

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là làm sạch chi tiết các dòng giao dịch item, loại trừ các trạng thái đơn hàng hủy bỏ mà không áp dụng cắt xén ngoại lai làm sai lệch doanh thu.

Bước 1. Lọc bỏ các trạng thái đơn hàng không hợp lệ
Triển khai hàm clean_weekly_source_data() trong src/data/builder.py.
Loại bỏ các đơn hàng có trạng thái canceled (hủy đơn) hoặc unavailable (hết hàng/không khả dụng) vì chúng không phát sinh doanh thu thực tế.
Chỉ giữ lại các đơn hàng có trạng thái delivered, shipped, invoiced, approved, processing.

Bước 2. Kiểm tra điều kiện giá bán hợp lệ
Xác thực điều kiện: giá bán price > 0.0 BRL.
Loại bỏ các dòng có giá bán âm hoặc bằng 0 do lỗi hệ thống kế toán.
Ghi nhận toàn bộ giá bán của sản phẩm Olist đều dương và hợp lệ.

Bước 3. Kiểm tra thông tin định danh sản phẩm
Loại bỏ các bản ghi thiếu mã product_id (giá trị null/NaN).
Xác nhận các sản phẩm đều có mã định danh duy nhất dài 32 ký tự hex.

Bước 4. Nguyên tắc không cắt xén nhãn mục tiêu (No Target Capping)
Tuân thủ nguyên tắc phương pháp luận: không áp dụng kỹ thuật kẹp biên (capping) hay loại bỏ các giao dịch có giá trị cực cao.
Các đơn hàng có giá trị hàng nghìn BRL là giao dịch thực tế trên sàn thương mại điện tử, việc cắt bỏ sẽ làm méo mó bản chất phân phối doanh thu.

Bước 5. Khóa số lượng dòng sản phẩm hợp lệ
Xác nhận số lượng dòng item sau làm sạch đạt đúng 112.650 bản ghi.
Bàn giao tập dữ liệu sạch cho bước tổng hợp bảng chuỗi thời gian Sản phẩm – Tuần.
