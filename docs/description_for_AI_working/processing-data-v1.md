Bản v1. Nạp và Kiểm Toán 6 Bảng Dữ Liệu Nguồn Olist

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là nạp dữ liệu thô an toàn, giải quyết các xung đột định dạng bảng mã và chuẩn hóa kiểu dữ liệu ban đầu.

Bước 1. Đọc 6 bảng CSV từ data/raw/
Sử dụng hàm load_data() trong src/data/loader.py để nạp 6 tệp dữ liệu Olist.
Áp dụng cơ chế giải mã an toàn: ưu tiên đọc bằng bảng mã utf-8, nếu gặp lỗi byte thì tự động chuyển sang fallback latin-1.
Ghi nhận không có file nào bị gián đoạn khi đọc.

Bước 2. Kiểm tra schema và tên cột
Kiểm tra sự tồn tại của các cột bắt buộc trong từng bảng.
Đổi tên các cột gây nhầm lẫn nếu có và đồng nhất định dạng chữ thường snake_case.
Xác nhận cột khóa order_id, product_id và customer_id có kiểu chuỗi ký tự hợp lệ.

Bước 3. Chuẩn hóa kiểu ngày tháng datetime64[ns]
Chuyển đổi trường order_purchase_timestamp của bảng orders sang kiểu datetime64[ns].
Chuyển đổi review_answer_timestamp của bảng reviews sang datetime64[ns] để phục vụ kiểm toán chống rò rỉ sau này.
Loại bỏ các bản ghi có thời gian bị lỗi NaT nếu phát sinh.

Bước 4. Hợp nhất sơ bộ các bảng giao dịch
Nối bảng order_items với orders theo order_id để có thông tin thời gian mua.
Nối tiếp với products để lấy thông tin ngành hàng và đặc tính vật lý.
Thu được bảng dữ liệu giao dịch hợp nhất ban đầu gồm 112.650 dòng.

Bước 5. Lưu trữ trạng thái sạch sơ bộ
Kiểm tra số lượng dòng và cột sau hợp nhất.
Xác nhận toàn bộ các dòng item đều có thông tin giá bán và thời gian mua hàng hợp lệ.
