Bản v1. Khám phá Schema và Dữ liệu Giao dịch Thô Olist

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là khảo sát toàn diện 6 bảng dữ liệu thương mại điện tử Olist gốc, xác định các trường khóa, kiểm tra kiểu dữ liệu và đánh giá độ phân mảnh trước khi tiến hành chuẩn hóa.

Bước 1. Nạp và kiểm tra 6 bảng dữ liệu nguồn Olist
Đọc các bảng dữ liệu từ data/raw/ gồm đơn hàng (orders), chi tiết đơn hàng (order_items), sản phẩm (products), khách hàng (customers), thanh toán (order_payments) và đánh giá (order_reviews).
Kiểm tra số dòng, số cột và danh sách tên trường của từng bảng.
In 5 dòng đầu tiên của từng bảng để xác nhận việc nạp dữ liệu diễn ra chính xác.
Ghi nhận bảng order_items chứa 112.650 bản ghi dòng sản phẩm giao dịch thành công.

Bước 2. Kiểm tra schema và các khóa liên kết (Foreign Keys)
Xác định khóa chính và khóa ngoại để nối các bảng: order_id liên kết giữa orders, items, payments và reviews; product_id liên kết giữa items và products; customer_id liên kết giữa orders và customers.
Kiểm tra tính toàn vẹn của các khóa nối: xác nhận không có mã product_id hay order_id bị hỏng hoặc null.

Bước 3. Xem trước một số dòng giao dịch mẫu
Trích xuất một số bản ghi giao dịch thực tế đại diện cho nhiều ngành hàng khác nhau.
Quan sát mối liên hệ giữa giá bán niêm yết (price), phí vận chuyển (freight_value) và tổng thanh toán.
Nhận diện các ngành hàng phổ biến như đồ gia dụng, linh kiện máy tính, mỹ phẩm sức khỏe.

Bước 4. Xác định các trường định lượng và phân loại
Phân loại các trường dữ liệu: trường định lượng gồm giá bán, cước phí, kích thước, khối lượng, số ảnh; trường phân loại gồm tên ngành hàng, bang khách hàng, phương thức thanh toán.
Đặc biệt lưu ý trường thời gian order_purchase_timestamp sẽ đóng vai trò trục thời gian cốt lõi của bài toán.

Bước 5. Đánh giá sơ bộ khối lượng và sự phân mảnh của dữ liệu
Ghi nhận dữ liệu giao dịch có độ phân tán cao với hàng chục nghìn mã sản phẩm khác nhau.
Đa số sản phẩm chỉ phát sinh vài giao dịch trong suốt thời gian quan sát.
Xác định rõ ràng: nếu mô hình hóa ở mức dòng giao dịch tĩnh thì sẽ không dự báo được nhu cầu tương lai; do đó bắt buộc phải chuyển đổi sang chuỗi thời gian Sản phẩm – Tuần.
