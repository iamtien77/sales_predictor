Bản v3. Phân Tích Thống Kê Biến Mục Tiêu Doanh Thu & Phép Biến Đổi Log

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là khảo sát các đặc tính phân phối của biến mục tiêu doanh thu tuần, phân tích độ lệch phải cực đoan và kiểm chứng tính hợp lệ của phép biến đổi log.

Bước 1. Tính toán tổng doanh thu tuần cho từng sản phẩm
Xác định công thức biến mục tiêu: doanh thu tuần là tổng giá trị price của các đơn hàng thành công trong tuần (không bao gồm cước phí vận chuyển).
Đơn vị tính thống nhất toàn hệ thống là đồng Real Brazil (BRL).

Bước 2. Đo lường các chỉ số thống kê mô tả
Tính toán các tham số thống kê cơ bản: giá trị nhỏ nhất (0.0 BRL), trung vị (0.0 BRL), trung bình (~11.9 BRL) và giá trị lớn nhất (>10.000 BRL).
Tính các phân vị quan trọng: phân vị 75% = 0.0 BRL, phân vị 90% = 0.0 BRL, phân vị 95% = 35.0 BRL, phân vị 99% = 180.0 BRL.
Nhận định: phân phối doanh thu có độ lệch phải (right-skewed) cực đoan và có đuôi rất dài.

Bước 3. Phân tích hiện tượng sai số trên miền tiền tệ thực tế
Đánh giá rủi ro khi tối ưu hóa trực tiếp hàm mất mát bình phương (MSE) trên dữ liệu lệch phải: các đơn hàng giá trị cực lớn sẽ tạo ra gradient khổng lồ, làm mất ổn định quá trình học.
Khẳng định sự cần thiết phải biến đổi dữ liệu về không gian ổn định hơn.

Bước 4. Đánh giá phép biến đổi log tự nhiên log1p(sales)
Áp dụng hàm biến đổi: y = log(1 + sales). Phép biến đổi này giúp nén đuôi dài của phân phối, kéo các giá trị ngoại lai về gần giá trị trung tâm mà vẫn bảo toàn giá trị 0 (log(1 + 0) = 0).
Kiểm tra đồ thị phân phối của nhãn trong không gian log: phân phối trở nên gọn gàng và ổn định hơn rất nhiều.

Bước 5. Kiểm tra cơ chế nghịch đảo an toàn expm1
Kiểm chứng tính khả nghịch: khi suy luận thực tế và nghiệm thu, giá trị dự báo log được chuyển đổi ngược bằng hàm: sales_hat = max(0.0, exp(y_hat) - 1).
Xác nhận rằng phép biến đổi này bảo đảm tính toàn vẹn và không tạo ra doanh thu âm.
