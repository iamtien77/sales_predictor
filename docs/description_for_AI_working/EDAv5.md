Bản v5. Phân Tích Tính Chu Kỳ Mùa Vụ & Xu Hướng Bán Lẻ

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là khảo sát tác động của yếu tố thời gian, các tháng trong năm và mùa mua sắm cao điểm lên doanh số sản phẩm e-commerce Olist.

Bước 1. Phân tích tổng doanh số theo các tháng trong năm
Tổng hợp doanh thu toàn sàn theo 12 tháng lịch sử.
Ghi nhận sự gia tăng doanh thu đều đặn từ quý 1 đến quý 4 hàng năm do sự mở rộng quy mô của sàn Olist giai đoạn 2017-2018.
Nhận diện các đợt giảm nhẹ sau kỳ nghỉ lễ đầu năm.

Bước 2. Khảo sát mùa mua sắm cao điểm Black Friday
Phân tích dữ liệu tuần cuối tháng 11 năm 2017 (tuần Black Friday): doanh thu bùng nổ gấp nhiều lần so với các tuần bình thường.
Nhận diện sự tập trung đột biến vào một số ngành hàng điện tử, máy tính và đồ gia dụng.
Khẳng định sự cần thiết phải có biến nhị phân is_peak_shopping_season để báo hiệu cho mô hình.

Bước 3. Khảo sát các tuần lễ hội quốc gia Brazil
Kiểm tra các tuần diễn ra lễ hội Carnival, Quốc khánh Brazil và Giáng sinh.
Ghi nhận xu hướng tăng mua sắm trực tuyến trước kỳ nghỉ lễ và giảm giao dịch trong tuần nghỉ lễ chính thức.
Tạo biến chỉ báo is_holiday_week để bắt các biến động này.

Bước 4. Khảo sát tính chu kỳ lượng giác của tháng
Phân tích sự gián đoạn số học giữa tháng 12 (tháng 12) và tháng 1 (tháng 1): khoảng cách số học là 11 đơn vị nhưng khoảng cách thời gian thực tế chỉ là 1 tháng.
Đề xuất sử dụng biến đổi lượng giác sin/cos: month_sin = sin(2*pi*month/12) và month_cos = cos(2*pi*month/12) để biểu diễn không gian chu kỳ vòng tròn liên tục.

Bước 5. Tổng hợp các đặc trưng thời gian vào pipeline
Đưa bộ 6 biến thời gian (is_peak_shopping_season, is_holiday_week, month_sin, month_cos, week_of_year, quarter) vào danh mục kỹ nghệ đặc trưng chính thức.
