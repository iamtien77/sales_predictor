Bản v4. Kỹ Nghệ Biến Sản Lượng & Vòng Đời Sản Phẩm (Units & Lifecycle)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là định lượng khối lượng hàng hóa xuất kho và độ trưởng thành của sản phẩm trên sàn thương mại điện tử.

Bước 1. Kỹ nghệ biến số lượng sản phẩm bán tuần trước (last_week_units)
Đếm tổng số đơn vị hàng hóa (order_item_id) được giao dịch thành công trong tuần t-1.
Phân biệt giữa sản phẩm giá rẻ bán số lượng nhiều với sản phẩm cao cấp bán số lượng ít nhưng giá trị lớn.

Bước 2. Kỹ nghệ biến vòng đời sản phẩm (weeks_since_first_sale)
Đo lường số tuần lịch tính từ thời điểm sản phẩm phát sinh đơn hàng đầu tiên trong lịch sử hệ thống cho đến tuần target:
weeks_since_first_sale = (target_week - first_sale_week) / 7 days.
Đặc trưng này phản ánh mức độ quen thuộc của khách hàng với mã sản phẩm và giai đoạn trong vòng đời sản phẩm (Ra mắt, Tăng trưởng, Bão hòa, Suy thoái).

Bước 3. Kỹ nghệ biến giá bán tuần gần nhất (price)
Lấy giá bán trung bình của sản phẩm trong tuần có phát sinh giao dịch gần nhất trước target week.
Áp dụng forward-fill theo chiều thời gian để đảm bảo mọi tuần đều có thông tin giá niêm yết hiện hành.
Không sử dụng giá bán phát sinh trong tuần target để tạo đặc trưng đầu vào.

Bước 4. Kiểm tra phân phối và tương quan
Khảo sát mối quan hệ giữa weeks_since_first_sale và xác suất bán được hàng: các sản phẩm có tuổi đời từ 10 đến 40 tuần có xác suất phát sinh đơn hàng ổn định hơn các sản phẩm quá mới hoặc quá cũ.

Bước 5. Đưa các đặc trưng vào danh mục số học chính thức
Chuẩn bị các biến last_week_units, weeks_since_first_sale và price vào danh sách số học cần chuẩn hóa Z-score.
