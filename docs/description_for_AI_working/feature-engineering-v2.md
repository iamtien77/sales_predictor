Bản v2. Kỹ Nghệ Nhóm Biến Quán Tính Doanh Thu Trễ (Lag Features)
Mục tiêu của bản này là khai thác tín hiệu tự tương quan chuỗi thời gian thông qua các bước trễ lịch sử doanh số.

Bước 1. Thiết lập biến doanh thu tuần liền trước (last_week_sales)
Định nghĩa: last_week_sales là tổng doanh thu thực tế của sản phẩm ở tuần hoàn tất ngay trước tuần target (tuần t-1).
Đây là đặc trưng quan trọng nhất trong bài toán, có hệ số tương quan Pearson cao nhất với biến mục tiêu (~0.35).
Nếu tuần t-1 không có giao dịch, giá trị được gán bằng 0.0 BRL.

Bước 2. Thiết lập biến doanh thu trễ 2 tuần (sales_lag_2_weeks)
Đo lường doanh thu của sản phẩm ở tuần t-2.
Đặc trưng này giúp mô hình nắm bắt được các chu kỳ dao động ngắn hạn và hiện tượng ngắt quãng 2 tuần một lần.

Bước 3. Thiết lập biến doanh thu trễ 4 tuần (sales_lag_4_weeks)
Đo lường doanh thu của sản phẩm ở tuần t-4 (tương đương 1 tháng trước đó).
Đặc trưng này đại diện cho quán tính nhu cầu hàng tháng của người tiêu dùng đối với từng nhóm sản phẩm.

Bước 4. Đảm bảo nguyên tắc không rò rỉ thời gian (Temporal Integrity)
Toàn bộ các giá trị trễ lag 1, 2, 4 được tính toán trên các tuần lịch đã kết thúc hoàn toàn trước ngày bắt đầu của tuần target.
Không sử dụng bất kỳ thông tin nào phát sinh trong tuần target.

Bước 5. Đánh giá phân phối của các biến trễ
Kiểm tra ma trận tương quan giữa các biến trễ: lag càng xa thì mức độ tương quan với doanh thu hiện tại càng giảm dần, phản ánh quy luật suy giảm tự nhiên của chuỗi thời gian.
