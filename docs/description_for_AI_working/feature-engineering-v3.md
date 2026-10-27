Bản v3. Kỹ Nghệ Nhóm Biến Thống Kê Trượt (Rolling Window Features)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là làm mịn các biến động ngẫu nhiên và đo lường độ phân tán nhu cầu thị trường trong cửa sổ 4 tuần gần nhất.

Bước 1. Tính toán doanh thu trung bình trượt 4 tuần (sales_mean_4_weeks)
Định nghĩa: trung bình cộng doanh thu của sản phẩm trong 4 tuần lịch sử gần nhất trước tuần target:
sales_mean_4_weeks = (sales(t-1) + sales(t-2) + sales(t-3) + sales(t-4)) / 4.
Đặc trưng này phản ánh quy mô nhu cầu nền tảng của sản phẩm, lọc bỏ các nhiễu đột biến cục bộ của từng tuần riêng lẻ.

Bước 2. Tính toán độ lệch chuẩn doanh thu trượt 4 tuần (sales_std_4_weeks)
Định nghĩa: độ lệch chuẩn mẫu của doanh thu trong cửa sổ 4 tuần gần nhất:
sales_std_4_weeks = sqrt( 1/3 * sum_{k=1}^4 (sales(t-k) - sales_mean_4_weeks)^2 ).
Đặc trưng này đo lường mức độ bất định và rủi ro biến động doanh số của sản phẩm.

Bước 3. Xử lý trường hợp sản phẩm mới có lịch sử dưới 4 tuần
Nếu sản phẩm mới phát sinh đơn hàng được ít hơn 4 tuần, tính trung bình và độ lệch chuẩn trên số tuần thực tế đã quan sát được.
Nếu chỉ có 1 tuần quan sát, gán sales_std_4_weeks = 0.0 BRL.

Bước 4. Đánh giá vai trò của biến thống kê trượt trong việc giảm phương sai
Phân tích đồ thị phân tán: sales_mean_4_weeks có mối quan hệ phi tuyến mạnh mẽ với doanh thu tuần kế tiếp, là một trong những đặc trưng chủ lực tại các nút phân nhánh trên cùng của Decision Tree.

Bước 5. Kiểm toán độ trễ thời gian
Bảo đảm toàn bộ cửa sổ 4 tuần kết thúc trước ngày thứ Hai của tuần target, tuân thủ nguyên tắc chống rò rỉ dữ liệu.
