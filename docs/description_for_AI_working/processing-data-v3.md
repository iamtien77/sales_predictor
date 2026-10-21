Bản v3. Xây Dựng Bảng Chuỗi Thời Gian Sản Phẩm – Tuần
Mục tiêu của bản này là chuyển đổi dữ liệu giao dịch item thành bảng chuỗi thời gian Sản phẩm – Tuần (Weekly Product Panel) bảo toàn các tuần 0 BRL.

Bước 1. Xác định mốc tuần lịch chuẩn cho từng giao dịch
Tính toán ngày thứ Hai của tuần chứa giao dịch: week_start = timestamp - timedelta(days=timestamp.weekday()).
Quy định tuần lịch từ 00:00:00 thứ Hai đến 23:59:59 Chủ nhật.
Gán nhãn week_start cho toàn bộ các giao dịch phát sinh trong tuần đó.

Bước 2. Tổng hợp doanh thu và sản lượng theo từng cặp Sản phẩm – Tuần
Gom nhóm dữ liệu theo [product_id, week_start]:
- sales = sum(price): Tổng doanh thu tuần (BRL).
- units = count(order_item_id): Tổng số lượng sản phẩm bán được trong tuần.
- price = mean(price): Giá bán trung bình trong tuần.

Bước 3. Tạo lưới thời gian đầy đủ từ tuần bán đầu tiên
Với mỗi mã product_id:
- Xác định min_week (tuần đầu tiên sản phẩm phát sinh đơn hàng) và max_week (tuần cuối cùng của hệ thống).
- Tạo chuỗi tuần liên tục với bước nhảy 1 tuần (freq='W-MON') giữa min_week và max_week.
- Không tạo các tuần trước min_week để tránh giả định sai về thời điểm ra mắt sản phẩm.

Bước 4. Điền giá trị 0 BRL cho các tuần không phát sinh đơn hàng
Ghép nối khung tuần liên tục với dữ liệu bán hàng thực tế qua phép nối trái (Left Join).
Điền sales = 0.0 BRL và units = 0 cho các tuần không có giao dịch.
Áp dụng forward-fill cho giá bán price từ tuần có giao dịch gần nhất trước đó.

Bước 5. Khóa quy mô bảng Weekly Panel
Thu được tập dữ liệu bảng chuỗi thời gian gồm 1.198.182 dòng Sản phẩm – Tuần.
Bảo toàn hơn 90% số tuần có doanh thu bằng 0 BRL, phản ánh chính xác đặc thù nhu cầu thưa thớt của bài toán bán lẻ.
