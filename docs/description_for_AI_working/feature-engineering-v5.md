Bản v5. Kỹ Nghệ Biến Phản Hồi Khách Hàng Chống Rò Rỉ (Anti-Leakage Reviews)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là đo lường uy tín chất lượng sản phẩm qua đánh giá của khách hàng mà không làm lộ dữ liệu tương lai vào quá khứ.

Bước 1. Phân tích cơ chế phát sinh đánh giá trên Olist
Khách hàng chỉ gửi phản hồi và chấm điểm sau khi đơn hàng được giao thành công (thời điểm review_answer_timestamp thường trễ từ 7 đến 21 ngày so với lúc đặt hàng).
Nếu lấy đánh giá của các đơn đặt trong tuần target thì sẽ vi phạm nghiêm trọng nguyên tắc chuỗi thời gian vì tại đầu tuần target, các đánh giá này chưa hề tồn tại trong thực tế.

Bước 2. Thiết lập bộ lọc thời gian nghiêm ngặt cho đánh giá
Chỉ tổng hợp các đánh giá có review_answer_timestamp < target_week_start.
Toàn bộ các phản hồi gửi sau mốc bắt đầu tuần target đều bị loại bỏ khỏi biến dự báo của tuần đó.

Bước 3. Tính toán tổng số lượng đánh giá lũy kế (past_review_count)
Đếm tổng số lượt đánh giá hợp lệ mà sản phẩm đã nhận được tính đến trước tuần target.
Đại diện cho độ tin cậy và khối lượng phản hồi thực tế từ cộng đồng người mua.

Bước 4. Tính toán điểm số sao trung bình lũy kế (past_review_rating)
Tính trung bình cộng điểm số sao (thang điểm 1 đến 5) của các đánh giá đã nhận trước tuần target.
Nếu sản phẩm chưa có bất kỳ đánh giá nào trước thời điểm đó, gán giá trị mặc định là trung vị của toàn sàn (4.0 sao).

Bước 5. Kiểm toán xác thực Anti-Leakage
Kiểm tra ngẫu nhiên 100 sản phẩm: xác nhận 100% các đánh giá được dùng để tính past_review_rating đều có ngày trả lời nằm hoàn toàn trong quá khứ so với tuần target.
