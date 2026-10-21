Bản v4. Phân Tích Hiện Tượng Dữ Liệu Thưa & Zero-Inflation
Mục tiêu của bản này là định lượng mức độ thưa thớt của dữ liệu tuần, mổ xẻ nguyên nhân kinh tế và đề xuất định hướng mô hình hóa đặc trị cho phân phối nhiều số 0.

Bước 1. Đo lường tỷ lệ các tuần không có giao dịch (Zero-Sales)
Đếm số lượng bản ghi có sales = 0.0 BRL trong tổng số 1.198.182 dòng weekly panel.
Ghi nhận tỷ lệ tuần không có đơn hàng vượt mức 90%.
Chỉ có dưới 10% tổng số tuần quan sát là thực sự phát sinh doanh thu dương.

Bước 2. Phân tích thời gian ngắt quãng giữa các đợt phát sinh đơn hàng
Khảo sát phân phối khoảng cách thời gian (số tuần) giữa hai lần bán liên tiếp của cùng một sản phẩm.
Nhận diện đặc trưng nhu cầu ngắt quãng (Intermittent / Lumpy Demand): sản phẩm bán được 1 đơn trong tuần này thường sẽ im ắng trong 3-5 tuần tiếp theo trước khi có đơn mới.

Bước 3. Đánh giá tác động của 90% số 0 lên mô hình hồi quy tiêu chuẩn
Phân tích hàm mất mát bình phương MSE khi học trên tập dữ liệu thưa:
Do 90% mẫu có nhãn 0, hàm mất mát bị kéo mạnh về hướng dự báo các giá trị sát 0 cho toàn bộ mẫu.
Hệ quả: mô hình hồi quy thông thường bị hiện tượng nén phẳng (over-smoothing), không thể bắt được các đỉnh doanh thu thực tế khi có đơn hàng.

Bước 4. Đánh giá sai số của các dự báo ngây thơ
Kiểm tra hiệu năng của Zero Baseline (luôn dự báo 0 BRL): đạt MAE cực thấp (~4.28 BRL) nhưng hoàn toàn vô dụng về mặt kinh doanh vì không dự báo được bất kỳ đơn hàng nào.
Khẳng định MAE thấp trên tập dữ liệu thưa là một thước đo đánh lừa nếu không đối chiếu với RMSE và R2.

Bước 5. Đề xuất kiến trúc Mô hình Hai giai đoạn (Hurdle Model)
Định hướng giải pháp: tách rời bài toán thành 2 giai đoạn độc lập:
1. Giai đoạn 1: Ước lượng xác suất sản phẩm có người mua trong tuần P(sales > 0 | x).
2. Giai đoạn 2: Hồi quy độ lớn doanh thu điều kiện khi có đơn hàng E[sales | sales > 0, x].
Kết hợp kỳ vọng toán học: E[sales] = P(sales > 0) * E[sales | sales > 0].
