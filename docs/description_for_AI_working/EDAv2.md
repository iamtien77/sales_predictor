Bản v2. Chuẩn Hóa Thời Gian & Cấu Trúc Bảng Sản Phẩm – Tuần

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là chuyển đổi mốc thời gian giao dịch riêng lẻ sang tuần lịch thống nhất và thiết lập khung dữ liệu chuỗi thời gian Sản phẩm – Tuần (Weekly Panel).

Bước 1. Chuyển đổi trường order_purchase_timestamp sang datetime64[ns]
Định dạng lại trường thời gian mua hàng về chuẩn ISO-8601.
Xử lý các mốc thời gian biên và loại trừ các bản ghi có định dạng thời gian bất thường.
Đảm bảo mốc thời gian được ghi nhận chính xác theo giờ địa phương Brazil.

Bước 2. Chuẩn hóa chu kỳ tuần lịch từ thứ Hai đến Chủ nhật
Quy định tuần lịch chuẩn: mỗi tuần bắt đầu từ 00:00:00 thứ Hai và kết thúc vào 23:59:59 Chủ nhật.
Gán nhãn thời gian của từng giao dịch về ngày thứ Hai của tuần phát sinh giao dịch.
Cột order_purchase_timestamp trong weekly panel từ đây đại diện cho ngày bắt đầu của tuần cần dự báo.

Bước 3. Nhận diện tuần phát sinh giao dịch đầu tiên của từng sản phẩm
Với mỗi mã product_id, quét toàn bộ lịch sử để tìm mốc thời gian của đơn hàng đầu tiên (first_sale_week).
Quy định nguyên tắc quan sát: một sản phẩm chỉ bắt đầu được theo dõi từ tuần phát sinh đơn đầu tiên.
Không tạo dữ liệu giả định trước thời điểm sản phẩm xuất hiện trên sàn Olist.

Bước 4. Xây dựng khung thời gian liên tục cho từng sản phẩm
Tạo lưới thời gian liên tục từ tuần bán đầu tiên đến tuần cuối cùng của tập dữ liệu cho từng sản phẩm.
Tính toán độ dài vòng đời quan sát của sản phẩm (số tuần từ lần bán đầu).
Đảm bảo không bị đứt đoạn các tuần trung gian giữa hai đợt phát sinh đơn hàng.

Bước 5. Điền đầy đủ các tuần không có giao dịch với doanh thu 0 BRL
Tại các tuần sản phẩm không phát sinh đơn hàng, điền giá trị doanh thu bằng đúng 0.0 BRL.
Tính toán tổng số lượng bản ghi của bảng chuỗi thời gian Sản phẩm – Tuần (đạt 1.198.182 dòng).
Ghi nhận đây là tập dữ liệu bảng đầy đủ, trung thực phản ánh lịch sử kinh doanh trên sàn thương mại điện tử.
