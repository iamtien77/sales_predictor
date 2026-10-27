Bản v5. Phân Tích Phần Dư & Mổ Xẻ Top 20 Ca Lệch Nhất (Error Analysis)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là mổ xẻ nguyên nhân kinh tế và thống kê của các trường hợp dự báo có sai lệch tuyệt đối lớn nhất trên tập Test.

Bước 1. Trích xuất Top 20 trường hợp sai lệch lớn nhất
Triển khai hàm extract_top_worst_predictions() trong src/evaluation/error_analysis.py.
Sắp xếp tập Test theo thứ tự giảm dần của sai số tuyệt đối error_abs = |y_true - y_pred| và lấy ra 20 bản ghi đứng đầu.
Lưu bảng phân tích ra file results/metrics/worst_predictions_analysis.csv.

Bước 2. Mổ xẻ nhóm nguyên nhân 1: Đột biến đơn sỉ B2B
Nhiều đơn hàng có giá trị đột biến lên tới hàng nghìn BRL do khách hàng doanh nghiệp đặt mua số lượng lớn vào một tuần cá biệt.
Lịch sử các tuần trước của sản phẩm hoàn toàn bình thường, khiến mô hình không thể có tín hiệu báo trước.

Bước 3. Mổ xẻ nhóm nguyên nhân 2: Hiện tượng trễ pha sau Flash Sales
Một số sản phẩm tham gia chương trình khuyến mãi chớp nhoáng đạt đỉnh doanh thu ở tuần t-1, nhưng sang tuần t thì sức mua giảm mạnh về 0.
Các mô hình dựa vào biến trễ last_week_sales bị đánh lừa và dự báo doanh số cao cho tuần t.

Bước 4. Mổ xẻ nhóm nguyên nhân 3: Thiếu thông tin tồn kho (Out-of-Stock)
Tập dữ liệu Olist không có trường tồn kho hàng ngày.
Một sản phẩm có doanh thu bằng 0 ở tuần t có thể do sản phẩm đã hết hàng trong kho chứ không phải do người tiêu dùng không có nhu cầu.

Bước 5. Đề xuất khuyến nghị vận hành
Khuyến nghị doanh nghiệp tích hợp thêm hệ thống kiểm soát tồn kho thời gian thực (ERP/WMS) để phân biệt giữa hết hàng và hết nhu cầu.
