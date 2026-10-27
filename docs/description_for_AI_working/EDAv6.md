Bản v6. Phân Bố 73 Ngành Hàng & Ma Trận Tương Quan Đặc Trưng

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là phân tích cơ cấu ngành hàng sản phẩm trên sàn Olist và khảo sát ma trận tương quan giữa các đặc trưng dự báo với biến mục tiêu doanh thu.

Bước 1. Thống kê tỷ trọng doanh thu giữa 73 danh mục sản phẩm Olist
Đếm số lượng sản phẩm và tổng doanh thu theo từng nhóm category.
Xác định Top 5 ngành hàng đóng góp doanh thu lớn nhất: cama_mesa_banho (đồ gia dụng phòng ngủ/tắm), informatica_acessorios (linh kiện máy tính), beleza_saude (sức khỏe sắc đẹp), relogios_presentes (đồng hồ quà tặng) và moveis_decoracao (nội thất).
Ghi nhận nhiều danh mục ngách có số lượng sản phẩm ít và doanh số thưa.

Bước 2. Phân tích mối liên hệ giữa giá bán, thuộc tính vật lý và doanh thu
Khảo sát tương quan giữa giá bán trung bình (price) với tổng doanh thu: tương quan dương rõ rệt.
Khảo sát các thuộc tính kích thước, khối lượng và số ảnh mô tả: sản phẩm có nhiều ảnh và mô tả chi tiết thường có tỷ lệ phát sinh đơn hàng cao hơn.

Bước 3. Phân tích tương quan giữa các biến trễ và doanh thu tuần
Tính toán hệ số tương quan Pearson giữa biến mục tiêu sales với các biến trễ:
- last_week_sales (doanh thu tuần trước) có hệ số tương quan cao nhất (~0.35).
- sales_mean_4_weeks (trung bình trượt 4 tuần) có tương quan mạnh thứ hai (~0.32).
- Các biến trễ sâu hơn (lag 2, lag 4) có tương quan giảm dần theo quy luật phân rã chuỗi thời gian.

Bước 4. Kiểm tra đa cộng tuyến giữa các đặc trưng trễ
Đo lường mức độ phụ thuộc giữa sales_mean_4_weeks và last_week_sales.
Xác nhận rằng mô hình hồi quy tuyến tính OLS có thể bị ảnh hưởng bởi đa cộng tuyến, đòi hỏi phải có điều chuẩn Ridge nhỏ (alpha = 1e-5).
Ngược lại, các mô hình gốc cây hoàn toàn miễn nhiễm với vấn đề đa cộng tuyến này.

Bước 5. Xuất bản báo cáo tổng kết EDA
Tổng hợp toàn bộ các kết luận EDA thành tài liệu làm căn cứ cho việc thiết kế 89 đặc trưng và lựa chọn các thuật toán học máy phù hợp.
