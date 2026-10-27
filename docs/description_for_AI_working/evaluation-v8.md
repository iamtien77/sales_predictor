Bản v8. Xây Dựng Mô Hình Hồi Quy Sản Phẩm Mới (Cold-Start Content-based Regressor From-Scratch)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là giải quyết triệt để bài toán Cold-Start cho các sản phẩm mới ra mắt hoàn toàn chưa có lịch sử bán hàng trong panel tuần, bằng cách học mối quan hệ phi tuyến giữa các đặc trưng nội dung (Content-based features) với doanh số tuần mở bán đầu tiên (Launch Week Sales).

Bước 1. Trích xuất tập dữ liệu tuần mở bán (Launch Week Dataset)
Từ 110.197 dòng giao dịch Olist, xác định mốc thời gian bán hàng đầu tiên (first_time) của từng mã sản phẩm (product_id).
Gom nhóm các giao dịch phát sinh trong vòng 7 ngày đầu tiên kể từ mốc first_time để xác định doanh thu tuần ra mắt (launch_sales_brl = price * launch_units).
Trích xuất thành công 32.216 sản phẩm độc lập phục vụ bài toán Cold-Start.

Bước 2. Thiết lập đường cơ sở đối chuẩn CategoryMedianColdStartBaseline
Cài đặt lớp CategoryMedianColdStartBaseline trong src/models/cold_start.py kế thừa BaseRegressorFromScratch.
Tính toán trung vị doanh số tuần đầu tiên theo từng danh mục sản phẩm trên tập Train.
Cung cấp cơ chế tự động fallback về trung vị toàn sàn (global median) khi gặp danh mục hoàn toàn mới chưa từng xuất hiện.

Bước 3. Huấn luyện mô hình cây nội dung ColdStartProductRegressor From-Scratch
Cài đặt lớp ColdStartProductRegressor trong src/models/cold_start.py.
Trích xuất 6 đặc trưng nội dung cốt lõi: mức giá niêm yết (price), trọng lượng (product_weight_g), số lượng ảnh (product_photos_qty), độ dài tiêu đề (product_name_lenght), độ dài mô tả (product_description_lenght) và mã hóa danh mục bằng Target Encoding có Laplace Smoothing.
Sử dụng DecisionTreeRegressorFromScratch huấn luyện trong không gian Log tự nhiên log(1 + y) và nghịch đảo expm1 khi dự báo (BRL).

Bước 4. Đánh giá thực nghiệm độc lập và kiểm chứng vượt bậc
Trên tập kiểm nghiệm độc lập gồm các sản phẩm chưa từng xuất hiện ở Train:
- Baseline trung vị danh mục: MAE = 119.68 BRL, RMSE = 348.39 BRL.
- ColdStartProductRegressor: MAE = 41.60 BRL (cải thiện gần 3 lần), RMSE = 253.15 BRL, R² Score = +0.5301.
Chứng minh năng lực giải thích phương sai vượt trội của các đặc tính nội dung sản phẩm trong tuần mở bán đầu tiên.

Bước 5. Đóng gói hàm triển khai predict_cold_start_product()
Tích hợp hàm tiện ích predict_cold_start_product() vào src/deployment/predictor.py và Thẻ 4 trên giao diện Streamlit app.py, trả về doanh thu dự báo tuần đầu và sản lượng ước tính tương ứng.
