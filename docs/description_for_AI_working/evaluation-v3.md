Bản v3. Tính Toán Chỉ Số Chuỗi Cung Ứng Bán Lẻ (WAPE)
Mục tiêu của bản này là bổ sung chỉ số Weighted Absolute Percentage Error (WAPE) chuẩn mực cho e-commerce, khắc phục nhược điểm chia cho 0 của MAPE.

Bước 1. Phân tích sự sụp đổ của MAPE trên dữ liệu thưa
Chỉ số MAPE = 1/N * sum(|y - y_hat| / y).
Trên tập dữ liệu có hơn 90% số tuần có doanh thu y = 0 BRL, phép chia cho 0 khiến MAPE tiến tới vô cùng (undefined/inf), hoàn toàn không thể sử dụng được.

Bước 2. Định nghĩa công thức chuẩn của WAPE
Triển khai hàm compute_wape() trong src/evaluation/metrics.py:
WAPE = sum_{i=1}^N |y_i - y_hat_i| / sum_{i=1}^N y_i.
Chia tổng sai số tuyệt đối cho tổng doanh thu thực tế toàn sàn, hoàn toàn không bị lỗi chia cho 0 ở từng mẫu đơn lẻ.

Bước 3. Tính toán WAPE cho toàn bộ 10 mô hình
- Random Forest đạt WAPE tốt nhất toàn bảng: 1,0346 (~1.03).
- Decision Tree và Tuned Model đạt WAPE: 1,039 (~1.04).
- Gradient Boosting đạt WAPE: 1,039 (~1.04).
- Revenue Mean Baseline đạt WAPE: 2,78.
- Linear Regression đạt WAPE: 17,96.

Bước 4. Ý nghĩa nghiệp vụ của WAPE
WAPE đo lường tỷ lệ sai số tổng thể so với quy mô kinh doanh của doanh nghiệp, giúp giám đốc chuỗi cung ứng biết được mức độ sai lệch của kế hoạch nhập hàng theo tuần.

Bước 5. Đưa WAPE vào báo cáo so sánh
Tích hợp cột WAPE vào bảng xếp hạng chính thức trong model_comparison.csv.
