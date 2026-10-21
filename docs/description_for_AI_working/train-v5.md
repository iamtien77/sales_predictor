Bản v5. Huấn Luyện Các Mô Hình Tổ Hợp (Random Forest & Gradient Boosting)
Mục tiêu của bản này là cài đặt hai phương pháp học kết hợp Ensemble Bagging và Ensemble Boosting để so sánh hiệu năng giảm phương sai và giảm độ lệch.

Bước 1. Huấn luyện Random Forest Regressor (Ensemble Bagging)
Triển khai lớp RandomForestRegressorFromScratch trong src/models/ensemble.py:
- Khởi tạo B = 15 cây con DecisionTreeRegressorFromScratch.
- Với mỗi cây, rút ngẫu nhiên có hoàn lại N mẫu (Bootstrap Sampling, chiếm ~63.2% dữ liệu duy nhất).
- Tại mỗi nút phân nhánh của cây con, chỉ chọn ngẫu nhiên k = floor(sqrt(89)) = 9 đặc trưng (Feature Subsampling).
- Dự báo cuối cùng là trung bình cộng của 15 cây: y_hat = 1/B * sum(T_b(x)).
Thời gian huấn luyện trên Train: 42,67 giây.

Bước 2. Huấn luyện Gradient Boosting Regressor (Ensemble Boosting)
Triển khai lớp GradientBoostingRegressorFromScratch trong src/models/ensemble.py:
- Khởi tạo giá trị dự báo cơ sở F_0(x) = mean(y_train).
- Vòng lặp m = 1..M (M = 15 cây nông với max_depth = 3):
  1. Tính phần dư âm (Residual): r_i = y_i - F_{m-1}(x_i).
  2. Khớp cây quyết định nông h_m(x) trên cặp dữ liệu (X, r).
  3. Cập nhật dự báo tích lũy với hệ số co hẹp: F_m(x) = F_{m-1}(x) + 0.1 * h_m(x).
Thời gian huấn luyện: 42,83 giây.

Bước 3. Đánh giá tính ổn định của các mô hình tổ hợp
Cả hai mô hình đều hoàn thành huấn luyện trơn tru, không gặp lỗi bộ nhớ và đạt R2 dương trên tập Validation.

Bước 4. Tổng hợp các mô hình vào bộ điều phối
Đóng gói Random Forest và Gradient Boosting vào từ điển trained_models sẵn sàng cho bước đánh giá độc lập.
