Bản v2. Tính Toán Bộ 4 Chỉ Số Hồi Quy Chuẩn (MSE, MAE, RMSE, R2)
Mục tiêu của bản này là đo lường định lượng hiệu năng của toàn bộ các mô hình theo các tiêu chuẩn học máy quốc tế.

Bước 1. Tính toán Mean Squared Error (MSE)
MSE = 1/N * sum_{i=1}^N (y_i - y_hat_i)^2 (đơn vị: BRL^2).
Đo lường mức độ phạt nặng đối với các sai số lớn.
Ghi nhận: Two-Stage Hurdle đạt MSE thấp nhất (2.585,01 BRL^2).

Bước 2. Tính toán Mean Absolute Error (MAE)
MAE = 1/N * sum_{i=1}^N |y_i - y_hat_i| (đơn vị: BRL).
Đo sai lệch tuyệt đối trung bình giữa dự báo và thực tế.
Ghi nhận: Zero Baseline đạt MAE nhỏ nhất (4,28 BRL), Tuned Model đạt 4,45 BRL, Random Forest đạt 4,43 BRL.

Bước 3. Tính toán Root Mean Squared Error (RMSE)
RMSE = sqrt(MSE) (đơn vị: BRL).
Chỉ số sai số cốt lõi trong kinh doanh, cùng đơn vị tiền tệ với doanh thu.
Ghi nhận: Two-Stage Hurdle đạt RMSE = 50,84 BRL, Tuned Model đạt 51,16 BRL, Decision Tree đạt 51,72 BRL.

Bước 4. Tính toán Hệ số xác định (R2 Score)
R2 = 1 - sum((y_i - y_hat_i)^2) / sum((y_i - mean(y))^2).
Đo lường tỷ lệ phương sai thực tế mà mô hình giải thích được.
Ghi nhận: Cả 4 mô hình gốc cây đều đạt R2 > 0 (Two-Stage Hurdle: +0,0559; Tuned Model: +0,0440), trong khi toàn bộ 4 mô hình Baseline đều có R2 < 0.

Bước 5. Xuất bảng tổng hợp model_comparison.csv
Tập hợp toàn bộ các chỉ số vào summary_df và lưu ra file results/metrics/model_comparison.csv.
