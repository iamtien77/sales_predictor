Bản v1. Thiết Kế Kiến Trúc Học Máy From-Scratch Thuần NumPy

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là thiết lập lớp cơ sở BaseRegressorFromScratch và chuẩn hóa giao diện hàm mất mát hồi quy trong không gian log.

Bước 1. Thiết kế lớp cơ sở BaseRegressorFromScratch
Xây dựng lớp trừu tượng trong src/models/base.py kế thừa các phương thức chuẩn:
- fit(X, y): Huấn luyện mô hình, trả về chính đối tượng self.
- predict(X): Dự báo trên ma trận đặc trưng X, trả về mảng 1D NumPy.
- get_params() và set_params(): Hỗ trợ thiết lập tham số linh hoạt cho GridSearchCV.

Bước 2. Cài đặt các hàm đo lường sai số cơ sở thuần NumPy
Tự cài đặt các hàm tính metrics độc lập với scikit-learn:
- compute_mse(y_true, y_pred) = mean((y_true - y_pred)^2)
- compute_mae(y_true, y_pred) = mean(|y_true - y_pred|)
- compute_rmse(y_true, y_pred) = sqrt(compute_mse)
- compute_r2(y_true, y_pred) = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)

Bước 3. Chuẩn hóa không gian tối ưu hóa Log
Toàn bộ các mô hình hồi quy được thiết kế để nhận vector nhãn y = log(1 + sales) làm biến mục tiêu.
Hàm mất mát bình phương MSE trên không gian log tương đương với việc tối thiểu hóa sai số tương đối trên miền tiền tệ.

Bước 4. Xây dựng cơ chế chuyển đổi an toàn _safe_expm1
Cài đặt hàm nghịch đảo an toàn trong src/models/base.py:
def _safe_expm1(y_log): return np.maximum(0.0, np.expm1(y_log)).
Bảo đảm không bao giờ trả về giá trị âm cho doanh thu tiền tệ.

Bước 5. Kiểm thử đơn vị lớp cơ sở
Chạy bài kiểm tra trên dữ liệu giả lập nhỏ: xác nhận các phép tính sai số và giao diện kế thừa hoạt động chính xác 100%.
