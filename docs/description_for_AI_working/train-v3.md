Bản v3. Huấn Luyện Hồi Quy Tuyến Tính (Linear Regression With Clamping)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là huấn luyện mô hình tham số tuyến tính OLS và tích hợp cơ chế kẹp biên an toàn chống nổ số học.

Bước 1. Thiết kế lớp LinearRegressionFromScratch
Triển khai trong src/models/linear.py hỗ trợ 2 bộ giải:
- Normal Equation OLS (mặc định): Nghiệm giải tích đóng qua phương trình chuẩn tắc.
- Gradient Descent: Tối ưu hóa hạ độ dốc lặp.

Bước 2. Tính toán nghiệm giải tích Normal Equation qua ma trận thưa
Mở rộng ma trận với cột hệ số chặn bias: X_tilde = [1, X].
Tính toán ma trận hiệp phương sai qua phép nhân thưa: A = X_tilde.T @ X_tilde.
Bổ sung điều chuẩn Ridge nhỏ (alpha = 1e-5) vào đường chéo (trừ vị trí bias) để chống suy biến ma trận.
Giải hệ phương trình tuyến tính bằng giả nghịch đảo Moore-Penrose: w = pinv(A) @ (X_tilde.T @ y).

Bước 3. Phát hiện hiện tượng nổ số học khi nghịch đảo Log (Failure Mode)
Phân tích thực nghiệm: do siêu phẳng tuyến tính không có giới hạn trên, một vài mẫu ngoại lai trên tập Test bị dự báo log vượt ngưỡng (y_hat ≈ 38).
Khi nghịch đảo exp(38) - 1, giá trị bùng nổ lên 10^16 BRL, khiến RMSE vọt lên 130 triệu BRL.

Bước 4. Cài đặt cơ chế Kẹp biên Thích ứng (Adaptive Numerical Clamping)
Trong quá trình fit(), lưu lại y_min và y_max của tập Train cùng độ trải span = y_max - y_min.
Khi predict(), tự động kẹp biên dự báo log trong khoảng an toàn:
y_clamped = clip(y_hat, y_min - 0.5 * span, y_max + 0.5 * span).
Kết quả: Kéo RMSE từ 130 triệu BRL xuống mức kiểm soát được (5.400,88 BRL).

Bước 5. Đánh giá tốc độ huấn luyện
Toàn bộ quá trình tính toán Normal Equation trên 655.066 dòng Train chỉ mất đúng 0,02 giây.
