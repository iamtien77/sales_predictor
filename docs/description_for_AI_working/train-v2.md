Bản v2. Huấn Luyện Hệ Thống 4 Mô Hình Cơ Sở (Baselines)
Mục tiêu của bản này là huấn luyện và thiết lập mốc đối chứng tham chiếu tối thiểu cho toàn bộ hệ thống học máy.

Bước 1. Huấn luyện Zero Baseline (ZeroRegressorFromScratch)
Triển khai trong src/models/baselines.py:
Phương thức predict(X) luôn trả về mảng toàn số 0.0 BRL.
Ý nghĩa: Mốc chặn dưới cho sai số tuyệt đối MAE trên tập dữ liệu thưa 90% số 0.

Bước 2. Huấn luyện Revenue Mean Baseline (RevenueMeanRegressorFromScratch)
Tính trung bình cộng doanh thu thực tế trên toàn bộ 655.066 dòng Train: mean_sales ≈ 11.90 BRL.
Lưu giá trị log tương ứng: value_ = log(1 + 11.90) ≈ 2.557.
Phương thức predict(X) luôn trả về hằng số value_.
Ý nghĩa: Nghiệm tối ưu lý thuyết của hàm mất mát MSE trên tập Train (R2_train = 0).

Bước 3. Huấn luyện Log Mean Baseline (MeanRegressorFromScratch)
Tính trung bình cộng của vector nhãn log trên tập Train: mean_log ≈ 0.283.
Phương thức predict(X) luôn trả về hằng số mean_log.
Ý nghĩa: Mốc đối chứng hằng số trong cùng không gian tối ưu hóa log.

Bước 4. Huấn luyện Previous Week Baseline (PreviousWeekRegressorFromScratch)
Xác định vị trí cột last_week_sales trong ma trận đặc trưng.
Phương thức predict(X) trích xuất giá trị doanh thu tuần trước (sau khi nghịch đảo chuẩn hóa Z-score) làm dự báo cho tuần này.
Ý nghĩa: Mô hình quán tính chuỗi thời gian tiêu chuẩn để kiểm chứng năng lực nắm bắt xu hướng.

Bước 5. Đóng gói kết quả 4 Baselines vào trained_models
Lưu trữ 4 mô hình cơ sở vào từ điển trained_models trong main.py với thời gian huấn luyện = 0.00 giây.
