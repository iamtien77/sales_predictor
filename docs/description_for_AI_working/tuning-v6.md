Bản v6. Tổng Hợp Kết Quả Tuning & Lựa Chọn Mô Hình Chiến Thắng

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là tổng hợp toàn bộ lịch sử thử nghiệm, lưu trữ file báo cáo và xác định mô hình tối ưu chung cuộc để đưa vào nghiệm thu độc lập.

Bước 1. Thu thập toàn bộ lịch sử tìm kiếm siêu tham số
Gom toàn bộ kết quả từ các lượt tìm kiếm của Random Forest, Decision Tree và Gradient Boosting thành danh sách từ điển gồm tên mô hình, bộ tham số và điểm Validation RMSE BRL.

Bước 2. Xuất bảng dữ liệu tuning_history.csv
Chuyển đổi danh sách lịch sử thành DataFrame pandas.
Lưu ra tệp results/metrics/tuning_history.csv phục vụ báo cáo và thẩm định thực nghiệm.

Bước 3. Áp dụng hàm chọn cực tiểu min() theo điểm số
Tìm kiếm mô hình có Validation_RMSE_BRL nhỏ nhất:
name, winner = min(searches, key=lambda pair: pair[1]["best_score"]).
Hệ thống tự động xác nhận Decision Tree (max_depth=8, min_samples_leaf=10) là Selected Winner Model với Val RMSE = 59.2288 BRL.

Bước 4. Đóng gói mô hình tối ưu đã fit đầy đủ
Lấy best_estimator từ kết quả tìm kiếm và gán nhãn "Tuned Model".
Chuẩn bị mô hình tối ưu cùng toàn bộ các mô hình khác sẵn sàng bước vào khâu kiểm định độc lập trên Test Set.

Bước 5. Ghi nhật ký hoàn thành Bước 7
Ghi log thông báo hoàn tất bước Hyperparameter Tuning, sẵn sàng chuyển sang Bước 8 nghiệm thu trên Test Set.
