Bản v6. Đóng Gói Model Persistence & Kiểm Tra Khả Năng Tái Lập

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là lưu trữ toàn bộ trạng thái tham số của mô hình và pipeline thành một artifact duy nhất, đảm bảo tính tái lập thực nghiệm 100%.

Bước 1. Cấu trúc gói artifact đóng gói
Triển khai hàm save_artifacts() trong src/models/persistence.py.
Gói artifact best_sales_model.pkl bao gồm một từ điển chứa đầy đủ:
- 'model': Đối tượng mô hình tối ưu đã huấn luyện hoàn chỉnh.
- 'pipeline': Đối tượng CustomDataPipeline đã fit trên tập Train.
- 'feature_names': Danh sách 89 tên đặc trưng theo đúng thứ tự cột.
- 'metadata': Thông tin phiên bản, thời gian huấn luyện, tác giả và các chỉ số nghiệm thu.

Bước 2. Lưu trữ artifact vào results/saved_models/
Lưu tệp nhị phân best_sales_model.pkl vào thư mục results/saved_models/.
Xác nhận kích thước file hợp lý (~vài MB) và quyền ghi hoàn tất trơn tru.

Bước 3. Kiểm tra nạp lại và tính tái lập (Reproducibility Test)
Triển khai hàm load_artifacts() để nạp ngược lại bundle từ file pkl.
Chạy dự báo lại trên 100 mẫu kiểm thử và so sánh kết quả với lúc vừa huấn luyện xong.
Xác thực: độ lệch số học tuyệt đối giữa hai lần dự báo bằng đúng 0.00000000.

Bước 4. Đảm bảo tính độc lập khi triển khai
Gói artifact hoàn toàn khép kín, ứng dụng web Streamlit chỉ cần nạp file pkl này là có thể tiền xử lý và suy luận trực tiếp cho bất kỳ dữ liệu mới nào mà không cần nạp lại dữ liệu thô.
