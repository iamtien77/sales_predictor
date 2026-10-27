Bản v4. Huấn Luyện Cây Quyết Định Hồi Quy (Decision Tree Regressor)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

Mục tiêu của bản này là xây dựng cây hồi quy đệ quy nhị phân thuần NumPy theo tiêu chuẩn giảm phương sai, tối ưu hóa tốc độ bằng 16 phân vị ứng viên.

Bước 1. Xây dựng cấu trúc nút cây DecisionTreeNode
Định nghĩa đối tượng nút trong src/models/tree.py lưu trữ: feature_idx, threshold, left, right, value, variance_reduction, n_samples.
Nút lá được xác định khi left và right đều là None và value chứa giá trị dự báo trung bình mẫu.

Bước 2. Cài đặt tiêu chuẩn phân chia Giảm phương sai (Variance Reduction)
Tại mỗi nút D có N mẫu, tính phương sai: Var(D) = mean(y^2) - mean(y)^2.
Độ lợi phân chia theo ngưỡng theta: Delta_Var = Var(D) - (|D_L|/N * Var(D_L) + |D_R|/N * Var(D_R)).
Thuật toán tìm kiếm thuộc tính j* và ngưỡng theta* làm cực đại hóa Delta_Var.

Bước 3. Tối ưu hóa duyệt ngưỡng bằng 16 Quantiles ứng viên
Thay vì duyệt qua toàn bộ các giá trị phân biệt O(N log N) gây nghẽn CPU, mô hình trích xuất 16 điểm phân vị (từ 5% đến 95%) qua np.percentile.
Giảm số phép thử ngưỡng xuống đúng 16 lần trên mỗi đặc trưng, tăng tốc độ huấn luyện gấp hàng trăm lần.

Bước 4. Cài đặt điều kiện dừng đệ quy
Dừng phân nhánh khi đạt một trong các điều kiện:
- Độ sâu hiện tại đạt max_depth (mặc định = 5).
- Số mẫu tại nút nhỏ hơn min_samples_split (mặc định = 10).
- Số mẫu ở nút con nhỏ hơn min_samples_leaf (mặc định = 5).
- Độ lợi giảm phương sai Delta_Var <= 0.
Giá trị tại nút lá là trung bình cộng của các mẫu rơi vào lá.

Bước 5. Tăng tốc độ suy luận bằng Batch Masks
Phương thức predict(X) duyệt cây bằng mặt nạ nhị phân trên toàn bộ mảng thay vì duyệt từng dòng, giúp suy luận trên 293.717 dòng Test chỉ mất 0,40 giây.
