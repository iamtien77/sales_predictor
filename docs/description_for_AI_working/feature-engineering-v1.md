Bản v1. Cấu Trúc Ma Trận Đặc Trưng Thưa & Quản Trị Bộ Nhớ RAM
Mục tiêu của bản này là thiết kế cấu trúc lưu trữ và tính toán ma trận đặc trưng hiệu quả cho 1,2 triệu mẫu sản phẩm–tuần mà không bị tràn bộ nhớ RAM.

Bước 1. Phân tích chi phí bộ nhớ của ma trận đặc (Dense Array)
Tập dữ liệu bảng chứa 1.198.182 dòng và 89 đặc trưng (16 số học + 73 danh mục One-Hot).
Nếu lưu dưới dạng mảng dense NumPy float64: dung lượng tiêu tốn = 1.198.182 x 89 x 8 bytes ≈ 853 MB cho mỗi phân vùng.
Trong 73 cột One-Hot, mỗi dòng chỉ có đúng 1 giá trị bằng 1 và 72 giá trị bằng 0 (độ thưa > 98%).

Bước 2. Chuyển đổi sang ma trận thưa SciPy CSR (Compressed Sparse Row)
Sử dụng scipy.sparse.csr_matrix để nén ma trận đặc trưng sau khi mã hóa One-Hot.
Cấu trúc CSR chỉ lưu trữ các giá trị khác 0, chỉ số cột tương ứng và con trỏ hàng.
Dung lượng bộ nhớ giảm từ hơn 850 MB xuống chỉ còn khoảng 120 MB (tiết kiệm hơn 85% RAM).

Bước 3. Tối ưu hóa phép nhân ma trận Gram X.T @ X cho Linear Regression
Tận dụng toán tử @ trực tiếp trên ma trận thưa CSR: X_sparse.T @ X_sparse.
Thời gian tính toán ma trận Gram kích thước (90, 90) từ 655.066 dòng Train giảm xuống chỉ còn 0,02 giây.

Bước 4. Hỗ trợ chuyển đổi linh hoạt Dense / Sparse cho các thuật toán gốc cây
Đối với Decision Tree, Random Forest và Gradient Boosting: hỗ trợ hàm tiện ích _ensure_dense_2d() để trích xuất các mảng 2D liên tục phục vụ quét phân vị và phân nhánh nhanh.

Bước 5. Kiểm tra tính toàn vẹn của ma trận số
Xác thực điều kiện: ma trận không chứa bất kỳ giá trị NaN hoặc Inf nào trước khi chuyển sang bước kỹ nghệ đặc trưng chi tiết.
