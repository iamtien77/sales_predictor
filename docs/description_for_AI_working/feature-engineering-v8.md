Bản v8. Ghép Nối Vector 89 Đặc Trưng & Lưu Cache Ma Trận
Mục tiêu của bản này là hoàn thiện việc ghép nối toàn bộ 16 biến số học và 73 biến danh mục, kiểm toán cấu trúc và lưu ma trận đệm vào đĩa.

Bước 1. Tổng hợp danh mục 16 đặc trưng số học
Danh sách 16 biến số học hoàn chỉnh:
price, last_week_sales, sales_lag_2_weeks, sales_lag_4_weeks,
sales_mean_4_weeks, sales_std_4_weeks, last_week_units, weeks_since_first_sale,
past_review_count, past_review_rating, customer_past_spend, customer_past_orders,
customer_state_count, is_peak_shopping_season, is_holiday_week, month_sin.

Bước 2. Tổng hợp danh mục 73 đặc trưng phân loại One-Hot
73 cột nhị phân tương ứng với 73 nhóm ngành hàng Olist:
category_agro_industria_e_comercio đến category_utilidades_domesticas.

Bước 3. Ghép nối thành ma trận đặc trưng 89 chiều
Sử dụng scipy.sparse.hstack để ghép nối ma trận số học đã chuẩn hóa Z-score với ma trận danh mục One-Hot.
Xác nhận kích thước ma trận:
- X_train: (655.066, 89)
- X_val: (249.399, 89)
- X_test: (293.717, 89)

Bước 4. Kiểm toán toàn diện chất lượng ma trận (Matrix Quality Audit)
Kiểm tra tính toàn vẹn: số phần tử NaN = 0, số phần tử Inf = 0.
Kiểm tra tính đồng nhất của số cột: đúng 89 cột trên cả 3 tập Train, Val, Test.

Bước 5. Lưu trữ ma trận đệm cache vào data/processed/
Lưu ma trận thưa qua sp.save_npz: cache_X_train.npz, cache_X_val.npz, cache_X_test.npz.
Lưu vector nhãn log qua np.save: cache_y_train.npy, cache_y_val.npy, cache_y_test.npy.
Lưu danh sách 89 tên đặc trưng vào cache_feature_names.json.
Lưu dấu vân tay SHA-256 vào manifest.json để tự động nhận diện thay đổi mã nguồn.
