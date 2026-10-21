Bản v5. Nghiên Cứu Mô Hình Hai Giai Đoạn (Two-Stage Hurdle Regressor)
Mục tiêu của bản này là xây dựng và kiểm chứng mô hình Hai giai đoạn thuần túy From-Scratch để giải quyết triệt để bài toán lạm phát số 0 (>90% tuần bằng 0 BRL).

Bước 1. Thiết kế cấu trúc TwoStageHurdleRegressorFromScratch
Triển khai trong src/models/hurdle.py gồm 2 thành phần độc lập:
- self.classifier_: Cây quyết định học xác suất phát sinh giao dịch p(x) in [0, 1].
- self.regressor_: Cây quyết định học doanh thu thực tế khi có đơn hàng y_pos in R+.

Bước 2. Huấn luyện Giai đoạn 1 (Occurrence Model)
Tạo nhãn nhị phân z = (y > 0).
Huấn luyện cây quyết định DecisionTreeRegressorFromScratch(max_depth=5, min_samples_leaf=5) trên toàn bộ tập dữ liệu với nhãn z.
Tại mỗi nút lá, giá trị trung bình chính là xác suất thực nghiệm p_i = P(sales > 0 | x).

Bước 3. Huấn luyện Giai đoạn 2 (Severity Model)
Lọc tập con dương: X_pos = X[y > 0], y_pos = y[y > 0] (khoảng 55.000 dòng).
Huấn luyện cây quyết định DecisionTreeRegressorFromScratch(max_depth=6, min_samples_leaf=5) chỉ trên tập con dương.
Cây hoàn toàn không bị 90% số 0 kéo tụt xuống, tập trung phân nhánh giải thích biên độ doanh thu.

Bước 4. Kết hợp kỳ vọng toán học và chuyển đổi Log
Khi suy luận: E[sales] = predict_proba_sale(X) * predict_positive_sales(X).
Chuyển đổi sang log: y_hat = log(1 + E[sales]) để đồng nhất giao diện toàn hệ thống.

Bước 5. Đánh giá hiệu năng đột phá
Mô hình đạt Test RMSE = 50,84 BRL và R2 = +0,0559 (cao nhất nhóm cơ sở), thời gian huấn luyện chỉ mất 0,41 giây nhờ kích thước tập dương nhỏ gọn.
