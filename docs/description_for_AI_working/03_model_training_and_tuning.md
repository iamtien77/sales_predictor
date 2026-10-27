# 🤖 Giai Đoạn 3: Huấn Luyện 5 Thuật Toán From-Scratch & Tối Ưu Siêu Tham Số (Phase 3 Playbook)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

> **Mục tiêu:** Xây dựng và huấn luyện 100% From-Scratch thuần NumPy cho 5 thuật toán hồi quy và 4 mô hình cơ sở đối chuẩn; tối ưu siêu tham số bằng GridSearchCV trên tập Validation 10% độc lập theo thời gian.  
> **Hàm điều phối:** `step_6_train_models()`, `step_7_hyperparameter_tuning()` trong [`main.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/main.py).  
> **Mã nguồn cốt lõi:** [`src/models/regressors.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/regressors.py), [`src/models/tree.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tree.py), [`src/models/ensemble.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/ensemble.py), [`src/models/linear.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/linear.py), [`src/models/hurdle.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/hurdle.py), [`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tuning.py).

---

### Bước 1. Thiết Lập Hệ Thống 4 Mô Hình Cơ Sở Đối Chuẩn (Baselines)
- Triển khai trong [`src/models/baselines.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/baselines.py) tuân thủ nguyên tắc Học máy (Rule 10: *Always propose a baseline before complex models*):
  1. `ZeroRegressorFromScratch`: Luôn dự báo doanh thu = 0 BRL (mốc chặn dưới MAE trên dữ liệu thưa 90% số 0).
  2. `RevenueMeanRegressorFromScratch`: Dự báo bằng giá trị trung bình doanh thu thực tế của tập Train: $\hat{y} = \log(1 + \bar{y}_{\text{money}})$.
  3. `MeanRegressorFromScratch`: Dự báo bằng trung bình nhãn log trong không gian tối ưu.
  4. `PreviousWeekRegressorFromScratch`: Mô hình quán tính chuỗi thời gian, lấy doanh thu tuần trước ($t-1$) làm dự báo tuần này.

### Bước 2. Huấn Luyện Hồi Quy Tuyến Tính Có Kẹp Biên Phòng Vệ (Linear Regression)
- Triển khai lớp `LinearRegressionFromScratch` trong [`src/models/linear.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/linear.py).
- Sử dụng bộ giải nghiệm giải tích đóng Normal Equation qua phép nhân ma trận thưa và giả nghịch đảo Moore-Penrose:
  
  $$\tilde{w}^* = \left(\tilde{X}^\top \tilde{X} + \alpha_{\text{ridge}} I_0\right)^{-1} \tilde{X}^\top y$$

- **Cơ chế Kẹp biên Thích ứng (Adaptive Clamping):** Nhằm ngăn chặn hiện tượng nổ số học ($1,30 \times 10^8$ BRL) khi nghịch đảo hàm mũ $\exp(w^T x) - 1$, mô hình tự động kẹp biên dự báo log trong miền $[y_{\min} - 0.5 \cdot \text{span}, y_{\max} + 0.5 \cdot \text{span}]$. Thời gian huấn luyện siêu tốc: **0,02 giây**.

### Bước 3. Huấn Luyện Cây Quyết Định Hồi Quy (Decision Tree Regressor)
- Triển khai lớp `DecisionTreeRegressorFromScratch` trong [`src/models/tree.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tree.py).
- Hàm độ lợi tối ưu hóa: **Giảm phương sai (Variance Reduction)** tại mỗi nút phân nhánh:
  
  $$\Delta \text{Var} = \text{Var}(D) - \left(\frac{|D_L|}{|D|}\text{Var}(D_L) + \frac{|D_R|}{|D|}\text{Var}(D_R)\right)$$

- **Tối ưu hóa 16 Quantiles:** Quét 16 điểm phân vị ứng viên trên 655k mẫu, giúp cây khớp nhanh chóng chỉ trong **15,22 giây**.

### Bước 4. Huấn Luyện Các Mô Hình Tổ Hợp (Random Forest & Gradient Boosting)
- Triển khai trong [`src/models/ensemble.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/ensemble.py):
  - `RandomForestRegressorFromScratch`: Huấn luyện $B$ cây độc lập qua lấy mẫu Bootstrap có hoàn lại và chọn ngẫu nhiên $k = \lfloor\sqrt{89}\rfloor = 9$ đặc trưng ở mỗi nút để triệt tiêu phương sai.
  - `GradientBoostingRegressorFromScratch`: Huấn luyện tuần tự $M$ cây quyết định nông trên phần dư âm (Residuals) với hệ số co hẹp $\eta = 0,05$ để giảm độ lệch (Bias).

### Bước 5. Huấn Luyện Mô Hình Hai Giai Đoạn (Two-Stage Hurdle Regressor)
- Triển khai trong [`src/models/hurdle.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/hurdle.py) đặc trị phân phối Zero-Inflated:
  - **Stage 1 (Occurrence):** Cây quyết định phân loại ước lượng xác suất phát sinh đơn hàng $p(x) = P(\text{sales} > 0 \mid x) \in [0, 1]$.
  - **Stage 2 (Severity):** Cây hồi quy chỉ học trên tập con dương ($y > 0$) để dự báo độ lớn doanh thu điều kiện $\hat{s}_{\text{pos}}(x) = \mathbb{E}[\text{sales} \mid \text{sales} > 0, x]$.
  - **Kỳ vọng hợp nhất:** $\mathbb{E}[\text{sales}] = p(x) \times \hat{s}_{\text{pos}}(x)$.
- Đạt hiệu năng cơ sở tốt nhất: **RMSE = 50,84 BRL, $R^2 = +0,0559$**, thời gian huấn luyện chỉ **0,41 giây**.

### Bước 6. Tinh Chỉnh Siêu Tham Số GridSearchCV trên Validation 10%
- Triển khai hàm `grid_search_from_scratch()` trong [`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tuning.py).
- Dò tìm lưới cấu hình trên 249.399 dòng Validation:
  - Decision Tree: `max_depth` $\in [3, 5, 8]$, `min_samples_leaf` $\in [10, 30]$.
  - Random Forest: `n_estimators` $\in [10, 20]$, `max_depth` $\in [5, 8]$, `min_samples_split` $\in [10, 20]$.
  - Gradient Boosting: `n_estimators` $\in [15, 30]$, `max_depth` $\in [2, 3]$, `learning_rate` $= 0.1$.
- Lưu toàn bộ lịch sử thử nghiệm ra file `results/metrics/tuning_history.csv` và lựa chọn cấu hình có Validation RMSE BRL thấp nhất.
