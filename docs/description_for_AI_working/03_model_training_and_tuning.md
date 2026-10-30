# 🤖 Giai Đoạn 3: Huấn Luyện 5 Thuật Toán From-Scratch & Tối Ưu Siêu Tham Số (Phase 3 Playbook)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn thực nghiệm của dự án. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy hoàn chỉnh trên toàn bộ dữ liệu.

> **Mục tiêu kỹ thuật:** Xây dựng và huấn luyện 100% From-Scratch thuần NumPy/SciPy cho 5 thuật toán hồi quy học máy và 4 mô hình cơ sở đối chuẩn; tối ưu siêu tham số trên tập Validation 10% độc lập theo thời gian và giải quyết triệt để bài toán phân phối lạm phát số không (Zero-Inflation).  
> **Hàm điều phối:** `step_6_train_models()`, `step_7_hyperparameter_tuning()` trong [`main.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/main.py).  
> **Mã nguồn cốt lõi:** [`src/models/regressors.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/regressors.py), [`src/models/tree.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tree.py), [`src/models/ensemble.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/ensemble.py), [`src/models/linear.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/linear.py), [`src/models/hurdle.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/hurdle.py), [`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tuning.py).

---

### Bước 1. Thiết Lập Hệ Thống 4 Mô Hình Cơ Sở Đối Chuẩn (Baselines)
- **Kỷ luật Baseline-First (AI/ML Rule 10):** Tuyệt đối không chấp nhận một thuật toán học máy phức tạp nếu nó không chứng minh được sự vượt trội rõ ràng trước các phỏng đoán thống kê đơn giản.
- Triển khai 4 mô hình đối chuẩn trong [`src/models/baselines.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/baselines.py):
  1. `ZeroRegressorFromScratch`: Luôn dự báo doanh thu bằng $0.0\text{ BRL}$ cho mọi sản phẩm. Trên tập dữ liệu có hơn 90% số tuần bằng 0, Zero Baseline là mốc chặn dưới tối ưu hóa chỉ số MAE trong lớp các dự báo hằng số.
  2. `RevenueMeanRegressorFromScratch`: Luôn dự báo bằng giá trị trung bình cộng doanh số thực tế của tập huấn luyện:

$$
\hat{y} = \log(1 + \bar{s}_{\mathrm{train}}), \quad \text{với } \bar{s}_{\mathrm{train}} = \frac{1}{N} \sum_{k=1}^N s_k
$$

  3. `MeanRegressorFromScratch`: Dự báo bằng trung bình cộng của nhãn log trong không gian tối ưu $\bar{y}_{\mathrm{train}}$, đóng vai trò là mốc đối chuẩn tối ưu nghiệm MSE hằng số ($R^2 = 0$).
  4. `PreviousWeekRegressorFromScratch`: Mô hình quán tính chuỗi thời gian (Persistence Baseline), lấy nguyên doanh thu tuần trước ($t-1$) từ cột `last_week_sales` làm dự báo cho tuần đích ($t$).

---

### Bước 2. Huấn Luyện Hồi Quy Tuyến Tính Nghiệm Giải Tích & Kẹp Biên Phòng Vệ
- Triển khai lớp `LinearRegressionFromScratch` trong [`src/models/linear.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/linear.py).
- **Bộ giải Phương trình Chuẩn tắc (Normal Equation with Ridge Regularization):**
  - Giả định quan hệ tuyến tính giữa ma trận mở rộng $\tilde{X} = [\mathbf{1}, X] \in \mathbb{R}^{N \times (D+1)}$ và vector nhãn $y \in \mathbb{R}^N$.
  - Nghiệm giải tích đóng được xác định qua giả nghịch đảo Moore-Penrose và tích vô hướng ma trận thưa:

$$
\tilde{w}^* = \left(\tilde{X}^\top \tilde{X} + \alpha_{\mathrm{ridge}} I_0\right)^{-1} \tilde{X}^\top y
$$

  - Hệ số điều chuẩn nhỏ $\alpha_{\mathrm{ridge}} = 10^{-5}$ cùng ma trận đơn vị $I_0$ (không phạt bias $b$) giúp triệt tiêu hoàn toàn nguy cơ ma trận Gram bị suy biến hoặc có định thức gần bằng 0 khi xuất hiện đa cộng tuyến giữa 73 danh mục.
- **Cơ chế Kẹp Biên Thích Ứng (Adaptive Clamping):**
  - *Vấn đề bùng nổ số học:* Do hàm hồi quy học trên không gian log, sai số dự báo dương nhỏ cũng có thể bị khuếch đại thành con số doanh thu vô lý (lên tới $1,30 \times 10^8\text{ BRL}$) khi nghịch đảo qua hàm mũ $\exp(w^T x) - 1$.
  - *Giải pháp:* Tự động kẹp biên dự báo log trong khoảng giới hạn an toàn:

$$
\hat{y}_{\mathrm{clamped}} \in [y_{\min} - 0.5 \cdot \mathrm{span}, \; y_{\max} + 0.5 \cdot \mathrm{span}]
$$

  - Thời gian huấn luyện siêu tốc: **0,02 giây**.

---

### Bước 3. Huấn Luyện Cây Quyết Định Hồi Quy (Decision Tree Regressor)
- Triển khai lớp `DecisionTreeRegressorFromScratch` trong [`src/models/tree.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tree.py).
- **Hàm mục tiêu phân nhánh:** Tối đa hóa mức giảm phương sai (Variance Reduction) tại mỗi điểm chia nhị phân:

$$
\Delta \mathrm{Var} = \mathrm{Var}(D) - \left(\frac{|D_L|}{|D|}\mathrm{Var}(D_L) + \frac{|D_R|}{|D|}\mathrm{Var}(D_R)\right)
$$

- **Chiến lược tối ưu hóa 16 Quantile Split Candidates:**
  - *Thách thức hiệu năng:* Quét toàn bộ các giá trị thực trên 655.066 dòng của 89 đặc trưng sẽ đòi hỏi hàng chục triệu phép tính phương sai, khiến việc huấn luyện cây From-Scratch bị đình trệ.
  - *Giải pháp thuật toán:* Thay vì quét mọi điểm, thuật toán chỉ lấy 16 điểm phân vị mẫu (Quantiles: 6.25%, 12.5%, ..., 93.75%) trên mỗi biến làm ngưỡng chia ứng viên. Thuật toán giảm độ phức tạp tính toán hơn 95%, hoàn thành khớp cây sâu 8 tầng trên 655k mẫu chỉ trong **15,22 giây** mà vẫn bảo toàn độ chính xác phân vùng.

---

### Bước 4. Huấn Luyện Các Mô Hình Tổ Hợp (Random Forest & Gradient Boosting)
- Triển khai trong [`src/models/ensemble.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/ensemble.py) giải quyết bài toán cân bằng Bias-Variance:
- **Random Forest (Ensemble Bagging - Giảm Phương Sai):**
  - Huấn luyện song song $B$ cây quyết định độc lập trên các tập dữ liệu con được tạo ra bằng phương pháp lấy mẫu Bootstrap có hoàn lại.
  - Tại mỗi nút phân nhánh, chỉ xét ngẫu nhiên $k = \lfloor\sqrt{89}\rfloor = 9$ đặc trưng để triệt tiêu độ tương quan giữa các cây con.
  - Dự báo bằng trung bình cộng số học của $B$ cây con, giúp mô hình ổn định và chống quá khớp (Overfitting).
- **Gradient Boosting (Ensemble Boosting - Giảm Độ Lệch):**
  - Khởi tạo giá trị cơ sở bằng trung bình mẫu: $F_0(x) = \bar{y}$.
  - Huấn luyện tuần tự $M$ cây hồi quy nông trên phần dư âm (Negative Pseudo-Residuals) của bước trước theo hàm mất mát MSE:

$$
r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}} = y_i - F_{m-1}(x_i)
$$

  - Cập nhật mô hình tổng hợp có kèm hệ số co hẹp tốc độ học (Shrinkage Learning Rate) $\eta = 0.05$ để kiểm soát hội tụ ổn định:

$$
F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)
$$

---

### Bước 5. Huấn Luyện Mô Hình Hai Giai Đoạn (Two-Stage Hurdle Regressor)
- Triển khai trong [`src/models/hurdle.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/hurdle.py) nhằm khắc phục hạn chế cố hữu của các mô hình hồi quy tiêu chuẩn trên dữ liệu thưa:
- **Nguyên lý phân rã xác suất độc lập:**

$$
\mathbb{E}[S \mid x] = P(S > 0 \mid x) \times \mathbb{E}[S \mid S > 0, x]
$$

  - **Giai đoạn 1 (Occurrence Stage):** Mô hình phân loại nhị phân ước lượng xác suất sản phẩm có phát sinh đơn hàng hay không: $\hat{p}(x) = P(s > 0 \mid x) \in [0, 1]$.
  - **Giai đoạn 2 (Severity Stage):** Mô hình hồi quy được huấn luyện chuyên biệt duy nhất trên tập con các bản ghi có doanh thu dương ($s > 0$) để dự báo mức doanh thu kỳ vọng có điều kiện: $\hat{s}_{>0}(x)$.
- **Dự báo hợp nhất:**

$$
\hat{s}_{\mathrm{final}} = \hat{p}(x) \times \hat{s}_{>0}(x)
$$
- **Hiệu quả thực nghiệm:** Đạt mức sai số tốt nhất trong các mô hình chưa tinh chỉnh (**Validation RMSE = 50,84 BRL**), chứng minh tính ưu việt của việc mô hình hóa chuyên biệt cho phân phối Zero-Inflated.

---

### Bước 6. Tinh Chỉnh Siêu Tham Số Trên Tập Validation 10% Độc Lập
- Triển khai hàm `grid_search_from_scratch()` trong [`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tuning.py).
- **Chiến lược Holdout Validation theo thời gian:**
  - Không sử dụng K-Fold CV nhằm tránh rò rỉ dữ liệu giữa các chu kỳ tuần.
  - Toàn bộ các cấu hình siêu tham số được đánh giá độc lập trên 249.399 dòng của tập Validation theo chỉ số **Validation RMSE (BRL)** thực tế sau khi đã nghịch đảo hàm log:
    - *Decision Tree:* `max_depth` $\in [3, 5, 8]$, `min_samples_leaf` $\in [10, 30]$.
    - *Random Forest:* `n_estimators` $\in [10, 20]$, `max_depth` $\in [5, 8]$, `min_samples_split` $\in [10, 20]$.
    - *Gradient Boosting:* `n_estimators` $\in [15, 30]$, `max_depth` $\in [2, 3]$, `learning_rate` $= 0.1$.
- Lưu vết toàn bộ lịch sử dò tìm ra tệp `results/metrics/tuning_history.csv` phục vụ việc truy xuất nguồn gốc và tái lập thực nghiệm.
