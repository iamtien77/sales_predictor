# 📉 Hồi Quy Tuyến Tính (Linear Regression From-Scratch)

> **Vị trí trong dự án:** Mô hình hồi quy tuyến tính tham chiếu cơ sở (Parametric Baseline), dùng để đối chiếu năng lực giải thích quan hệ tuyến tính trước khi áp dụng các mô hình phi tuyến phức tạp.
> **Mã nguồn triển khai:** Class [`LinearRegressionFromScratch`](../../src/models/linear.py) trong [`src/models/linear.py`](../../src/models/linear.py), được re-export qua Facade [`src/models/regressors.py`](../../src/models/regressors.py).
> **Kiểm thử đơn vị:** Tệp [`tests/test_hurdle_and_metrics.py`](../../tests/test_hurdle_and_metrics.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Theo nguyên tắc chuẩn mực trong phát triển hệ thống Học máy (AI/ML Rule 10): **"Luôn đề xuất mô hình cơ sở đơn giản trước khi xây dựng các mô hình phức tạp"**.

Mục đích của thuật toán **Linear Regression** trong dự án:
- Thiết lập đường cơ sở tham số (Parametric Baseline) nhanh nhất để đo lường mức độ tương quan tuyến tính giữa 89 đặc trưng đầu vào với biến mục tiêu doanh thu.
- Kiểm tra giả định tuyến tính (Linearity Assumption): Liệu tổng doanh thu tuần có thể được ước lượng thỏa đáng bằng một tổ hợp tuyến tính có trọng số của giá, các tuần trễ và danh mục hàng hóa hay không?
- Đánh giá khả năng mở rộng và tốc độ tính toán nghiệm ma trận giải tích đóng (*Analytical Closed-form Solution*) trên tập dữ liệu lớn (>655.000 dòng ma trận thưa).

### 1.2. Cơ sở toán học & Các bộ giải (Solvers)
Mô hình giả định mối quan hệ giữa vector đặc trưng $x \in \mathbb{R}^D$ ($D = 89$) và biến mục tiêu log doanh thu $y \in \mathbb{R}$ có dạng tuyến tính:

$$
\hat{y} = w^\top x + b
$$

Để tính toán đồng thời cả trọng số $w$ và hệ số chặn (bias) $b$, ma trận dữ liệu được mở rộng thêm một cột toàn số 1: $\tilde{X} = [\mathbf{1}, X] \in \mathbb{R}^{N \times (D+1)}$ và vector tham số $\tilde{w} = [b, w_1, \dots, w_D]^\top$.

Lớp `LinearRegressionFromScratch` hỗ trợ hai bộ giải:

#### A. Bộ giải Phương trình Chuẩn tắc (Normal Equation OLS - Mặc định)
Cực tiểu hóa hàm mất mát bình phương tối thiểu (Ordinary Least Squares - OLS) có bổ sung điều chuẩn Ridge nhỏ ($\alpha_{\text{ridge}} = 10^{-5}$) nhằm chống suy biến ma trận:

$$
J(\tilde{w}) = \frac{1}{N} \left(\|\tilde{X} \tilde{w} - y\|_2^2 + \alpha_{\text{ridge}} \|w\|_2^2\right)
$$

Nghiệm giải tích đóng được xác định bằng giải thuật giả nghịch đảo Moore-Penrose (`np.linalg.pinv`):

$$
\tilde{w}^* = \left( \tilde{X}^\top \tilde{X} + \alpha_{\text{ridge}} I_0 \right)^{+} \tilde{X}^\top y
$$

Trong đó $I_0$ là ma trận đơn vị có phần tử đầu tiên bằng 0 (không phạt hệ số chặn $b$). Mô hình hỗ trợ tính toán trực tiếp ma trận Gram $\tilde{X}^\top \tilde{X}$ qua toán tử `@` của `scipy.sparse.csr_matrix`.

#### B. Bộ giải Hạ độ dốc (Gradient Descent)
Khởi tạo từng trọng số với phân phối chuẩn có trung bình 0, độ lệch chuẩn 0,01 ($w_j^{(0)} \sim \mathcal{N}(0, 0.01^2)$) và cập nhật lặp qua $T$ vòng:

$$
\nabla_{\tilde{w}} J = \frac{2}{N} \tilde{X}^\top (\tilde{X} \tilde{w} - y) + \frac{2\alpha_{\text{ridge}}}{N} \begin{bmatrix} 0 \\ w \end{bmatrix}
$$

$$
\tilde{w}^{(t+1)} = \tilde{w}^{(t)} - \eta \cdot \nabla_{\tilde{w}} J
$$

---

Hai solver dùng cùng quy ước `alpha_ridge`, không phạt intercept. Khi Gradient Descent hội tụ, nghiệm phải khớp Normal Equation trong sai số số học. `cost_history_` chỉ lưu MSE, không gồm phần phạt Ridge. Lần chạy full ngày 27/10/2026 đã cập nhật artifact và metrics. CLI dùng Normal Equation; kết quả này không đo tác động của bản sửa Gradient Descent.

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
- **Ma trận đặc trưng $X$:** Mảng NumPy 2D hoặc ma trận thưa `scipy.sparse.csr_matrix` kích thước $(N, 89)$.
- **Vector nhãn mục tiêu $y$:** Vector NumPy 1D kích thước $(N,)$ trong không gian log tự nhiên: $y = \log(1 + \text{sales})$.

### 2.2. Định dạng dữ liệu đầu ra (Output)
- **Phương thức `predict(X)`:** Trả về vector dự báo $\hat{y}$ trong không gian log tự nhiên (có áp dụng cơ chế Kẹp biên Thích ứng).
- **Hàm chuyển đổi tiền tệ thực tế BRL:**

$$
  \hat{y}_{\text{sales}} = \max\left(0.0, \exp(\hat{y}) - 1\right) \text{ BRL}
$$

---

## 3. Data used (Tập dữ liệu sử dụng & Đặc thù phân phối)

- **Chiến lược phân chia:** Phân chia chuỗi thời gian tuần lịch (Temporal Split 80/10/10) trên 1.198.182 mẫu:
  - **Tập Train (80% số tuần):** 655.066 dòng (chiếm các tuần đầu lịch sử).
  - **Tập Validation (10% số tuần):** 249.399 dòng.
  - **Tập Test (10% số tuần):** 293.717 dòng.
- **Hệ thống 89 đặc trưng:** 16 biến số học (đã qua `CustomStandardScaler`) và 73 biến danh mục (đã qua `CustomOneHotEncoder`), được fit preprocessing trên Train; điều này chưa xử lý hạn chế snapshot delivered hồi cứu.

---

## 4. Evaluation (Phương pháp đánh giá)

Protocol CLI chọn bằng Validation RMSE BRL trước khi báo cáo Test. Tỷ lệ split tính theo tuần; lag trong Test dùng lịch sử trước từng tuần đích, không phải dự báo nhiều tuần từ một origin cố định. Test đã được xem trong các vòng phân tích nên không coi là holdout mù mới. Số features thuộc snapshot, có thể thay đổi khi fit encoder trên dữ liệu khác.

- **Hàm mất mát huấn luyện:** MSE trên không gian log tự nhiên.
- **Bộ 6 chỉ số đo lường trên Test (Thang đo tiền tệ BRL):**
  1. Mean Squared Error (MSE): $\text{BRL}^2$.
  2. Mean Absolute Error (MAE): $\text{BRL}$.
  3. Root Mean Squared Error (RMSE): $\text{BRL}$.
  4. Hệ số xác định ($R^2$ Score).
  5. Weighted Absolute Percentage Error (WAPE).
  6. Supply Chain Cost Per Item (BRL).

---

## 5. Status (Trạng thái triển khai & Tối ưu kỹ thuật)

- **Trạng thái:** Hoàn thiện 100% From-Scratch thuần NumPy, không dùng `sklearn.linear_model.LinearRegression`.
- **Đóng gói Deep Modules:** Tách riêng thành module độc lập tại [`src/models/linear.py`](../../src/models/linear.py).
- **Tối ưu hóa ma trận thưa (Sparse Support):**
  - Dùng `sp.hstack` và `(X_b.T @ X_b)` để tính Gram $(90,90)$ trong snapshot 89 features. Gram được chuyển thành dense để giải; CSV chỉ ghi tổng fit 0,62 giây, chưa có benchmark riêng phép nhân Gram.
  - Dùng giả nghịch đảo (ký hiệu $+$) để xử lý hệ suy biến. Việc tạo Gram có thể làm xấu condition number; Ridge/pinv không bảo đảm mọi bộ dữ liệu đều ổn định số học.
- **Tích hợp Kẹp biên Thích ứng (Adaptive Numerical Clamping):** `clip_predictions=True` giới hạn dự báo log theo miền nhãn Train. Đây là guard số học, không bảo đảm dự báo BRL chính xác hoặc xử lý mọi đầu vào NaN/Inf.

---

## 6. Results & Mổ Xẻ Hiện Tượng Thất Bại (Failure Mode & Mitigation Analysis)

### 6.1. Kết quả Thực nghiệm Đối chứng trên Test Set (293.717 dòng)

| Cấu hình mô hình | MSE ($\text{BRL}^2$) | MAE (BRL) | RMSE (BRL) | $R^2$ Score | WAPE | Supply Chain Cost (BRL) | Fit Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression (Unbounded OLS — ghi chú lịch sử)** | $1,69 \times 10^{16}$ | $246.299,23$ | $1,30 \times 10^8$ | $-6,17 \times 10^{12}$ | — | — | 0,53s |
| **Linear Regression (Adaptive Clamping)** | **$3,04 \times 10^7$** | **44,38** | **5.511,67** | **$-11.093,46$** | **10,37** | **120,70** | **0,62s** |

### 6.2. Phân tích Bản chất Toán học & Nguyên nhân Thất bại ban đầu (Failure Mode)
Trong thử nghiệm ban đầu khi chưa có kẹp biên, Linear Regression bị hiện tượng **nổ số học (*numerical explosion*) khi suy luận ngược về thang tiền tệ BRL**, khiến RMSE lên tới 130 triệu BRL:

1. **Hiệu ứng đòn bẩy hàm mũ khi nghịch đảo Log ($\text{expm1}$):**
   - Mô hình tối ưu hóa hàm mất mát trên không gian log: $\hat{y} = w^\top x + b$.
   - Trong không gian log, siêu phẳng tuyến tính không có biên chặn trên ($\hat{y} \in (-\infty, +\infty)$). Do hiện tượng cộng hưởng của các trọng số âm/dương lớn trên không gian 89 chiều thưa, chỉ cần một vài mẫu kiểm thử có đặc trưng ngoại lai khiến giá trị log dự báo vượt ngưỡng bình thường (ví dụ: $\hat{y} \approx 38$).
   - Khi áp dụng hàm chuyển đổi ngược $\hat{y}_{\text{sales}} = \exp(\hat{y}) - 1$, giá trị bùng nổ theo cấp số nhân:

$$
     \exp(38) \approx 3,18 \times 10^{16} \text{ BRL}
$$

   - Ví dụ minh họa: nếu một dự báo khoảng $3,18\times10^{16}$ BRL có actual nhỏ hơn nhiều, đóng góp MSE trên 293.717 mẫu khoảng $3,44\times10^{27}$, không phải $10^{16}$. Ví dụ này giải thích khuếch đại hàm mũ, không phải phép tái hiện con số lịch sử trong bảng.

2. **Sự không tương đương giữa Log Loss và Money Loss:**
   - Việc tối thiểu hóa MSE trên không gian log không đồng nghĩa với việc tối thiểu hóa RMSE trên thang tiền tệ BRL. Đối với mô hình tham số tuyến tính, việc cân bằng sai số trên log khiến mô hình chấp nhận hy sinh độ chính xác ở vùng biên trên, dẫn đến thảm họa khi nghịch đảo hàm mũ.

3. **Tính phi tuyến tính bản chất của bài toán bán lẻ:**
   - Quan hệ phi tuyến là một giả thuyết cần kiểm chứng. Kết quả kém của cấu hình tuyến tính hiện tại không đủ phân biệt ảnh hưởng của đặc trưng, regularization, ngoại suy và phép nghịch đảo log.

### 6.3. Giải pháp Kỹ thuật Phòng vệ: Cơ chế Adaptive Numerical Clamping
Mã nguồn [src/models/linear.py](../../src/models/linear.py) bổ sung kẹp biên nhằm hạn chế ngoại suy log. Cơ chế này chưa đủ để chứng nhận khả năng sử dụng trong production:
- **Nguyên lý hoạt động:** Trong quá trình `fit(X, y)`, mô hình ghi nhận miền giá trị nhãn thực tế của tập huấn luyện: $y_{\min}$ và $y_{\max}$, với độ trải $\text{span} = \max(1, y_{\max} - y_{\min})$.
- Khi `predict(X)`, mọi giá trị dự báo vượt khỏi biên an toàn sẽ được kẹp lại:

$$
  \hat{y}_{\text{safe}} = \text{clip}\left(\hat{y}, y_{\min} - 0.5 \cdot \text{span}, y_{\max} + 0.5 \cdot \text{span}\right)
$$

- **Kết quả hiện tại:** RMSE **5.511,67 BRL**, vẫn lớn. Số 130 triệu BRL thuộc ghi chú lịch sử, chưa có đối chứng cùng lần chạy để quy chênh lệch cho clamping.

### 6.4. Ý nghĩa Học thuật & Kết luận
- Cấu hình tuyến tính hiện tại thua các mô hình cây theo RMSE trên snapshot này. Chưa có cơ sở loại bỏ mọi mô hình tuyến tính; xem [thí nghiệm đối chứng](experimental_validation.md) về Ridge và clipping.
- Cây hồi quy đơn dự báo trung bình nhãn trong lá; Random Forest lấy trung bình các cây nên giữ giới hạn nhãn Train hữu hạn. Gradient Boosting cộng các cây residual, không có cùng bảo đảm nằm trong miền nhãn gốc. Không suy rộng thành cam kết không bao giờ overflow.
- Đóng góp bài học thực tiễn giá trị về **Nguyên lý kỹ thuật phòng vệ (Defensive Machine Learning)**: khi đưa mô hình vào môi trường thực tế, việc thiết lập các đai an toàn toán học (clamping/guardrails) là điều kiện bắt buộc để bảo vệ tính toàn vẹn của hệ thống.
