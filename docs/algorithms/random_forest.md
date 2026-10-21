# 🌲 Rừng Ngẫu Nhiên Hồi Quy (Random Forest Regressor From-Scratch)

> **Vị trí trong dự án:** Phương pháp học kết hợp (Ensemble Bagging) đa mô hình cây quyết định nhằm giảm phương sai dự báo và tăng cường tính ổn định.  
> **Mã nguồn triển khai:** Class [`RandomForestRegressorFromScratch`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/ensemble.py#L15) trong [`src/models/ensemble.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/ensemble.py), được re-export qua Facade [`src/models/regressors.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/regressors.py).  
> **Module tinh chỉnh:** Hàm [`tune_random_forest`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tuning.py#L169) trong [`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tuning.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Một cây quyết định đơn lẻ (Decision Tree) dù có khả năng nắm bắt quan hệ phi tuyến rất tốt nhưng thường có phương sai cao (High Variance) và nhạy cảm với các biến động ngẫu nhiên trong dữ liệu mẫu. 

Mục đích của thuật toán **Random Forest Regressor** trong dự án:
- Ứng dụng kỹ thuật **Ensemble Bagging (Bootstrap Aggregating)** kết hợp lấy mẫu thuộc tính ngẫu nhiên (Feature Subsampling) để xây dựng một tập hợp gồm $B$ cây hồi quy độc lập.
- Làm giảm mạnh phương sai của toàn hệ thống thông qua việc lấy trung bình cộng kết quả của các cây thành phần, duy trì độ lệch (Bias) thấp và ngăn chặn hiện tượng quá khớp (Overfitting).
- Tăng tính vững chắc (Robustness) khi dự báo doanh thu cho hàng chục nghìn mã sản phẩm có độ phân tán cao.

### 1.2. Cơ sở toán học & Nguyên lý Ensemble Bagging
Thuật toán hoạt động dựa trên hai nguyên lý cốt lõi:

#### A. Lấy mẫu Bootstrap có hoàn lại (Bootstrap Sampling)
Từ tập huấn luyện gốc $D$ có $N$ mẫu, với mỗi cây $b \in \{1, 2, \dots, B\}$, thuật toán rút ngẫu nhiên có hoàn lại $N$ mẫu để tạo ra tập con $D_b$. Về mặt lý thuyết, xác suất một mẫu được chọn vào $D_b$ xấp xỉ:

$$
P = 1 - \left(1 - \frac{1}{N}\right)^N \approx 1 - \frac{1}{e} \approx 63,2\%
$$

Khoảng $36,8\%$ số mẫu còn lại không được chọn tạo thành tập ngoài mẫu (Out-Of-Bag - OOB).

#### B. Phân nhánh ngẫu nhiên đặc trưng (Feature Subsampling)
Tại mỗi nút phân nhánh của từng cây con, thay vì tìm kiếm trên toàn bộ $d = 89$ đặc trưng, thuật toán chỉ chọn ngẫu nhiên một tập con gồm:

$$
k = \lfloor \sqrt{d} \rfloor = \lfloor \sqrt{89} \rfloor = 9 \text{ đặc trưng}
$$

Điều này làm giảm độ tương quan (Correlation) giữa các cây, giúp phương sai của giá trị trung bình giảm đi đáng kể theo công thức phương sai tổ hợp:

$$
\text{Var}\left(\frac{1}{B} \sum_{b=1}^{B} T_b(x)\right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2
$$

Khi $\rho \to 0$ (các cây càng ít tương quan), phương sai toàn hệ thống càng giảm mạnh khi $B$ tăng lên.

#### C. Dự đoán trung bình cộng
Kết quả dự báo cuối cùng trong không gian log là trung bình cộng của $B$ cây con:

$$
\hat{y}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
$$

---

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
- **Ma trận đặc trưng $X$:** Mảng dense NumPy 2D kích thước $(N, 89)$ biểu diễn 16 đặc trưng số học đã chuẩn hóa và 73 danh mục sản phẩm đã mã hóa One-Hot.
- **Vector nhãn mục tiêu $y$:** Vector NumPy 1D kích thước $(N,)$ trong không gian log tự nhiên: $y = \log(1 + \text{sales})$.

### 2.2. Định dạng dữ liệu đầu ra (Output)
- **Phương thức `predict(X)`:** Trả về vector dự báo liên tục $\hat{y}$ (không gian log).
- **Hàm chuyển đổi tiền tệ thực tế BRL:**

  $$
  \hat{y}_{\text{sales}} = \max\left(0.0, \exp(\hat{y}) - 1\right) = \max\left(0.0, \text{expm1}(\hat{y})\right)
  $$

---

## 3. Data Used (Dữ liệu sử dụng & Hợp đồng chuỗi thời gian)

- **Phân chia Temporal Split 80/10/10:**
  - Tập Huấn luyện (Train Set): **655.066 dòng** sản phẩm–tuần.
  - Tập Kiểm định (Validation Set): **249.399 dòng** sản phẩm–tuần.
  - Tập Kiểm thử Độc lập (Test Set): **293.717 dòng** sản phẩm–tuần.
- **Bảo toàn tuần doanh thu bằng 0:** Giữ nguyên các tuần `sales = 0.0` BRL để phản ánh đúng mật độ nhu cầu thưa.
- **Tuân thủ hợp đồng thời gian:** Toàn bộ 89 đặc trưng đầu vào (biến trễ, thống kê trượt, giá tuần gần nhất, lịch sử đánh giá) đều được xác định trước 00:00 thứ Hai của tuần đích.

---

## 4. Evaluation (Phương pháp đánh giá)

- **Hàm mục tiêu cây con:** Tối đa hóa lượng giảm phương sai (Variance Reduction) tại mỗi nút con.
- **Tiêu chí tối ưu lưới trên Validation:** Root Mean Squared Error đo bằng tiền tệ BRL (Validation RMSE BRL).
- **Bộ 4 chỉ số đo lường độc lập trên Test (Thang đo BRL):**
  1. Mean Squared Error (MSE): $\text{BRL}^2$.
  2. Mean Absolute Error (MAE): $\text{BRL}$.
  3. Root Mean Squared Error (RMSE): $\text{BRL}$.
  4. Hệ số xác định ($R^2$ Score).

---

## 5. Status (Trạng thái triển khai & Tối ưu kỹ thuật)

- **Trạng thái:** Hoàn thiện 100% From-Scratch thuần NumPy, không sử dụng `scikit-learn`.
- **Đóng gói hướng đối tượng mô-đun:** Tái sử dụng lớp `DecisionTreeRegressorFromScratch` làm cây thành phần.
- **Quản lý trạng thái ngẫu nhiên độc lập (`RandomState`):** Mỗi cây con được gán một seed ngẫu nhiên độc lập sinh từ `rng.randint()`, đảm bảo tính tái lập (Reproducibility) và đa dạng hóa quá trình lấy mẫu.
- **Trích xuất Feature Importances tổng hợp:**
  
  $$
  \text{Importance}(j) = \frac{1}{B} \sum_{b=1}^{B} \text{Importance}_b(j)
  $$

---

## 6. Results (Kết quả thực nghiệm định lượng)

### 6.1. Quá trình Tinh chỉnh Siêu tham số (Grid Search trên Validation)
Mô hình Random Forest được khảo sát 8 cấu hình trên tập Validation độc lập (249.399 dòng):

| Cấu hình | `n_estimators` | `max_depth` | `min_samples_split` | Validation RMSE (BRL) |
|:---:|:---:|:---:|:---:|:---:|
| RF 1 | 10 | 5 | 10 | 63,2538 |
| RF 2 | 10 | 5 | 20 | 63,2615 |
| **RF 3 (Tuned RF)** | **10** | **8** | **10** | **61,9221** |
| RF 4 | 10 | 8 | 20 | 62,2742 |
| RF 5 | 20 | 5 | 10 | 63,1885 |
| RF 6 | 20 | 5 | 20 | 63,1986 |
| RF 7 | 20 | 8 | 10 | 62,2110 |
| RF 8 | 20 | 8 | 20 | 62,2910 |

Cấu hình tốt nhất của Random Forest trên tập Validation đạt RMSE **61,9221 BRL** (với `n_estimators=10, max_depth=8, min_samples_split=10`). Tuy nhiên, con số này vẫn cao hơn mức **59,2288 BRL** của Decision Tree đơn lẻ được tối ưu.

### 6.2. Kết quả Đánh giá Độc lập trên Tập Kiểm thử Test (293.717 dòng)
Cấu hình mặc định tham chiếu của Random Forest (`n_estimators=30, max_depth=6, min_samples_split=10, min_samples_leaf=5, max_features='sqrt'`) đạt kết quả:

| Mô hình | MSE ($\text{BRL}^2$) | MAE (BRL) | RMSE (BRL) | $R^2$ Score | WAPE | Supply Chain Cost (BRL) | Fit Time (s) | Xếp hạng toàn hệ thống |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Regressor** | **2.725,33** | **4,43** | **52,20** | **+0,0047** | **1,04** | **42,69** | **94,45s** | **Hạng 5 / 10** |

Đối chiếu với các mô hình và mốc cơ sở:
- **Đánh bại toàn bộ 4 Baselines:** Tốt hơn Revenue Mean Baseline (RMSE 52,46 BRL), Log Mean Baseline (RMSE 52,48 BRL), Zero Baseline (RMSE 52,50 BRL) và Previous Week Baseline (RMSE 75,39 BRL).
- **Đứng sau các cấu trúc cây khác:** Kém hơn Two-Stage Hurdle (RMSE 49,23 BRL, $R^2 = +0,1148$), Tuned Decision Tree (RMSE 50,41 BRL, $R^2 = +0,0719$), Decision Tree Default (RMSE 51,14 BRL) và Gradient Boosting (RMSE 52,04 BRL).

### 6.3. Nhận xét kỹ thuật & Phân tích nguyên nhân
1. **Khả năng khái quát hóa dương ($R^2 > 0$):** Random Forest đạt $R^2 = +0,0047$, chứng minh năng lực dự báo thực sự và đánh bại toàn bộ các mốc cơ sở.
2. **Tại sao Random Forest lại có sai số cao hơn Decision Tree trên tập dữ liệu này?**
   - **Tác động của Feature Subsampling ($k = \sqrt{89} \approx 9$):** Trong bài toán dự báo doanh thu tuần, tín hiệu thông tin tập trung rất mạnh vào một số ít biến then chốt (như `last_week_sales`, `price`, `sales_mean_4_weeks`). Khi giới hạn ngẫu nhiên 9 thuộc tính ở mỗi nút, nhiều nút phân nhánh không tiếp cận được các biến quan trọng này, dẫn đến việc các cây con phân nhánh theo các biến thứ yếu ít tính dự báo.
   - **Bản chất dữ liệu zero-sales chiếm trên 90%:** Việc lấy trung bình cộng dự báo giữa các cây con vô tình làm "mịn" (smoothing) các giá trị dự báo, kéo các dự báo doanh thu dương về gần mức trung bình nhỏ, làm giảm khả năng nhận diện các đỉnh doanh thu thực tế so với cây quyết định đơn lẻ có độ sâu phù hợp.
