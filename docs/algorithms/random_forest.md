# 🌲 Rừng Ngẫu Nhiên Hồi Quy (Random Forest Regressor From-Scratch)

> **Vị trí trong dự án:** Phương pháp học kết hợp (Ensemble Bagging) đa mô hình cây quyết định nhằm giảm phương sai dự báo và tăng cường tính ổn định.
> **Mã nguồn triển khai:** Class [`RandomForestRegressorFromScratch`](../../src/models/ensemble.py) trong [`src/models/ensemble.py`](../../src/models/ensemble.py), được re-export qua Facade [`src/models/regressors.py`](../../src/models/regressors.py).
> **Module tinh chỉnh:** Hàm [`tune_random_forest`](../../src/models/tuning.py) trong [`src/models/tuning.py`](../../src/models/tuning.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Một cây quyết định đơn lẻ (Decision Tree) dù có khả năng nắm bắt quan hệ phi tuyến rất tốt nhưng thường có phương sai cao (High Variance) và nhạy cảm với các biến động ngẫu nhiên trong dữ liệu mẫu.

Mục đích của thuật toán **Random Forest Regressor** trong dự án:
- Ứng dụng kỹ thuật **Ensemble Bagging (Bootstrap Aggregating)** kết hợp lấy mẫu thuộc tính ngẫu nhiên (Feature Subsampling) để xây dựng một tập hợp gồm $B$ cây hồi quy độc lập.
- Hướng tới giảm variance bằng trung bình dự báo các cây; hiệu quả phụ thuộc tương quan giữa cây, không bảo đảm bias thấp hoặc loại bỏ overfitting.
- Tăng tính vững chắc (Robustness) khi dự báo doanh thu cho hàng chục nghìn mã sản phẩm có độ phân tán cao.

### 1.2. Cơ sở toán học & Nguyên lý Ensemble Bagging
Thuật toán hoạt động dựa trên hai nguyên lý cốt lõi:

#### A. Lấy mẫu Bootstrap có hoàn lại (Bootstrap Sampling)
Từ tập huấn luyện gốc $D$ có $N$ mẫu, với mỗi cây $b \in \{1, 2, \dots, B\}$, thuật toán rút ngẫu nhiên có hoàn lại $N$ mẫu để tạo ra tập con $D_b$. Về mặt lý thuyết, xác suất một mẫu được chọn vào $D_b$ xấp xỉ:

$$
P = 1 - \left(1 - \frac{1}{N}\right)^N \approx 1 - \frac{1}{e} \approx 63,2\%
$$

Khoảng $36,8\%$ số mẫu còn lại không được chọn tạo thành tập ngoài mẫu (Out-Of-Bag - OOB).

Đây là tính chất lý thuyết khi N lớn; implementation không tính OOB score. Bootstrap theo dòng không loại phụ thuộc giữa các tuần/sản phẩm và không thay thế temporal evaluation. Với `bootstrap=False`, mỗi cây dùng toàn Train. Các cây fit riêng nhưng dự báo vẫn tương quan; bagging không bảo đảm loại bỏ overfitting hoặc giữ bias thấp.

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

Công thức trên giả định variance các cây bằng nhau và correlation mọi cặp bằng nhau. Mức giảm correlation do feature subsampling chưa được đo trong snapshot.

#### C. Dự đoán trung bình cộng
Kết quả dự báo cuối cùng trong không gian log là trung bình cộng của $B$ cây con:

$$
\hat{y}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
$$

---

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
API nhận sparse matrix nhưng densify thành `float64`; số 89 thuộc snapshot. Caller phải bảo đảm Train không rỗng, dữ liệu hữu hạn, X/y cùng số dòng và đúng thứ tự feature; lớp chưa kiểm tra đầy đủ. Cần dự trù RAM cả bản sao bootstrap và ma trận prediction `(B, N)`.

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
- **Hợp đồng thời gian:** Lag/rolling dùng tuần quá khứ và preprocessing fit Train. Trạng thái `delivered` cuối kỳ và metadata snapshot chưa được chứng minh khả dụng tại origin; temporal split không tự khắc phục giới hạn này. 80/10/10 là tỷ lệ số tuần, không phải số dòng.

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

Implementation fit các cây tuần tự. Seed cố định cho phép tái lập khi giữ dữ liệu, thứ tự dòng và môi trường; seed riêng không làm dự báo độc lập thống kê. Cần `n_estimators >= 1`; constructor chưa kiểm tra đầy đủ tham số. `predict` trước fit gây `RuntimeError`. Điều kiện dừng và ngưỡng chia kế thừa [Decision Tree](decision_tree.md).

Importance thực tế lấy trung bình importance đã chuẩn hóa của các cây rồi chuẩn hóa lại tổng nếu dương; toàn cây không split thì vector 0. Công thức trung bình bên dưới là bước trung gian khi có cây importance bằng 0. Importance không phải tác động nhân quả.

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
Grid dùng một temporal holdout, không phải nhiều fold CV. Bảng Test mục 6.2 thuộc default RF, không phải tuned RF. Trung bình prediction log rồi `expm1` khác trung bình prediction BRL của các cây và không mặc nhiên là kỳ vọng số học.

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
| **Random Forest Regressor** | **2.725,33** | **4,43** | **52,20** | **+0,0047** | **1,04** | **42,69** | **42,11s** | **Hạng 5 / 10** |

Đối chiếu với các mô hình và mốc cơ sở:
- **Đánh bại toàn bộ 4 Baselines:** Tốt hơn Revenue Mean Baseline (RMSE 52,46 BRL), Log Mean Baseline (RMSE 52,48 BRL), Zero Baseline (RMSE 52,50 BRL) và Previous Week Baseline (RMSE 75,39 BRL).
- **Đứng sau các cấu trúc cây khác:** Kém hơn Two-Stage Hurdle (RMSE 49,23 BRL, $R^2 = +0,1148$), Tuned Decision Tree (RMSE 50,41 BRL, $R^2 = +0,0719$), Decision Tree Default (RMSE 51,14 BRL) và Gradient Boosting (RMSE 52,04 BRL).

### 6.3. Nhận xét kỹ thuật & Phân tích nguyên nhân
1. **Kết quả trên snapshot:** Random Forest có R² = 0,0047 và RMSE thấp hơn bốn baseline; chênh lệch với Zero nhỏ và chưa có khoảng tin cậy. MAE 4,43 BRL vẫn cao hơn Zero 4,28 BRL. R² dùng trung bình Test làm tham chiếu, không chứng minh lợi ích kinh doanh ổn định.
2. **Tại sao Random Forest lại có sai số cao hơn Decision Tree trên tập dữ liệu này?**
   - **Giả thuyết feature subsampling:** Chọn 9 feature có thể bỏ qua biến hữu ích tại một số nút. Cần ablation `max_features` với các tham số còn lại cố định để kiểm chứng.
   - **Giả thuyết smoothing:** Trung bình dự báo log có thể giảm đỉnh prediction; chưa có đối chứng tách tác động bootstrap, depth và log-transform để quy chắc chắn nguyên nhân.
3. **Phạm vi metrics:** WAPE là tỷ lệ (1 tương ứng 100%). Supply Chain Cost là điểm phạt giả định mỗi mẫu sản phẩm–tuần, không phải chi phí tồn kho thực đo. Xem [kiểm chứng thực nghiệm](experimental_validation.md) để biết các đối chứng đã có.
