# ⚡ Tăng Cường Độ Dốc Hồi Quy (Gradient Boosting Regressor From-Scratch)

> **Vị trí trong dự án:** Phương pháp học kết hợp tuần tự (Ensemble Boosting) tiên tiến, huấn luyện các cây quyết định nông trên phần dư âm để tối ưu hóa hàm mất mát bình phương.
> **Mã nguồn triển khai:** Class [`GradientBoostingRegressorFromScratch`](../../src/models/ensemble.py) trong [`src/models/ensemble.py`](../../src/models/ensemble.py), được re-export qua Facade [`src/models/regressors.py`](../../src/models/regressors.py).
> **Module tinh chỉnh:** Hàm [`tune_gradient_boosting`](../../src/models/tuning.py) trong [`src/models/tuning.py`](../../src/models/tuning.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Bagging cho phép fit các cây riêng (implementation Random Forest hiện tại chạy tuần tự). **Gradient Boosting** phải học tuần tự vì mỗi cây dùng residual của tổng các cây trước; mục tiêu là cải thiện mức khớp hàm, không bảo đảm loại bỏ bias hay overfitting.

Mục đích của thuật toán Gradient Boosting trong dự án:
- Học xấp xỉ hàm phi tuyến tính phức tạp bằng chuỗi cộng gộp các cây quyết định nông (*weak learners* / *shallow trees*).
- Bù trừ sai số bằng cách fit xấp xỉ cây con vào phần dư (Residuals) của thế hệ trước.
- Ứng dụng hệ số học co hẹp (*Learning Rate Shrinkage*) để kiểm soát tốc độ hội tụ, giảm thiểu nguy cơ quá khớp trên dữ liệu chuỗi thời gian bán lẻ có độ nhiễu cao.

### 1.2. Cơ sở toán học & Thuật toán huấn luyện tuần tự

#### A. Hàm mất mát tối ưu (Loss Function)
Mô hình hoạt động trong không gian log tự nhiên với hàm mất mát bình phương (Squared Error Loss):

$$
L(y, F(x)) = \frac{1}{2} \left( y - F(x) \right)^2
$$

Đạo hàm riêng (Gradient) âm của hàm mất mát theo dự báo hiện tại chính là phần dư (Residual):

$$
r_i = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}} = y_i - F_{m-1}(x_i)
$$

#### B. Quy trình thuật toán tuần tự $M$ vòng lặp
1. **Khởi tạo giá trị dự báo cơ sở $F_0(x)$:** Giá trị tối ưu là trung bình cộng của nhãn huấn luyện:

   $$
   F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, \gamma) = \frac{1}{N} \sum_{i=1}^{N} y_i = \bar{y}_{\text{train}}
   $$

2. **Vòng lặp $m = 1, 2, \dots, M$ (với $M = \text{n\_estimators}$):**
   - **Bước 2.1 (Tính phần dư):** Với mỗi mẫu $i \in \{1, \dots, N\}$, tính phần dư bằng negative gradient:

     $$
     r_{im} = y_i - F_{m-1}(x_i)
     $$

     Lưu lại giá trị hàm mất mát trung bình: $\text{Loss}_m = \frac{1}{N} \sum_{i=1}^{N} r_{im}^2$.

   - **Bước 2.2 (Huấn luyện cây con trên phần dư):** Khớp một cây quyết định hồi quy nông $h_m(x)$ (thường chọn `max_depth=3`) trên tập dữ liệu $\{(x_i, r_{im})\}_{i=1}^N$.

   - **Bước 2.3 (Cập nhật mô hình tích lũy với hệ số co hẹp $\eta$):**

     $$
     F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)
     $$

     Trong đó $\eta \in (0, 1]$ là `learning_rate` (hệ số co hẹp), giúp kiểm soát đóng góp của từng cây con và chống phân kỳ.

3. **Dự báo cuối cùng:**

   $$
   \hat{y}(x) = F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)
   $$

---

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
API nhận sparse matrix nhưng densify thành `float64`; số 89 thuộc snapshot, không cố định trong thuật toán. Caller phải bảo đảm Train không rỗng, dữ liệu hữu hạn, X/y cùng số dòng và đúng thứ tự feature; lớp chưa kiểm tra đầy đủ. Cần dự trù RAM cho dense matrix và các bản sao khi chia nút.

- **Ma trận đặc trưng $X$:** Mảng dense NumPy 2D kích thước $(N, 89)$ chứa 16 đặc trưng số học đã chuẩn hóa Z-Score và 73 cột danh mục sản phẩm đã mã hóa One-Hot.
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
- **Bảo toàn tuần doanh thu bằng 0:** Giữ nguyên các tuần `sales = 0.0` BRL để phản ánh đúng cấu trúc dữ liệu thương mại điện tử thực tế.
- **Hợp đồng thời gian:** Lag/rolling dùng tuần quá khứ và preprocessing fit Train. Trạng thái `delivered` cuối kỳ và metadata snapshot chưa được chứng minh khả dụng tại origin; temporal split không tự khắc phục giới hạn này. 80/10/10 là tỷ lệ số tuần, không phải số dòng.

---

## 4. Evaluation (Phương pháp đánh giá)

- **Hàm mất mát tối ưu:** L2 Loss (Mean Squared Error) trên không gian log tự nhiên: $\mathcal{L} = \frac{1}{N} \sum (y_i - \hat{y}_i)^2$.
- **Lựa chọn siêu tham số:** Đánh giá trên tập Validation độc lập bằng thang đo RMSE tiền tệ BRL.
- **Bộ 4 chỉ số đo lường độc lập trên Test (Thang đo BRL):**
  1. Mean Squared Error (MSE): $\text{BRL}^2$.
  2. Mean Absolute Error (MAE): $\text{BRL}$.
  3. Root Mean Squared Error (RMSE): $\text{BRL}$.
  4. Hệ số xác định ($R^2$ Score).

---

## 5. Status (Trạng thái triển khai & Tối ưu kỹ thuật)

- **Trạng thái:** Hoàn thiện 100% From-Scratch thuần NumPy, không sử dụng `scikit-learn` hay các thư viện ngoài như XGBoost/LightGBM.
- **Kiến trúc cây nông (`max_depth=3`):** Hạn chế độ sâu của từng cây con để đảm bảo mỗi cây chỉ đóng vai trò là một "người học yếu" (*weak learner*), tránh việc cây đầu tiên học quá chi tiết dẫn đến hiện tượng quá khớp sớm.
- **Ghi nhận Loss (`train_loss_history_`):** Lưu MSE trước update mỗi vòng, ứng với F₀ đến Fₘ₋₁; không chứa loss sau cây cuối. Không có early stopping.
- **Độ quan trọng đặc trưng tổng hợp:**

  $$
  \text{Importance}(j) = \frac{1}{M} \sum_{m=1}^{M} \text{Importance}_{h_m}(j)
  $$

---

## 6. Results (Kết quả thực nghiệm định lượng)

Implementation lấy trung bình importance đã chuẩn hóa của các cây rồi chuẩn hóa tổng lại nếu dương (toàn cây không split thì vector 0). Đây không phải tổng raw gain theo mức giảm loss mỗi vòng hoặc tác động nhân quả.

Lá cây fit trung bình residual cho squared error nên không có line search riêng. Code chạy đủ số cây; cần `n_estimators >= 1` và learning rate hữu hạn phù hợp (quy ước shrinkage ở đây là `0 < learning_rate <= 1`), nhưng constructor chưa kiểm tra đầy đủ. `predict` trước fit gây `RuntimeError`. Tổng các cây residual không nhất thiết nằm trong miền nhãn Train; không suy ra bảo đảm chống overflow từ giới hạn của từng lá.

### 6.1. Quá trình Tinh chỉnh Siêu tham số (Grid Search trên Validation)
Mô hình Gradient Boosting được khảo sát 4 cấu hình lưới trên tập Validation (249.399 dòng):

| Cấu hình | `n_estimators` | `max_depth` | `learning_rate` | Validation RMSE (BRL) |
|:---:|:---:|:---:|:---:|:---:|
| GB 1 | 15 | 2 | 0.1 | 63,1879 |
| GB 2 | 15 | 3 | 0.1 | 62,8054 |
| GB 3 | 30 | 2 | 0.1 | 62,4007 |
| **GB 4 (Tuned GB)** | **30** | **3** | **0.1** | **61,2637** |

Cấu hình tốt nhất của Gradient Boosting trên Validation đạt RMSE **61,2637 BRL** (với `n_estimators=30, max_depth=3, learning_rate=0.1`), thấp hơn tuned Random Forest (61,9221 BRL) nhưng cao hơn ứng viên Decision Tree depth=5/leaf=10 (61,1664 BRL). Grid dùng một temporal holdout, không phải nhiều fold CV. Bảng Test bên dưới thuộc default GB với learning_rate=0.05, không phải tuned GB.

### 6.2. Kết quả Đánh giá Độc lập trên Tập Kiểm thử Test (293.717 dòng)
Cấu hình mặc định tham chiếu (`n_estimators=30, learning_rate=0.05, max_depth=3, min_samples_split=10, min_samples_leaf=5`) đạt kết quả:

| Mô hình | MSE ($\text{BRL}^2$) | MAE (BRL) | RMSE (BRL) | $R^2$ Score | WAPE | Supply Chain Cost (BRL) | Fit Time (s) | Xếp hạng toàn hệ thống |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting Regressor** | **2.708,22** | **4,42** | **52,04** | **+0,0109** | **1,03** | **42,49** | **138,50s** | **Hạng 4 / 10** |

So sánh với các mô hình khác:
- **Đánh bại Random Forest:** RMSE 52,04 BRL so với 52,20 BRL; $R^2 = +0,0109$ so với $+0,0047$.
- **Đánh bại toàn bộ Baselines:** Tốt hơn Revenue Mean Baseline (RMSE 52,46 BRL), Log Mean Baseline (RMSE 52,48 BRL) và Zero Baseline (RMSE 52,50 BRL).
- **Đứng sau Decision Tree & Two-Stage Hurdle:** Kém hơn Two-Stage Hurdle (RMSE 49,23 BRL, $R^2 = +0,1148$), Tuned Decision Tree (RMSE 50,41 BRL, $R^2 = +0,0719$) và Decision Tree Default (RMSE 51,14 BRL, $R^2 = +0,0449$).

### 6.3. Nhận xét kỹ thuật & Trade-offs
1. **Chênh lệch quan sát trong snapshot:** RMSE 52,04 BRL thấp hơn Random Forest 52,20 BRL; R² lần lượt 0,0109 và 0,0047. Chưa có khoảng tin cậy cho chênh lệch; tỷ số hai R² nhỏ không chứng minh năng lực vượt trội. MAE 4,42 BRL vẫn cao hơn Zero 4,28 BRL.
2. **Chi phí tính toán:** CSV ghi **138,50 giây**, khoảng 8,37 lần Decision Tree (16,55s) và 3,29 lần Random Forest (42,11s). Đây là các cấu hình khác nhau trong cùng snapshot; chưa tách ảnh hưởng số cây, số feature và depth để quy riêng cho tính tuần tự.
3. **Giả thuyết cần đối chứng:** Depth=3, 30 vòng và learning_rate=0.05 có thể giới hạn mức khớp residual. Cần ablation trên Validation để phân biệt underfitting, log-transform và độ sâu; metrics tổng hợp chưa chứng minh nguyên nhân. MSE log không trực tiếp tối ưu RMSE BRL; nghịch đảo trung bình log không mặc nhiên là kỳ vọng BRL.
4. **Phạm vi metrics:** WAPE là tỷ lệ (1 tương ứng 100%). Supply Chain Cost là điểm phạt giả định mỗi mẫu sản phẩm–tuần, không phải chi phí tồn kho thực đo. Xem [kiểm chứng thực nghiệm](experimental_validation.md) để biết các đối chứng đã có.
