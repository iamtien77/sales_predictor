# 🌳 Cây Quyết Định Hồi Quy (Decision Tree Regressor From-Scratch)

> **Vị trí trong dự án:** Thuật toán hồi quy phi tuyến tính cốt lõi và là **Mô hình đơn khối tốt nhất (Best Single Model) - Hạng 2 toàn hệ thống** sau khi tinh chỉnh siêu tham số trên tập Validation độc lập. Đây cũng là khối xây dựng cơ sở (Base Learner) cho Random Forest, Gradient Boosting và Two-Stage Hurdle Regressor.  
> **Mã nguồn triển khai:** Class [`DecisionTreeRegressorFromScratch`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tree.py#L41) và đối tượng nút [`DecisionTreeNode`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tree.py#L15) trong [`src/models/tree.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tree.py), được re-export qua Facade [`src/models/regressors.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/regressors.py).  
> **Module tinh chỉnh:** Hàm [`tune_decision_tree`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tuning.py#L141) trong [`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tuning.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong bài toán dự báo doanh thu bán lẻ chuỗi thời gian theo tuần (Weekly Product Sales Forecasting) trên sàn thương mại điện tử Olist, dữ liệu mang các đặc thù phân phối rất phức tạp:
- **Tính phi tuyến tính mạnh mẽ:** Mối quan hệ giữa doanh thu với các biến trễ thời gian (`last_week_sales`, `sales_mean_4_weeks`), giá bán (`price`) và đánh giá uy tín (`past_review_rating`) không tuân theo quy luật đường thẳng.
- **Dữ liệu thưa và nhiều số 0 (Zero-Inflated Demand):** Đa số sản phẩm có nhiều tuần không phát sinh đơn hàng (`sales = 0`), khiến các mô hình tuyến tính dễ bị nhiễu và dự báo lệch lạc.
- **Tương tác đặc trưng phức hợp:** Sự kết hợp giữa các tuần lễ hội (`is_peak_shopping_season`) và ngành hàng cụ thể (`category_*`) tạo ra các ngưỡng phân nhánh có điều kiện cục bộ.

Mục đích của thuật toán **Decision Tree Regressor** là:
- Chia tách không gian 89 đặc trưng thành các siêu hình hộp trực giao (Axis-aligned hyper-rectangles) đồng nhất về doanh thu bằng cơ chế đệ quy nhị phân.
- Học các quy tắc quyết định dạng cây dễ diễn giải (*If-Else Rules*), phản ánh chính xác các ngưỡng nhạy cảm về doanh thu của từng nhóm sản phẩm.
- Đóng vai trò là khối xây dựng nền tảng (*base learner*) cho các thuật toán Ensemble Bagging (Random Forest) và Boosting (Gradient Boosting).

### 1.2. Cơ sở toán học & Tiêu chuẩn phân chia
Mô hình cài đặt thuần NumPy 100% dựa trên tiêu chuẩn **Giảm phương sai (Variance Reduction)** làm hàm độ lợi tối ưu hóa tại mỗi nút phân nhánh:

Tại một nút $D$ chứa $N$ mẫu với biến mục tiêu $y$, phương sai của nút được định nghĩa:

$$
\text{Var}(D) = \frac{1}{|D|} \sum_{i \in D} \left( y_i - \bar{y}_D \right)^2, \quad \text{với } \bar{y}_D = \frac{1}{|D|} \sum_{i \in D} y_i
$$

Khi phân chia nút $D$ theo thuộc tính $j$ tại ngưỡng $\theta$ thành hai nút con $D_L = \{i \in D \mid x_{ij} \le \theta\}$ và $D_R = \{i \in D \mid x_{ij} > \theta\}$, độ lợi giảm phương sai $\Delta \text{Var}$ được tính:

$$
\Delta \text{Var}(D, j, \theta) = \text{Var}(D) - \left( \frac{|D_L|}{|D|} \text{Var}(D_L) + \frac{|D_R|}{|D|} \text{Var}(D_R) \right)
$$

Thuật toán quét qua các thuộc tính ứng viên và ngưỡng chia cắt để tìm bộ $(j^*, \theta^*)$ làm cực đại hóa $\Delta \text{Var}$. Giá trị dự báo tại mỗi nút lá là trung bình cộng của các mẫu rơi vào lá đó: $\hat{y}_{\text{leaf}} = \bar{y}_{\text{leaf}}$.

---

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
- **Ma trận đặc trưng $X$:** Mảng dense NumPy 2D `float64` kích thước $(N, 89)$ gồm:
  - 16 đặc trưng số học đã qua chuẩn hóa Z-Score (`CustomStandardScaler`).
  - 73 đặc trưng phân loại đã mã hóa One-Hot (`CustomOneHotEncoder`).
- **Vector nhãn mục tiêu $y$:** Vector NumPy 1D kích thước $(N,)$ trong không gian log tự nhiên:
  
  $$
  y = \log(1 + \text{sales}) = \text{log1p}(\text{sales})
  $$

### 2.2. Định dạng dữ liệu đầu ra (Output)
- **Phương thức `predict(X)`:** Trả về vector dự báo liên tục $\hat{y}$ trong không gian log tự nhiên.
- **Hàm chuyển đổi tiền tệ thực tế BRL:** Khi suy luận thực tế và đánh giá hiệu năng, giá trị được chuyển đổi ngược về thang đo BRL và chặn dưới bằng 0:

  $$
  \hat{y}_{\text{sales}} = \max\left(0.0, \exp(\hat{y}) - 1\right) = \max\left(0.0, \text{expm1}(\hat{y})\right)
  $$

---

## 3. Data Used (Dữ liệu sử dụng & Hợp đồng chuỗi thời gian)

### 3.1. Phân chia tập dữ liệu (Temporal Split)
Dự án áp dụng cơ chế phân chia thuần theo dòng thời gian tuần lịch (Temporal Split 80/10/10), ngăn chặn 100% rò rỉ dữ liệu tương lai (*Data Leakage*):
- **Tập Huấn luyện (Train Set):** 655.066 dòng sản phẩm–tuần (~80% số tuần lịch đầu tiên).
- **Tập Kiểm định (Validation Set):** 249.399 dòng sản phẩm–tuần (~10% số tuần lịch tiếp theo, dùng để tinh chỉnh siêu tham số và chọn mô hình chiến thắng).
- **Tập Kiểm thử Độc lập (Test Set):** 293.717 dòng sản phẩm–tuần (~10% số tuần lịch cuối cùng, hoàn toàn giữ kín đến bước nghiệm thu).

### 3.2. Bảo toàn tuần doanh thu bằng 0 (Zero-Sales Weeks)
- Với mỗi sản phẩm, toàn bộ các tuần không có đơn hàng từ tuần bán đầu tiên đến tuần kết thúc quan sát đều được gán `sales = 0.0` BRL.
- Tỷ lệ số dòng có doanh thu bằng 0 chiếm trên 90% tổng số bản ghi trong bảng weekly panel, phản ánh trung thực bản chất thưa của thị trường thương mại điện tử.

### 3.3. Thuộc tính đầu vào (89 Features)
Toàn bộ đặc trưng đều tuân thủ nguyên tắc: **Chỉ sử dụng dữ liệu được chốt trước 00:00 thứ Hai của tuần đích**.
- Lịch sử bán: `price`, `last_week_sales`, `sales_lag_2_weeks`, `sales_lag_4_weeks`.
- Thống kê trượt: `sales_mean_4_weeks`, `sales_std_4_weeks`.
- Sản lượng & Vòng đời: `last_week_units`, `weeks_since_first_sale`.
- Đánh giá & Khách hàng: `past_review_count`, `past_review_rating` (chỉ tính review trả lời trước tuần đích), `customer_past_spend`, `customer_past_orders`, `customer_state_count`.
- Mùa vụ & Ngành hàng: `month_sin`, `month_cos`, `is_peak_shopping_season`, 73 cột danh mục `category_*`.

---

## 4. Evaluation (Phương pháp đánh giá)

### 4.1. Hàm mất mát huấn luyện
Trong quá trình xây dựng cây, tiêu chí tối ưu tương đương với cực tiểu hóa hàm mất mát bình phương trung bình (Mean Squared Error - MSE) trên không gian log:

$$
\mathcal{L}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

### 4.2. Bộ 4 chỉ số đo lường trên thang đo tiền tệ thực tế (BRL)
Để phản ánh chính xác sai số tài chính trong quản trị kho vận, toàn bộ đánh giá trên tập Validation và Test đều được đo trực tiếp trên thang đo tiền tệ BRL:
1. **Mean Squared Error (MSE):** Đơn vị $\text{BRL}^2$.
2. **Mean Absolute Error (MAE):** Đơn vị $\text{BRL}$, đo sai lệch doanh thu trung bình trên mỗi sản phẩm–tuần.
3. **Root Mean Squared Error (RMSE):** Đơn vị $\text{BRL}$, tiêu chí chính để xếp hạng và chọn mô hình.
4. **Hệ số xác định ($R^2$ Score):** Đo lường tỷ lệ phương sai doanh thu được giải thích so với mốc Baseline trung bình.

---

## 5. Status (Trạng thái triển khai & Tối ưu kỹ thuật)

- **Trạng thái:** Hoàn thiện 100% From-Scratch thuần NumPy, không sử dụng thư viện `scikit-learn`.
- **Tối ưu hóa phân vị Quantiles (`n_quantiles=16`):**
  - Trên tập dữ liệu Train khổng lồ gồm 655.066 dòng, việc duyệt qua toàn bộ các giá trị phân biệt $\mathcal{O}(N \log N)$ cho 89 đặc trưng sẽ gây nghẽn nghiêm trọng tài nguyên CPU.
  - Mô hình cài đặt kỹ thuật lấy mẫu 16 điểm phân vị ứng viên (`np.percentile`) trải từ phân vị thứ 5 đến 95, giảm thời gian tìm điểm cắt xuống $\mathcal{O}(Q)$ ($Q = 16$), giúp cây huấn luyện xong chỉ trong **15,22 giây**.
- **Suy luận vector hóa theo batch masks:**
  - Thay vì duyệt từng dòng qua đệ quy lá (`_predict_row`), hàm `predict` chia tập chỉ số theo mặt nạ nhị phân (`left_mask`, `~left_mask`), tăng tốc độ suy luận gấp hàng chục lần.
- **Trích xuất độ quan trọng đặc trưng (Feature Importances):**
  - Tích lũy lượng giảm phương sai có trọng số tại mỗi nút: $\text{Importance}(j) = \sum_{\text{node } t \in \text{split on } j} \Delta \text{Var}(t) \times N_t$, sau đó chuẩn hóa tổng về 1.0.

---

## 6. Results (Kết quả thực nghiệm định lượng)

### 6.1. Quá trình Tinh chỉnh Siêu tham số (Grid Search trên Validation)
Mô hình Decision Tree được quét lưới 6 cấu hình siêu tham số độc lập trên 249.399 dòng Validation:

| Cấu hình | `max_depth` | `min_samples_leaf` | Validation RMSE (BRL) | Đánh giá |
|:---:|:---:|:---:|:---:|:---|
| Config 1 | 3 | 10 | 62.4085 | Cây quá nông, underfitting |
| Config 2 | 3 | 30 | 62.4085 | Underfitting tương tự |
| Config 3 | 5 | 10 | 61.1664 | Bản Default cơ sở |
| Config 4 | 5 | 30 | 61.1664 | Tương đương bản Default |
| **Config 5 (Tuned Winner)** | **8** | **10** | **59.2288** | **Tốt nhất toàn diện, giảm RMSE sâu nhất** |
| Config 6 | 8 | 30 | 59.3857 | Giới hạn lá quá chặt làm giảm độ linh hoạt |

Hệ thống chính thức chọn cấu hình **`max_depth=8, min_samples_leaf=10`** làm **Tuned Model (Mô hình đơn khối tốt nhất)**.

### 6.2. Hiệu năng Độc lập trên Tập Kiểm thử Test (293.717 dòng)

| Phiên bản / Mô hình | MSE ($\text{BRL}^2$) | MAE (BRL) | RMSE (BRL) | $R^2$ Score | WAPE | Supply Chain Cost (BRL) | Fit Time (s) | Xếp hạng toàn hệ thống |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Two-Stage Hurdle (Tham chiếu)* | *2.423,73* | *8,44* | *49,23* | *+0,1148* | *1,97* | *44,80* | *38,32s* | *🥇 Hạng 1 (Winner)* |
| **Tuned Decision Tree (Best Single Model)** | **2.541,22** | **4,30** | **50,41** | **+0,0719** | **1,00** | **40,62** | — | **🥈 Hạng 2 (Đơn khối tốt nhất)** |
| Decision Tree (Default) | 2.615,33 | 4,33 | 51,14 | +0,0449 | 1,01 | 41,26 | 32,77s | 🥉 Hạng 3 |

### 6.3. Nhận xét kỹ thuật & Trade-offs
1. **Năng lực vượt trội so với Baseline:** Cả hai phiên bản Decision Tree đều đạt $R^2 > 0$ và đánh bại hoàn toàn 4 mô hình Baseline (Zero, Revenue Mean, Log Mean, Previous Week có $R^2 < 0$).
2. **Hiệu quả của việc tinh chỉnh (Tuning):** Khi nâng độ sâu từ `max_depth=5` lên `max_depth=8` kết hợp ràng buộc `min_samples_leaf=10`, RMSE trên tập Test giảm từ 51,14 BRL xuống **50,41 BRL** (giảm 0,73 BRL), đồng thời hệ số xác định $R^2$ tăng từ 0,0449 lên **0,0719** (+60,1% năng lực giải thích phương sai). Đặc biệt, mô hình đạt chi phí chuỗi cung ứng thấp nhất toàn dự án (**40,62 BRL/sản phẩm**) và MAE tối ưu (**4,30 BRL**).
3. **Ưu thế tốc độ & Khả năng mở rộng:** Nhờ cơ chế Quantile Split Candidates, cây đạt tốc độ huấn luyện vượt trội (32,77s so với 318,54s của Gradient Boosting) và đóng vai trò làm nền tảng vững chắc cho cả mô hình Two-Stage Hurdle và các giải pháp Ensemble.
