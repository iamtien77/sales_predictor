# 🎯 Mô Hình Hai Giai Đoạn (Two-Stage Hurdle Regressor From-Scratch)

> **Vai trò:** Được chọn bằng Validation RMSE **55,9393 BRL**; Test RMSE **49,2314 BRL**, R² **0,1148**. Artifact phục vụ demo học thuật; không thắng mọi metric.
> **Mã nguồn triển khai:** Class [`TwoStageHurdleRegressorFromScratch`](../../src/models/hurdle.py) trong [`src/models/hurdle.py`](../../src/models/hurdle.py), được re-export qua Facade [`src/models/regressors.py`](../../src/models/regressors.py).
> **Kiểm thử đơn vị:** Tệp [`tests/test_hurdle_and_metrics.py`](../../tests/test_hurdle_and_metrics.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong bài toán dự báo chuỗi thời gian bán lẻ theo từng cặp Sản phẩm – Tuần (Weekly Product Sales Panel) trên sàn thương mại điện tử Olist Brazil:
- **Đặc thù phân phối Zero-Inflated cực đoan:** Hơn 90% số quan sát trong tập dữ liệu có doanh thu bằng đúng **0 BRL** (tuần không phát sinh đơn hàng). Chỉ có dưới 10% tuần thực sự bán được hàng.
- **Xung đột mục tiêu:** Nhiều mẫu 0 có thể kéo dự báo xuống thấp, nhưng mức chi phối MSE còn phụ thuộc độ lớn sai số của mẫu dương. Tối ưu MSE log không tương đương tối ưu RMSE BRL.
- **Nhu cầu ra quyết định kinh doanh:** Nghiệp vụ quản trị chuỗi cung ứng cần trả lời hai câu hỏi độc lập:
  1. *Tuần này sản phẩm có khả năng bán được hay không?* (Phục vụ lập kế hoạch kích cầu / tiếp thị)
  2. *Nếu bán được, khối lượng doanh thu kỳ vọng là bao nhiêu?* (Phục vụ nhập hàng và chuẩn bị kho bãi)

### 1.2. Cơ sở toán học & Kiến trúc Hai giai đoạn (Hurdle Architecture)
Mô hình Hurdle tách bài toán thành hai giai đoạn để học xác suất bán và độ lớn doanh thu dương; cách tách này không bảo đảm loại bỏ mọi sai số:

```
                       ┌─────────────────────────┐
                       │  Ma trận đặc trưng x   │
                       └────────────┬────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌───────────────────────────────┐        ┌───────────────────────────────────────┐
│   Giai đoạn 1 (Occurrence)    │        │         Giai đoạn 2 (Severity)        │
│   Ước lượng Xác suất có đơn   │        │     Hồi quy Doanh thu có điều kiện    │
│    p(x) = P(sales > 0 | x)    │        │  s_pos(x) = expm1(predicted log)   │
└──────────────┬────────────────┘        └───────────────────┬───────────────────┘
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │    Dự báo Hợp nhất       │
                        │ sales_hat = p(x) * s_pos(x)│
                        └───────────────────────────┘
```

#### A. Giai đoạn 1: Mô hình Ước lượng Xác suất phát sinh giao dịch (Occurrence Stage)
- **Tập dữ liệu huấn luyện:** Toàn bộ mẫu huấn luyện với nhãn nhị phân:

  $$
  z_i = \mathbb{I}(y_i > 0) = \begin{cases} 1 & \text{nếu } y_i > 0 \\ 0 & \text{nếu } y_i = 0 \end{cases}
  $$

- **Bộ phân loại:** Cài đặt bằng `DecisionTreeRegressorFromScratch(max_depth=5, min_samples_leaf=5)`. Tại mỗi nút lá $L$, giá trị trung bình mẫu chính là xác suất thực nghiệm rơi vào lớp dương:

  $$
  p(x) = \hat{P}(\text{sales} > 0 \mid x) = \frac{1}{|L|} \sum_{i \in L} z_i \in [0, 1]
  $$

#### B. Giai đoạn 2: Mô hình Hồi quy Doanh thu có điều kiện (Severity Stage)
- **Tập dữ liệu huấn luyện:** Chỉ bao gồm các mẫu **thực sự phát sinh đơn hàng** ($y_i > 0$):

  $$
  \mathcal{D}_{\text{pos}} = \{(x_i, y_i) \mid y_i > 0\}
  $$

- **Bộ hồi quy:** Cài đặt bằng `DecisionTreeRegressorFromScratch(max_depth=6, min_samples_leaf=5)`. Mô hình học hoàn toàn không bị nhiễu bởi các mẫu 0, tập trung toàn bộ năng lực phân nhánh để giải thích độ lớn của các đơn hàng.
- **Giá trị dự báo tiền tệ có điều kiện:**

  $$
  \hat{s}_{\text{pos}}(x) = \max\left(0.0, \exp(\hat{y}_{\text{pos}}(x)) - 1\right)
  $$

#### C. Tổng hợp Kỳ vọng Toán học (Expected Revenue Combination)
Đẳng thức lý thuyết về kỳ vọng là công thức dưới đây. Tuy nhiên, estimator nhánh dương trong code học trung bình log, nên phép nghịch đảo không mặc nhiên bằng kỳ vọng số học:

$$
\mathbb{E}[\text{sales} \mid x] = P(\text{sales} > 0 \mid x) \times \mathbb{E}[\text{sales} \mid \text{sales} > 0, x]
$$

Implementation kết hợp $\hat{s}(x)=p(x)\hat{s}_{\text{pos}}(x)$ rồi trả về log1p. Trong một lá dương, Jensen cho $\exp(\overline{\log(1+s)})-1 \le \overline{s}$; đẳng thức chỉ khi doanh thu trong lá không biến thiên. Chưa có hiệu chỉnh retransformation hoặc calibration xác suất. Vì vậy đây là dự báo kết hợp, chưa phải estimator đã chứng minh không chệch của kỳ vọng doanh thu:

$$
\hat{y} = \log\left(1 + p(x)\hat{s}_{\text{pos}}(x)\right)
$$

---

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
- **Ma trận đặc trưng $X$:** Mảng dense NumPy 2D `float64` kích thước $(N, 89)$ bao gồm:
  - 16 đặc trưng số học (Lag 1 tuần, Lag 2 tuần, Lag 4 tuần, rolling mean 4 tuần, rolling std, giá tuần trước, lịch thời gian...).
  - 73 đặc trưng phân loại đã mã hóa One-Hot (nhóm ngành hàng sản phẩm).
- **Vector nhãn mục tiêu $y$:** Vector NumPy 1D kích thước $(N,)$ trong không gian log tự nhiên: $y = \log(1 + \text{sales})$.

### 2.2. Định dạng dữ liệu đầu ra (Output)
- **Phương thức `predict(X)`:** Trả về vector log1p của dự báo kết hợp $\hat{y}=\log(1+p(x)\hat{s}_{\text{pos}}(x))$.
- **Phương thức nghiệp vụ mở rộng:**
  - `predict_proba_sale(X)`: Trả về xác suất phát sinh giao dịch $p(x) \in [0, 1]$.
  - `predict_positive_sales(X)`: Trả về mức doanh thu BRL dự kiến trong tình huống có người mua.
- **Chuyển đổi tiền tệ thực tế BRL:**

  $$
  \hat{y}_{\text{money}} = \max\left(0.0, \exp(\hat{y}) - 1\right) \text{ BRL}
  $$

---

## 3. Data used (Tập dữ liệu sử dụng & Đặc thù phân phối)

- **Chiến lược phân chia:** Phân chia chuỗi thời gian theo tuần lịch (Temporal Split 80/10/10) trên 1.198.182 mẫu:
  - **Tập Train (80% số tuần):** 655.066 dòng (chiếm các tuần đầu lịch sử).
  - **Tập Validation (10% số tuần):** 249.399 dòng (các tuần kế tiếp).
  - **Tập Test (10% số tuần):** 293.717 dòng (các tuần cuối cùng, kiểm định độc lập).
- **Đặc trưng tập dương trong Train:**
  - Nhánh severity chỉ dùng các mẫu Train có nhãn dương. CSV hiện có không ghi số mẫu dương hoặc thời gian riêng từng stage, nên không suy ra các số này từ tổng thời gian fit.
- **Biến mục tiêu:** Doanh thu tuần của từng sản phẩm (tổng `price` các đơn hàng giao dịch thành công trong tuần, đơn vị BRL).

---

## 4. Evaluation (Phương pháp đánh giá)

Protocol CLI chọn bằng Validation RMSE BRL trước khi báo cáo Test. Tỷ lệ split tính theo tuần; lag trong Test dùng lịch sử trước từng tuần đích, không phải dự báo nhiều tuần từ một origin cố định. Test đã được xem trong các vòng phân tích nên không coi là holdout mù mới. Số features thuộc snapshot, có thể thay đổi khi fit encoder trên dữ liệu khác.

Hệ thống đánh giá trên thang đo tiền tệ thực tế BRL với đầy đủ bộ chỉ số khoa học và nghiệp vụ:
1. **Root Mean Squared Error (RMSE):** Đo lường độ lệch bình phương trung bình trên đơn vị BRL.
2. **Mean Squared Error (MSE):** Thang đo phạt các sai số cực đoan ($\text{BRL}^2$).
3. **Mean Absolute Error (MAE):** Đo lường sai lệch tuyệt đối trung bình (BRL).
4. **Hệ số xác định ($R^2$ Score):** So sánh SSE với dự báo hằng bằng trung bình của chính tập đánh giá. R² dương không chứng minh ý nghĩa thống kê hoặc lợi ích kinh doanh; khi target hằng, cần nêu quy ước xử lý riêng.
5. **Weighted Absolute Percentage Error (WAPE):**

   $$
   \text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}
   $$

   WAPE chấp nhận từng dòng actual bằng 0 nếu tổng actual dương. Khi tổng actual ≤ 1e-9, implementation trả 0 nếu tổng lỗi ≤ 1e-9, ngược lại trả 1; đây là quy ước phần mềm, không phải tỷ lệ WAPE có nghĩa khi mẫu số bằng 0. Giá trị 1,97 tương ứng khoảng 197%.
6. **Supply Chain Cost Per Item (BRL):** Điểm phạt giả định $N^{-1}\sum_i[10\max(s_i-\hat{s}_i,0)+2\max(\hat{s}_i-s_i,0)]$ trên mỗi mẫu sản phẩm–tuần. Hệ số 10 và 2 nhân sai số doanh thu, không phải chi phí tồn kho thực đo trên mỗi đơn vị hàng.

---

## 5. Status (Trạng thái triển khai & Tối ưu kỹ thuật)

- **Trạng thái:** Logic mô hình được cài đặt From-Scratch bằng NumPy; dự án vẫn dùng các thư viện hỗ trợ dữ liệu và ma trận, không dùng estimator của scikit-learn.
- **Tách module độc lập:** Mã nguồn được đóng gói sạch tại [`src/models/hurdle.py`](../../src/models/hurdle.py) và kiểm thử hồi quy tại [`tests/test_hurdle_and_metrics.py`](../../tests/test_hurdle_and_metrics.py).
- **Cơ chế Fallback an toàn:** Nếu số lượng mẫu dương trong tập huấn luyện quá nhỏ ($< \max(10, \text{min\_samples\_leaf})$), mô hình tự động kích hoạt cơ chế dự phòng `pos_fallback_value_` bằng trung bình **log của mẫu dương**; nếu không có mẫu dương thì dùng 0 và dự báo kết hợp bằng 0. Nếu tất cả mẫu đều dương, occurrence bằng 1. Các phương thức dự báo yêu cầu đã fit; fallback không thay thế kiểm tra dữ liệu đầu vào.

---

## 6. Results & Thảo luận Thực nghiệm (Results & Discussion)

### 6.1. Kết quả Kiểm định Độc lập trên Test Set (293.717 dòng)

| Chỉ số kiểm định | Two-Stage Hurdle | Decision Tree | Random Forest | Gradient Boosting | Revenue Mean Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|
| **RMSE (BRL)** | **49,23** | 51,14 | 52,20 | 52,04 | 52,46 |
| **MSE ($\text{BRL}^2$)** | **2.423,73** | 2.615,33 | 2.725,33 | 2.708,22 | 2.752,28 |
| **$R^2$ Score** | **+0,1148** | +0,0449 | +0,0047 | +0,0109 | -0,0052 |
| **MAE (BRL)** | 8,44 | 4,33 | 4,43 | 4,42 | 11,90 |
| **WAPE** | 1,97 | 1,01 | 1,04 | 1,03 | 2,78 |
| **Cost Per Item (BRL)**| 44,80 | 41,26 | 42,69 | 42,49 | 56,35 |
| **Thời gian huấn luyện** | **15,08s** | 16,55s | 42,11s | 138,50s | 0,00s |

### 6.2. Phân tích Bản chất Kỹ thuật & Đánh đổi (Trade-off Analysis)

1. **Vượt trội về khả năng phân tán phương sai ($R^2$ cao kỷ lục toàn hệ thống):**
   - Đạt **$R^2 = +0,1148$**, cao gấp hơn **2,5 lần** so với Decision Tree mặc định ($R^2 = +0,0449$) và vượt qua Tuned Decision Tree ($R^2 = +0,0719$).
   - Về chỉ số RMSE, mô hình đạt **49,23 BRL**, là mức sai số thấp nhất trong toàn bộ hệ thống (giảm 6,2% so với Zero Baseline 52,50 BRL).
   - Giả thuyết: tách nhánh dương có thể giảm tác động của các mẫu 0 lên severity. Cần ablation cùng dữ liệu và protocol để xác định đóng góp của từng stage; metrics tổng hợp chưa chứng minh cơ chế này.

2. **Tốc độ huấn luyện tối ưu:**
   - CSV ghi tổng thời gian fit **15,08 giây**, không tách hai giai đoạn. Random Forest: **42,11 giây**; Gradient Boosting: **138,50 giây**.

3. **Đánh đổi giữa MAE và RMSE (The MAE vs. RMSE Trade-off):**
   - Chỉ số MAE của mô hình Hurdle (**8,44 BRL**) cao hơn Decision Tree (**4,33 BRL**) và Tuned DT (**4,30 BRL**).
   - Hurdle có RMSE thấp hơn nhưng MAE cao hơn Zero Baseline. Đây là trade-off quan sát trên Test; chưa đo riêng khả năng phát hiện nhu cầu đột biến. Dự báo dương nghịch đảo từ log không mặc nhiên là kỳ vọng số học.
