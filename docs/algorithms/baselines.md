# 🎯 Hệ Thống Mô Hình Đối Chứng Cơ Sở (Baselines From-Scratch)

> **Vị trí trong dự án:** Hệ thống 4 mô hình cơ sở tham chiếu (Non-Machine Learning Baselines) tuân thủ nghiêm ngặt nguyên tắc học máy (Rule 10: *Always propose a baseline before complex models*).  
> **Mã nguồn triển khai:** Các lớp [`ZeroRegressorFromScratch`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/baselines.py#L17), [`RevenueMeanRegressorFromScratch`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/baselines.py#L28), [`MeanRegressorFromScratch`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/baselines.py#L47) và [`PreviousWeekRegressorFromScratch`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/baselines.py#L65) trong [`src/models/baselines.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/baselines.py), được re-export qua Facade [`src/models/regressors.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/regressors.py).

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong các bài toán hồi quy chuỗi thời gian bán lẻ, nếu chỉ báo cáo các chỉ số tuyệt đối (như RMSE = 50 BRL) mà không có hệ thống đối chứng, người nghiên cứu sẽ không thể xác định được mô hình máy học có thực sự mang lại giá trị gia tăng (*added value*) hay chỉ đang dự báo ngẫu nhiên quanh mức trung bình.

Mục đích của việc thiết lập hệ thống 4 mô hình cơ sở (Baselines):
- Thiết lập các mốc tham chiếu định lượng tối thiểu (Lower Bounds of Performance) mà bất kỳ mô hình máy học hợp lệ nào cũng bắt buộc phải vượt qua ($R^2 > 0$).
- Kiểm tra các giả định nghiệp vụ đặc thù: Hiện tượng dữ liệu thưa nhiều số 0 (Zero-Inflated), quy luật quán tính tuần trước (Persistence Lag), và kỳ vọng toán học của phân phối doanh thu.
- Tách bạch rõ ràng giữa năng lực học thực sự của các thuật toán với sự may rủi do cấu trúc dữ liệu tạo ra.

### 1.2. Cơ sở toán học của 4 Mô hình Baseline

#### A. Zero Baseline (`ZeroRegressorFromScratch`)
- **Nguyên lý:** Luôn dự báo doanh thu bằng 0 BRL cho mọi sản phẩm ở mọi tuần lịch:
  
  $$
  \hat{y}_{\text{zero}}(x) = 0.0 \implies \hat{y}_{\text{sales}} = 0.0 \text{ BRL}
  $$

- **Ý nghĩa:** Kiểm tra năng lực chống chịu sai số trên tập dữ liệu có trên 90% số dòng là tuần không bán được hàng (`sales = 0`). Đây là mốc chặn dưới tối thiểu cho sai số tuyệt đối MAE.

#### B. Revenue Mean Baseline (`RevenueMeanRegressorFromScratch`)
- **Nguyên lý:** Tính trung bình cộng của doanh thu thực tế trên toàn bộ tập Huấn luyện và luôn dự báo hằng số này:
  
  $$
  \bar{y}_{\text{money}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \text{sales}_i
  $$

- Giá trị trả về trong không gian log để thống nhất giao diện: $\hat{y} = \log(1 + \bar{y}_{\text{money}})$.
- **Ý nghĩa:** Theo định nghĩa của hệ số xác định $R^2$, dự báo bằng trung bình mẫu là nghiệm tối ưu của hàm mất mát MSE trên tập Train ($R^2_{\text{train}} = 0$).

#### C. Log Mean Baseline (`MeanRegressorFromScratch`)
- **Nguyên lý:** Tính trung bình cộng của nhãn trong không gian log tự nhiên trên tập Train:
  
  $$
  \hat{y}_{\text{log\_mean}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} y_i = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \log(1 + \text{sales}_i)
  $$

- **Ý nghĩa:** Mốc đối chứng hằng số trong cùng không gian tối ưu hóa của các thuật toán hồi quy.

#### D. Previous Week Baseline (`PreviousWeekRegressorFromScratch`)
- **Nguyên lý:** Mô hình quán tính chuỗi thời gian (Persistence Forecasting), lấy doanh thu tuần liền trước ($t-1$, tương ứng đặc trưng `last_week_sales`) làm giá trị dự báo cho tuần này ($t$):
  
  $$
  \hat{y}_{\text{sales}}(t) = \text{sales}(t-1)
  $$

- **Ý nghĩa:** Trong phân tích chuỗi thời gian truyền thống, mô hình quán tính là mốc so sánh tiêu chuẩn để chứng minh mô hình mới có khả năng nắm bắt xu hướng tốt hơn việc chỉ "sao chép" số liệu của kỳ trước.

---

## 2. Input / Output (Đầu vào & Đầu ra)

- **Input:** Ma trận $X$ hoặc vector đặc trưng (đối với `PreviousWeekRegressorFromScratch`, mô hình trích xuất đúng cột `last_week_sales` dựa trên vị trí `column_index` và tham số phân phối $(\mu, \sigma)$ của Train).
- **Output:** Vector dự báo liên tục trong không gian log, chuyển đổi sang BRL bằng hàm $\max(0, \exp(\hat{y}) - 1)$.

---

## 3. Data Used (Dữ liệu sử dụng)

- **Phân chia Temporal Split 80/10/10:**
  - Fit duy nhất trên **655.066 dòng** Train.
  - Đánh giá trên **249.399 dòng** Validation và **293.717 dòng** Test.
- **Tuân thủ quy tắc đóng băng tham số:** Các tham số thống kê ($\bar{y}_{\text{money}}$, $\bar{y}_{\text{log}}$) chỉ được học trên tập Train, hoàn toàn không được tính lại trên Validation hoặc Test để tránh rò rỉ dữ liệu.

---

## 4. Evaluation (Phương pháp đánh giá)

- Toàn bộ 4 mô hình cơ sở đều được đánh giá độc lập trên tập Kiểm thử Test (293.717 dòng) bằng bộ 4 chỉ số trên thang đo tiền tệ BRL:
  1. Mean Squared Error (MSE): $\text{BRL}^2$.
  2. Mean Absolute Error (MAE): $\text{BRL}$.
  3. Root Mean Squared Error (RMSE): $\text{BRL}$.
  4. Hệ số xác định ($R^2$ Score).

---

## 5. Status (Trạng thái triển khai)

- **Trạng thái:** Hoàn thiện 100% From-Scratch thuần NumPy, đóng gói dưới dạng các class có đầy đủ phương thức chuẩn `fit(X, y)` và `predict(X)`.
- **Tương thích giao diện thống nhất:** Cho phép tích hợp trực tiếp vào quy trình đánh giá chéo và xếp hạng tự động trong `main.py` mà không cần xử lý ngoại lệ.

---

## 6. Results (Kết quả thực nghiệm định lượng)

### 6.1. Bảng Kết quả Chi tiết của 4 Mô hình Baseline trên Test Set (293.717 dòng)

| STT | Mô hình Baseline | MSE ($\text{BRL}^2$) | MAE (BRL) | RMSE (BRL) | $R^2$ Score | Fit Time (s) | Xếp hạng trong nhóm Baseline |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Revenue Mean Baseline** | **2.752,28** | 11,90 | **52,46** | **-0,0052** | 0,00s | 🥇 Hạng 1 (RMSE thấp nhất) |
| 2 | **Mean Baseline (Log space)** | 2.754,52 | 4,50 | 52,48 | -0,0060 | 0,00s | 🥈 Hạng 2 |
| 3 | **Zero Baseline** | 2.756,48 | **4,28** | 52,50 | -0,0067 | 0,00s | 🥉 Hạng 3 (MAE thấp nhất) |
| 4 | **Previous Week Baseline** | 5.683,53 | 9,08 | 75,39 | -1,0757 | 0,00s | 4 (Kém nhất) |

### 6.2. Phân tích Hiện tượng Thống kê & Bài học Kỹ thuật

#### A. Hiện tượng $R^2$ âm nhẹ của các Baseline Hằng số (-0,0052 đến -0,0067)
- Cả Revenue Mean, Log Mean và Zero Baseline đều có $R^2$ dao động quanh mức $0$ (từ $-0,0052$ đến $-0,0067$).
- Giá trị này mang tính chuẩn mực lý thuyết: Do tập Test nằm ở tương lai (Temporal Split) nên phân phối doanh thu trung bình thực tế của Test có sự dịch chuyển nhẹ (*Concept Drift*) so với Train, khiến việc dự báo bằng trung bình Train đạt $R^2$ âm rất nhỏ trên Test.
- **Zero Baseline đạt MAE thấp nhất (4,28 BRL):** Vì hơn 90% số tuần có doanh thu bằng 0, việc dự báo hằng số 0 sẽ đạt sai số tuyệt đối bằng 0 trên phần lớn tập dữ liệu. Tuy nhiên, mô hình này hoàn toàn vô dụng về mặt kinh doanh vì không thể dự báo được bất kỳ đơn hàng nào phát sinh.

#### B. Sự sụp đổ của Previous Week Baseline ($R^2 = -1,0757$)
- Mô hình quán tính tuần trước có sai số RMSE vọt lên **75,39 BRL** (gấp 1,5 lần các baseline khác) và $R^2$ âm nặng tới **$-1,0757$**.
- **Nguyên nhân cốt lõi (Hiện tượng trễ pha & Nhu cầu ngắt quãng):**
  - Trong thương mại điện tử, các sản phẩm bán lẻ thường có tính ngắt quãng cao (Intermittent Demand): Một sản phẩm bán được 100 BRL ở tuần $t-1$ thường sẽ rơi vào tình trạng không có đơn hàng (`sales = 0`) ở tuần $t$ tiếp theo.
  - Mô hình quán tính mắc hai sai lầm đối xứng nghiêm trọng:
    1. **False Positive (Báo ảo):** Tuần trước có đơn hàng $\implies$ tuần này dự báo có doanh thu cao, nhưng thực tế doanh thu bằng 0.
    2. **False Negative (Bỏ sót):** Tuần trước không bán được $\implies$ tuần này dự báo 0 BRL, nhưng thực tế sản phẩm bất ngờ có giao dịch lớn.
  - Điều này chứng minh rằng việc sao chép số liệu kỳ trước là chiến lược tồi tệ nhất cho bài toán dự báo chuỗi thời gian thưa.

### 6.3. Khẳng định Giá trị Gia tăng của các Thuật toán Máy học
Sự tồn tại của hệ thống 4 Baseline này làm nổi bật giá trị thực tế của các mô hình học máy được triển khai:
- **Two-Stage Hurdle (Selected Artifact):** Đạt **RMSE 49,23 BRL** (giảm 3,23 BRL so với Revenue Mean 52,46 BRL), MSE **2.423,73**, và $R^2 = \mathbf{+0,1148}$ (kỷ lục toàn dự án).
- **Tuned Decision Tree (Best Single Model):** Đạt **RMSE 50,41 BRL**, MAE **4,30 BRL**, $R^2 = \mathbf{+0,0719}$, cùng chi phí chuỗi cung ứng tối ưu nhất (**40,62 BRL/sản phẩm**).
- Khẳng định các mô hình máy học đã thực sự học được quy luật có ý nghĩa từ dữ liệu thay vì chỉ dự báo ngẫu nhiên quanh giá trị trung bình.
