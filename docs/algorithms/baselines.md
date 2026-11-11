# 🎯 Hệ Thống Mô Hình Đối Chứng Cơ Sở (Baselines From-Scratch)

> **Vị trí trong bài lab:** Hệ thống 4 mô hình cơ sở tham chiếu (Non-Machine Learning Baselines) được nhóm thiết kế nhằm tuân thủ nguyên tắc cơ bản của Học Máy: luôn xây dựng mô hình cơ sở đơn giản trước khi thử nghiệm các mô hình phức tạp.
> **Mục tiêu trong bài lab:** Thiết lập mốc so sánh cơ bản để biết mô hình học máy có thực sự học được quy luật hay không trước khi thử nghiệm các mô hình phức tạp hơn.

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong bài toán dự báo doanh thu chuỗi thời gian, nếu chỉ nhìn vào con số sai số mà không có mốc so sánh, nhóm sẽ không biết mô hình học máy có thực sự hiệu quả hay chỉ đang dự báo ngẫu nhiên quanh mức trung bình.

Mục đích của việc thiết lập hệ thống 4 mô hình cơ sở (Baselines):
- Thiết lập mốc đối chiếu theo metric đã chọn. Một mô hình phức tạp chỉ có giá trị khi chứng minh được mức cải thiện rõ ràng so với các baseline đơn giản trên tập Validation.
- Kiểm tra các giả định nghiệp vụ đặc thù: Hiện tượng dữ liệu thưa nhiều số 0 (Zero-Inflated), quy luật quán tính tuần trước (Persistence Lag), và kỳ vọng toán học của phân phối doanh thu.
- Tách bạch rõ ràng giữa năng lực học thực sự của các thuật toán với sự may rủi do cấu trúc dữ liệu tạo ra.

### 1.2. Cơ sở toán học của 4 Mô hình Baseline

#### A. Zero Baseline
- **Nguyên lý:** Luôn dự báo doanh thu bằng 0 BRL cho mọi sản phẩm ở mọi tuần lịch:

$$
  \hat{y}_{\text{zero}}(x) = 0.0 \implies \hat{y}_{\text{sales}} = 0.0 \text{ BRL}
$$

- **Ý nghĩa:** Kiểm tra năng lực chống chịu sai số trên tập dữ liệu có phần lớn số dòng là tuần không bán được hàng (`sales = 0`). Khi median của tập đánh giá bằng 0, Zero là nghiệm tối ưu hàm MAE trong lớp dự báo hằng số; đây là mốc sàn cơ bản để đánh giá hiện tượng thiên lệch dự báo.

#### B. Revenue Mean Baseline
- **Nguyên lý:** Tính trung bình cộng của doanh thu thực tế trên toàn bộ tập Huấn luyện và luôn dự báo hằng số này:

$$
  \bar{y}_{\text{money}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \text{sales}_i
$$

- Giá trị trả về trong không gian log để thống nhất giao diện: $\hat{y} = \log(1 + \bar{y}_{\text{money}})$.
- **Ý nghĩa:** Theo định nghĩa của hệ số xác định $R^2$, dự báo bằng trung bình mẫu là nghiệm tối ưu MSE trong lớp dự báo hằng số trên tập Train.

#### C. Log Mean Baseline
- **Nguyên lý:** Tính trung bình cộng của biến mục tiêu trong không gian log tự nhiên trên tập Train:

$$
  \hat{y}_{\text{log-mean}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} y_i = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \log(1 + \text{sales}_i)
$$

- **Ý nghĩa:** Mốc đối chứng hằng số trong cùng không gian tối ưu hóa logarit của các thuật toán hồi quy.

#### D. Previous Week Baseline
- **Nguyên lý:** Mô hình quán tính chuỗi thời gian (Persistence Forecasting), lấy doanh thu tuần liền trước ($t-1$, tương ứng đặc trưng `last_week_sales`) làm giá trị dự báo cho tuần này ($t$):

$$
  \hat{y}_{\text{sales}}(t) = \text{sales}(t-1)
$$

- **Ý nghĩa:** Trong phân tích chuỗi thời gian truyền thống, mô hình quán tính là mốc so sánh tiêu chuẩn để kiểm chứng xem mô hình học máy có thực sự học được xu hướng và quy luật mùa vụ hay chỉ sao chép số liệu của tuần trước.

---

## 2. Input / Output (Đầu vào & Đầu ra)

- **Input:** Ma trận đặc trưng $X$ hoặc vector đặc trưng (đối với Previous Week Baseline, mô hình trích xuất đúng cột `last_week_sales` dựa trên vị trí đặc trưng và khôi phục giá trị doanh thu gốc).
- **Output:** Vector dự báo liên tục trong không gian log, chuyển đổi sang đơn vị tiền tệ BRL bằng hàm:

$$
\hat{y}_{\text{sales}} = \max(0.0, \exp(\hat{y}) - 1)
$$

---

## 3. Data Used (Dữ liệu sử dụng)

- **Phân chia Temporal Split 80/10/10 theo số tuần lịch:**
  - Tập dữ liệu được chia theo trục thời gian: khoảng 80% số tuần đầu cho Train, 10% số tuần tiếp theo cho Validation, và 10% số tuần cuối cho Test.
- **Nguyên tắc chống rò rỉ dữ liệu (Data Leakage):**
  - Trong học máy, việc chia dữ liệu phải thực hiện TRƯỚC khi tính toán bất kỳ tham số thống kê nào.
  - Các tham số ($\bar{y}_{\text{money}}$, $\bar{y}_{\text{log}}$) CHỈ được tính duy nhất trên tập Train, tuyệt đối không tính trên hoặc cập nhật từ tập Validation hay Test.

---

## 4. Evaluation (Phương pháp đánh giá)

Quy trình đánh giá của nhóm tuân thủ giao thức: lựa chọn mô hình dựa trên sai số tập Validation trước khi đánh giá khách quan trên Test.

- Toàn bộ 4 mô hình cơ sở được đánh giá độc lập trên cùng một tập dữ liệu theo các độ đo trên thang đo tiền tệ thực tế BRL:
  1. Mean Squared Error (MSE): $\text{BRL}^2$.
  2. Mean Absolute Error (MAE): $\text{BRL}$.
  3. Root Mean Squared Error (RMSE): $\text{BRL}$ (tiêu chí chính).
  4. Hệ số xác định ($R^2$ Score).
  5. Weighted Absolute Percentage Error (WAPE).
  6. Cost_Per_Item_BRL: Độ đo chi phí bất đối xứng giữa dự báo thiếu và dự báo thừa trên từng dòng sản phẩm–tuần.

---

## 5. Status (Trạng thái cài đặt & Kết quả đối chuẩn)

- **Đã cài đặt From-Scratch ([`src/models/baselines.py`](../../src/models/baselines.py)):** Cài đặt đầy đủ 4 mô hình đối chứng cơ sở thuần NumPy/Pandas (bao gồm lớp `MeanRegressorFromScratch` cho Log Mean Baseline, hiển thị là `Mean Baseline` trong các bảng kết quả) tuân thủ giao diện `fit()` / `predict()` thống nhất.
- **Kết quả thực nghiệm mới nhất (`11/11/2026`, Validation 249.399 dòng / Test 293.717 dòng):**

| Mô hình Baseline | Val RMSE (BRL) | Test RMSE (BRL) | Test MAE (BRL) | Test $R^2$ | Test WAPE |
| --- | --- | --- | --- | --- | --- |
| Revenue Mean Baseline | `63.3996` | `52.4622` | `11.8955` | `-0.0052` | `2.7798` |
| Mean Baseline (`MeanRegressorFromScratch`) | `63.5862` | `52.4835` | `4.5026` | `-0.0060` | `1.0522` |
| Zero Baseline | `63.6067` | `52.5022` | `4.2792` | `-0.0067` | `1.0000` |
| Previous Week Baseline | `81.9053` | `75.3892` | `9.0765` | `-1.0757` | `2.1211` |

---

## 6. Tiêu Chuẩn Đối Chuẩn & Đánh Giá Mô Hình

### 6.1. Vai trò mốc đối chứng của các baseline hằng số
- Zero Baseline, Revenue Mean Baseline và Log Mean Baseline đại diện cho các giả định hằng số cơ bản (không dùng đặc trưng đầu vào).
- Việc đối chiếu với các baseline này giúp xác định xem mô hình học máy có thực sự học được quy luật từ đặc trưng $X$ hay không.

### 6.2. Vai trò mốc đối chứng chuỗi thời gian của Previous Week Baseline
- Previous Week Baseline đại diện cho giả định quán tính ngắn hạn trong phân tích chuỗi thời gian truyền thống ($s_t \approx s_{t-1}$).
- Đối chiếu với mô hình quán tính là bước bắt buộc để chứng minh mô hình máy học có khả năng nắm bắt xu hướng tốt hơn việc chỉ sao chép số liệu tuần trước. Thực tế trên tập dữ liệu thưa `96.21%` số 0 của Olist, việc lấy nguyên tuần trước cho sai số rất cao (Val RMSE `81.9053 BRL`, Test RMSE `75.3892 BRL`).

### 6.3. Tiêu chuẩn đánh giá mô hình học máy
Nhóm đặt ra tiêu chuẩn: Một mô hình học máy ứng viên (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Two-Stage Hurdle) chỉ được nhóm đề xuất giữ lại nếu:
1. Đạt **Validation RMSE (BRL)** thấp hơn baseline tốt nhất trong 4 baseline trên (`Revenue Mean Baseline`: `63.3996 BRL`).
2. Có khả năng giải thích hoặc mô hình hóa được phân phối xác suất phát sinh giao dịch thay vì chỉ dự báo theo một quy tắc hằng số cứng nhắc.

Trong lần chạy gần nhất, `Two-Stage Hurdle` (`55.9393 BRL`), `Tuned Model` (`Decision Tree max_depth=8`: `59.2288 BRL`), `Decision Tree` (`61.1664 BRL`), `Gradient Boosting` (`62.8054 BRL`) và `Random Forest` (`63.1106 BRL`) đều vượt mốc baseline này trên tập Validation, còn `Linear Regression` không đạt tiêu chuẩn.
