# 🎯 Hệ Thống Mô Hình Đối Chứng Cơ Sở (Baselines From-Scratch)

> **Vị trí trong bài lab:** Hệ thống 4 mô hình cơ sở tham chiếu (Non-Machine Learning Baselines) được nhóm thiết kế nhằm tuân thủ nguyên tắc cơ bản của Học Máy: luôn xây dựng mô hình cơ sở đơn giản trước khi thử nghiệm các mô hình phức tạp.
> **Mục tiêu trong bài lab:** Thiết lập mốc so sánh cơ bản để biết mô hình học máy có thực sự học được quy luật hay không trước khi thử nghiệm các mô hình phức tạp hơn.

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong bài toán dự báo doanh thu chuỗi thời gian, nếu chỉ nhìn vào con số sai số mà không có mốc so sánh, nhóm sẽ không biết mô hình học máy có thực sự hiệu quả hay chỉ đang dự báo ngẫu nhiên quanh mức trung bình.

Mục đích của việc thiết lập hệ thống 4 mô hình cơ sở (Baselines):
- Thiết lập mốc đối chiếu theo metric đã chọn. Mức cải thiện trên Validation là căn cứ lựa chọn; các mô hình vẫn có thể được thử để học cách hoạt động.
- Kiểm tra các giả định nghiệp vụ đặc thù: Hiện tượng dữ liệu thưa nhiều số 0 (Zero-Inflated), quy luật quán tính tuần trước (Persistence Lag), và kỳ vọng toán học của phân phối doanh thu.
- Hạn chế diễn giải sai số thấp như bằng chứng đủ về khả năng dự báo khi chưa có mốc so sánh.

### 1.2. Cơ sở toán học của 4 Mô hình Baseline

#### A. Zero Baseline
- **Nguyên lý:** Luôn dự báo doanh thu bằng 0 BRL cho mọi sản phẩm ở mọi tuần lịch:

$$
  \hat{y}_{\text{zero}}(x) = 0.0 \implies \hat{y}_{\text{sales}} = 0.0 \text{ BRL}
$$

- **Ý nghĩa:** Kiểm tra năng lực chống chịu sai số trên tập dữ liệu có phần lớn số dòng là tuần không bán được hàng (doanh thu bằng 0). Khi median của tập đánh giá bằng 0, Zero là nghiệm tối ưu hàm MAE trong lớp dự báo hằng số; đây là mốc sàn cơ bản để đánh giá hiện tượng thiên lệch dự báo.

#### B. Revenue Mean Baseline
- **Nguyên lý:** Tính trung bình cộng của doanh thu thực tế trên toàn bộ tập Huấn luyện và luôn dự báo hằng số này:

$$
  \bar{y}_{\text{money}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \text{sales}_i
$$

- Giá trị trả về trong không gian log để dùng chung cách biểu diễn dự báo: $\hat{y} = \log(1 + \bar{y}_{\text{money}})$.
- **Ý nghĩa:** Theo định nghĩa của hệ số xác định $R^2$, dự báo bằng trung bình mẫu là nghiệm tối ưu MSE trong lớp dự báo hằng số trên tập Train.

#### C. Log Mean Baseline
- **Nguyên lý:** Tính trung bình cộng của biến mục tiêu trong không gian log tự nhiên trên tập Train:

$$
  \hat{y}_{\text{log-mean}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} y_i = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \log(1 + \text{sales}_i)
$$

- **Ý nghĩa:** Mốc đối chứng hằng số trong cùng không gian tối ưu hóa logarit của các thuật toán hồi quy.

#### D. Previous Week Baseline
- **Nguyên lý:** Mô hình quán tính chuỗi thời gian (Persistence Forecasting), lấy doanh thu tuần liền trước ($t-1$, tương ứng đặc trưng doanh thu tuần trước) làm giá trị dự báo cho tuần này ($t$):

$$
  \hat{y}_{\text{sales}}(t) = \text{sales}(t-1)
$$

- **Ý nghĩa:** Trong phân tích chuỗi thời gian truyền thống, mô hình quán tính là mốc so sánh tiêu chuẩn để kiểm chứng xem mô hình học máy có thực sự học được xu hướng và quy luật mùa vụ hay chỉ sao chép số liệu của tuần trước.

---

## 2. Input / Output (Đầu vào & Đầu ra)

- **Input:** Thông tin tương ứng với từng baseline; riêng Previous Week cần doanh thu tuần liền trước trên thang BRL.
- **Output:** Vector dự báo liên tục trong không gian log, chuyển đổi sang đơn vị tiền tệ BRL bằng hàm:

$$
\hat{y}_{\text{sales}} = \max(0.0, \exp(\hat{y}) - 1)
$$

---

## 3. Data Used (Dữ liệu sử dụng)

- **Phân chia Temporal Split 80/10/10 theo số tuần lịch:**
  - Tập dữ liệu được chia theo trục thời gian: khoảng 80% số tuần đầu cho Train, 10% số tuần tiếp theo cho Validation, và 10% số tuần cuối cho Test.
- **Nguyên tắc chống rò rỉ dữ liệu (Data Leakage):**
  - Chia tập trước khi học tham số mô hình và thống kê tiền xử lý; đặc trưng lịch sử tuân thủ mốc dự báo của từng mẫu.
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
  6. Điểm phạt thiếu/thừa giả định: Điểm phạt theo trọng số giả định, không phải chi phí vận hành đo được, cho dự báo thiếu và thừa trên từng dòng sản phẩm–tuần.

---

## 5. Design (Phương án đối chứng)

Nhóm dự kiến dùng cùng cách chia dữ liệu và cùng đơn vị đánh giá cho bốn baseline. Bảng sau nêu nội dung đối chiếu, chưa ghi số đo thực nghiệm:

| Mô hình Baseline | Thông tin sử dụng | Nội dung dự kiến kiểm tra |
| --- | --- | --- |
| Revenue Mean Baseline | Trung bình doanh thu Train | Sai số khi dùng một mức doanh thu chung |
| Log Mean Baseline | Trung bình nhãn log trên Train | Khác biệt với trung bình doanh thu sau phép nghịch đảo log |
| Zero Baseline | Giá trị 0 | MAE và RMSE khi không dự báo doanh thu dương |
| Previous Week Baseline | Doanh thu tuần liền trước | Khả năng tham chiếu của lịch sử gần nhất |

---

## 6. Tiêu Chuẩn Đối Chuẩn & Đánh Giá Mô Hình

### 6.1. Vai trò mốc đối chứng của các baseline hằng số
- Zero Baseline, Revenue Mean Baseline và Log Mean Baseline đại diện cho các giả định hằng số cơ bản (không dùng đặc trưng đầu vào).
- Việc đối chiếu với các baseline này giúp xác định xem mô hình học máy có thực sự học được quy luật từ đặc trưng $X$ hay không.

### 6.2. Vai trò mốc đối chứng chuỗi thời gian của Previous Week Baseline
- Previous Week Baseline đại diện cho giả định quán tính ngắn hạn trong phân tích chuỗi thời gian truyền thống ($s_t \approx s_{t-1}$).
- Đối chiếu với mô hình quán tính giúp kiểm tra liệu cách học từ nhiều đặc trưng có cải thiện hơn việc sao chép doanh thu tuần trước hay không.

### 6.3. Tiêu chuẩn đánh giá mô hình học máy
Nhóm đặt ra tiêu chuẩn: Một mô hình học máy ứng viên (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Two-Stage Hurdle) chỉ được nhóm đề xuất giữ lại nếu:
1. Đạt **Validation RMSE (BRL)** thấp hơn baseline tốt nhất trong bốn baseline.
2. Trình bày được cách dự báo, mức độ phức tạp và các trường hợp còn hạn chế.

Nếu baseline có điểm tốt nhất theo tiêu chí đã chọn, nhóm sẽ giữ baseline. Test chỉ dùng để đánh giá sau lựa chọn.
