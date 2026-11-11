# 🌳 Cây Quyết Định Hồi Quy (Decision Tree Regressor From-Scratch)

> **Vị trí trong bài lab:** Mô hình hồi quy phi tuyến tính dạng cây, được nhóm thiết kế nhằm học các mối quan hệ phi tuyến trong dữ liệu thương mại điện tử. Đây đồng thời là khối cơ sở cho các thuật toán Ensemble (Random Forest, Gradient Boosting) và mô hình Two-Stage Hurdle.
> **Mục tiêu trong bài lab:** Thử nghiệm mô hình cây quyết định dạng quy tắc If-Else đơn giản, dễ giải thích và xem có cải thiện hơn so với các baseline cơ sở hay không.

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong bài toán dự báo doanh thu bán lẻ chuỗi thời gian theo tuần trên sàn thương mại điện tử Olist, dữ liệu mang các đặc thù phân phối phức tạp:
- **Tính phi tuyến tính mạnh mẽ:** Mối quan hệ giữa doanh thu với các biến trễ thời gian (`last_week_sales`, `sales_mean_4_weeks`), giá bán (`price`) và đánh giá chất lượng (`past_review_rating`) không tuân theo quy luật đường thẳng.
- **Dữ liệu thưa và nhiều số 0 (Zero-Inflated Demand):** Đa số sản phẩm có nhiều tuần không phát sinh đơn hàng (`sales = 0`), khiến mô hình tuyến tính dễ bị nhiễu cục bộ.
- **Tương tác đặc trưng phức hợp:** Sự kết hợp giữa các tuần lễ hội (`is_peak_shopping_season`) và ngành hàng cụ thể (`category`, được ánh xạ từ `product_category_name`) tạo ra các ngưỡng phân nhánh có điều kiện cục bộ.

Mục đích của thuật toán **Decision Tree Regressor** trong thiết kế của nhóm:
- Chia tách không gian đặc trưng thành các siêu hình hộp trực giao (Axis-aligned hyper-rectangles) đồng nhất về doanh thu bằng cơ chế đệ quy nhị phân.
- Học các quy tắc quyết định dạng cây dễ diễn giải (*If-Else Rules*), phản ánh chính xác các ngưỡng nhạy cảm về doanh thu của từng nhóm sản phẩm.
- Đóng vai trò là khối xây dựng nền tảng (*base learner*) cho các thuật toán Ensemble Bagging và Boosting.

### 1.2. Cơ sở toán học & Tiêu chuẩn phân chia
Mô hình được thiết kế dựa trên tiêu chuẩn **Giảm phương sai (Variance Reduction)** làm hàm độ lợi tối ưu hóa tại mỗi nút phân nhánh:

Tại một nút $D$ chứa $|D|$ mẫu với biến mục tiêu $y$, phương sai của nút được định nghĩa:

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
- **Ma trận đặc trưng $X$:** Mảng ma trận 2D số thực kích thước $(N, D)$ gồm các đặc trưng số học (trễ, cuộn, giá, lịch) và các đặc trưng phân loại ngành hàng đã qua mã hóa.
- **Vector nhãn mục tiêu $y$:** Vector số thực 1D kích thước $(N,)$ trong không gian log tự nhiên:

$$
y = \log(1 + \text{sales})
$$

### 2.2. Định dạng dữ liệu đầu ra (Output)
- **Dự báo trong không gian log:** Trả về vector dự báo liên tục $\hat{y}$ tương ứng với giá trị trung bình tại nút lá.
- **Hàm chuyển đổi tiền tệ thực tế BRL:**

$$
\hat{y}_{\text{sales}} = \max\left(0.0, \exp(\hat{y}) - 1\right) \text{ BRL}
$$

---

## 3. Data Used (Dữ liệu sử dụng & Ràng buộc thời gian)

### 3.1. Phân chia tập dữ liệu (Temporal Split)
- **Tập Train (khoảng 80% số tuần đầu):** Học cấu trúc cây phân nhánh và giá trị tại các lá.
- **Tập Validation (khoảng 10% số tuần kế tiếp):** Dùng để tinh chỉnh siêu tham số độ sâu và số mẫu tối thiểu tại lá.
- **Tập Test (khoảng 10% số tuần cuối):** Đánh giá kiểm định độc lập cuối cùng.

### 3.2. Bảo toàn tuần doanh thu bằng 0 (Zero-Sales Weeks)
- Với mỗi sản phẩm, toàn bộ các tuần không có đơn hàng từ tuần bán đầu tiên đến hết kỳ quan sát đều được giữ nguyên với `sales = 0.0` BRL để phản ánh đúng mật độ nhu cầu thực tế.

### 3.3. Quy tắc thời gian của đặc trưng
- Toàn bộ các đặc trưng trễ (`last_week_sales`, `sales_lag_2_weeks`, `sales_lag_4_weeks`) và thống kê cuộn (`sales_mean_4_weeks`, `sales_std_4_weeks`) đều chỉ được tính trên cửa sổ thời gian đã kết thúc trước 00:00 thứ Hai của tuần dự báo.

---

## 4. Evaluation (Phương pháp đánh giá)

### 4.1. Hàm mất mát huấn luyện
Trong quá trình xây dựng cây, tiêu chí tối đa hóa độ giảm phương sai tương đương với việc cực tiểu hóa sai số bình phương trung bình (MSE) trên không gian log:

$$
\mathcal{L}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

### 4.2. Bộ chỉ số đo lường trên thang đo tiền tệ thực tế (BRL)
Toàn bộ đánh giá lựa chọn mô hình đều được đo lường trực tiếp trên thang đo tiền tệ BRL:
1. **Root Mean Squared Error (RMSE):** Đơn vị $\text{BRL}$, tiêu chí chính để chọn mô hình tối ưu trên tập Validation.
2. **Mean Absolute Error (MAE):** Đơn vị $\text{BRL}$, đo sai lệch doanh thu trung bình trên mỗi sản phẩm–tuần.
3. **Mean Squared Error (MSE):** Đơn vị $\text{BRL}^2$.
4. **Hệ số xác định ($R^2$ Score):** Đo lường tỷ lệ phương sai doanh thu được giải thích so với mốc Baseline trung bình.
5. **Weighted Absolute Percentage Error (WAPE):** Sai số phần trăm có trọng số.
6. **Cost_Per_Item_BRL:** Độ đo chi phí bất đối xứng giữa dự báo thiếu và dự báo thừa.

---

## 5. Status (Trạng thái cài đặt & Tối ưu kỹ thuật)

- **Đã cài đặt From-Scratch ([`src/models/tree.py`](../../src/models/tree.py)):** Lớp `DecisionTreeRegressorFromScratch` cài đặt đệ quy phân nhánh nhị phân theo mức giảm phương sai. Trong `main.py` Step 6, mô hình mặc định chạy với `max_depth=5, min_samples_split=20, min_samples_leaf=10, n_quantiles=16` (fit `17,51s`), đạt Validation RMSE `61,1664 BRL` và Test RMSE `51,1403 BRL` (`MAE 4,3274 BRL`, $R^2$ `0,0449`, WAPE `1,0113`, Cost `41,2640`).
- **Nguyên tắc dừng cây:** Cây dừng phân nhánh khi đạt độ sâu tối đa `max_depth`, số mẫu tại nút nhỏ hơn `min_samples_split`, phương sai nút nhỏ hơn ngưỡng $10^{-10}$, hoặc không tìm được phép chia nào có $\Delta \text{Var} > 0$.
- **Ràng buộc lá:** Cả hai nút con sau khi chia đều bắt buộc phải thỏa mãn số mẫu tối thiểu `min_samples_leaf`.
- **Tối ưu hóa ngưỡng phân vị (Quantile Split Candidates):**
  - Để tránh duyệt toàn bộ giá trị phân biệt gây chậm, bộ chia lấy mẫu tối đa `n_quantiles=16` phân vị đại diện (từ phân vị `5%` đến `95%`) trên các giá trị phân biệt của từng đặc trưng làm ngưỡng cắt ứng viên.
- **Trích xuất độ quan trọng đặc trưng (Feature Importance):**
  - Tính tổng lượng giảm phương sai có trọng số qua tất cả các nút phân nhánh theo từng đặc trưng $j$:

$$
\text{Importance}(j) = \sum_{t \in \text{splits on } j} \Delta \text{Var}(t) \cdot N_t
$$

---

## 6. Kế Hoạch & Kết Quả Tinh Chỉnh Siêu Tham Số

### 6.1. Không gian tìm kiếm lưới siêu tham số trên tập Validation
Nhóm chạy lưới 6 cấu hình (`main.py` Step 7, cố định `min_samples_split=20, n_quantiles=16`) trên tập Validation độc lập:
- **Độ sâu tối đa (`max_depth`):** $\{3, 5, 8\}$.
- **Số mẫu tối thiểu tại lá (`min_samples_leaf`):** $\{10, 30\}$.

### 6.2. Tiêu chí lựa chọn & kết quả lưới thực tế (`tuning_history.csv`)
- Cấu hình đại diện được chọn theo **Validation RMSE (BRL)** thấp nhất, không dùng tập Test để chọn siêu tham số:

| `max_depth` | `min_samples_leaf` | Validation RMSE (BRL) | Ghi chú |
| --- | --- | --- | --- |
| `3` | `10` | `62.4085` | |
| `3` | `30` | `62.4085` | |
| `5` | `10` | `61.1664` | Trùng cấu hình Step 6 |
| `5` | `30` | `61.1664` | |
| `8` | `10` | **`59.2288`** | Tốt nhất trong lưới tuning (`Tuned Model`) |
| `8` | `30` | `59.3857` | |

- Cấu hình thắng vòng tuning (`max_depth=8, min_samples_leaf=10`) được gán tên **`Tuned Model`** trong bảng tổng hợp; khi đánh giá trên Test đạt RMSE `50,4106 BRL`, MAE `4,2989 BRL`, MSE `2.541,2236`, $R^2$ `0,0719`, WAPE `1,0046`, Cost `40,6158` (xếp thứ 2 toàn bảng sau `Two-Stage Hurdle`).
