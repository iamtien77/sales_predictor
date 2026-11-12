# 🌳 Cây Quyết Định Hồi Quy (Decision Tree Regressor From-Scratch)

> **Vị trí trong bài lab:** Mô hình hồi quy phi tuyến tính dạng cây, được nhóm thiết kế nhằm học các mối quan hệ phi tuyến trong dữ liệu thương mại điện tử. Đây đồng thời là khối cơ sở cho các thuật toán Ensemble (Random Forest, Gradient Boosting) và mô hình Two-Stage Hurdle.
> **Mục tiêu trong bài lab:** Thử nghiệm mô hình cây quyết định dạng quy tắc If-Else đơn giản, dễ giải thích và xem có cải thiện hơn so với các baseline cơ sở hay không.

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong bài toán dự báo doanh thu bán lẻ chuỗi thời gian theo tuần trên sàn thương mại điện tử Olist, dữ liệu mang các đặc thù phân phối phức tạp:
- **Quan hệ phi tuyến:** Doanh thu có thể liên hệ với giá, lịch sử bán và đánh giá theo cách không tuyến tính.
- **Dữ liệu thưa và nhiều số 0 (Zero-Inflated Demand):** Đa số sản phẩm có nhiều tuần không phát sinh đơn hàng (doanh thu bằng 0), khiến mô hình tuyến tính dễ bị nhiễu cục bộ.
- **Tương tác đặc trưng:** Dự kiến xem xét tương tác giữa lịch, ngành hàng và lịch sử bán qua các nhánh cây.

Mục đích của thuật toán **Decision Tree Regressor** trong thiết kế của nhóm:
- Chia tách không gian đặc trưng thành các siêu hình hộp trực giao (Axis-aligned hyper-rectangles) đồng nhất về doanh thu bằng cơ chế đệ quy nhị phân.
- Học các quy tắc quyết định dạng cây dễ diễn giải (*If-Else Rules*), để quan sát các ngưỡng được học từ dữ liệu.
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
- Giữ tuần không bán với doanh thu bằng 0 trong khoảng quan sát đầy đủ. Tuần bán đầu làm lịch sử, mẫu dự báo bắt đầu từ tuần sau. Thiếu dữ liệu tồn kho nên doanh thu chưa thể xem là toàn bộ nhu cầu.

### 3.3. Quy tắc thời gian của đặc trưng
- Doanh thu trễ 1, 2, 4 tuần và thống kê trượt bốn tuần chỉ dùng thông tin trước 00:00 thứ Hai của tuần đích. Tham số điền thiếu và mã hóa được học từ Train.

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
4. **Hệ số xác định ($R^2$ Score):** Đo lường tỷ lệ phương sai doanh thu được giải thích so với dự báo bằng trung bình của chính tập đánh giá.
5. **Weighted Absolute Percentage Error (WAPE):** Sai số phần trăm có trọng số.
6. **Điểm phạt thiếu/thừa giả định:** Điểm phạt theo trọng số giả định, không phải chi phí vận hành đo được, cho dự báo thiếu và thừa.

---

## 5. Design (Phương án cây và giới hạn tính toán)

- **Phương án cây:** Chia nhị phân theo mức giảm phương sai, dùng trung bình nhãn log tại lá để dự báo.
- **Nguyên tắc dừng cây:** Cây dừng phân nhánh khi đạt độ sâu tối đa độ sâu tối đa, số mẫu tại nút nhỏ hơn số mẫu tối thiểu để chia nút, phương sai nút nhỏ hơn ngưỡng $10^{-10}$, hoặc không tìm được phép chia nào có $\Delta \text{Var} > 0$.
- **Ràng buộc lá:** Cả hai nút con sau khi chia đều bắt buộc phải thỏa mãn số mẫu tối thiểu số mẫu tối thiểu tại lá.
- **Tối ưu hóa ngưỡng phân vị (Quantile Split Candidates):**
  - Khi có nhiều giá trị phân biệt, dự kiến thử 16 phân vị từ 5% đến 95%; khi ít giá trị, thử trung điểm hai giá trị liên tiếp. Giới hạn số ứng viên có thể bỏ qua ngưỡng chia tốt hơn.
- **Trích xuất độ quan trọng đặc trưng (Feature Importance):**
  - Tính tổng lượng giảm phương sai có trọng số qua tất cả các nút phân nhánh theo từng đặc trưng $j$:

$$
\text{Importance}(j) = \sum_{t \in \text{splits on } j} \Delta \text{Var}(t) \cdot N_t
$$

---

## 6. Kế Hoạch Tinh Chỉnh Siêu Tham Số

### 6.1. Không gian tìm kiếm lưới siêu tham số trên tập Validation
Nhóm dự kiến khảo sát sáu cấu hình trên Validation, giữ số mẫu tối thiểu để chia nút là 20 và quy tắc chọn ngưỡng như mục 5:
- **Độ sâu tối đa (độ sâu tối đa):** $\{3, 5, 8\}$.
- **Số mẫu tối thiểu tại lá (số mẫu tối thiểu tại lá):** $\{10, 30\}$.

### 6.2. Tiêu chí lựa chọn và kế hoạch đối chiếu
- Cấu hình đại diện được chọn theo **Validation RMSE (BRL)** thấp nhất, không dùng tập Test để chọn siêu tham số:

| Độ sâu tối đa | Số mẫu tối thiểu tại lá | Chỉ số lựa chọn | Nội dung đối chiếu |
| --- | --- | --- | --- |
| 3 | 10 | Validation RMSE | Cây nông, lá nhỏ |
| 3 | 30 | Validation RMSE | Tăng kích thước lá |
| 5 | 10 | Validation RMSE | Tăng độ sâu |
| 5 | 30 | Validation RMSE | Độ sâu vừa và lá lớn |
| 8 | 10 | Validation RMSE | Rủi ro quá khớp khi cây sâu hơn |
| 8 | 30 | Validation RMSE | Tác dụng của lá lớn với cây sâu |

- Sau khi chọn cấu hình, so sánh ứng viên với baseline và các họ mô hình khác trên Validation. Chưa xác định cấu hình thắng trước khi đánh giá.
