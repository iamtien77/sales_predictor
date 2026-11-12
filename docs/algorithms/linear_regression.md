# 📉 Hồi Quy Tuyến Tính (Linear Regression From-Scratch)

> **Vị trí trong bài lab:** Mô hình hồi quy tuyến tính cơ sở (Parametric Baseline), được nhóm thiết kế nhằm xem xét mức độ tương quan tuyến tính trước khi thử nghiệm các mô hình phi tuyến.
> **Mục tiêu trong bài lab:** Kiểm tra xem mối quan hệ tuyến tính đơn giản có đủ tốt không trước khi dùng các mô hình cây phức tạp hơn.

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Theo nguyên tắc cơ bản trong Học Máy: luôn thử nghiệm mô hình cơ sở đơn giản trước khi xây dựng các mô hình phức tạp.

Mục đích của thuật toán **Linear Regression** trong bài lab:
- Thiết lập đường cơ sở tham số (Parametric Baseline) để đo lường mức độ tương quan tuyến tính giữa các đặc trưng đầu vào với biến mục tiêu doanh thu.
- Kiểm tra giả định tuyến tính (Linearity Assumption): Liệu tổng doanh thu tuần có thể được ước lượng thỏa đáng bằng một tổ hợp tuyến tính có trọng số của giá, các tuần trễ và danh mục hàng hóa hay không?
- Đánh giá khả năng mở rộng và tốc độ tính toán nghiệm ma trận giải tích đóng (*Analytical Closed-form Solution*) trên tập dữ liệu bảng lớn có nhiều biến giả phân loại.

### 1.2. Cơ sở toán học & Các bộ giải (Solvers)
Mô hình giả định mối quan hệ giữa vector đặc trưng $x \in \mathbb{R}^D$ và biến mục tiêu log doanh thu $y \in \mathbb{R}$ có dạng tuyến tính:

$$
\hat{y} = w^\top x + b
$$

Để tính toán đồng thời cả trọng số $w$ và hệ số chặn (bias) $b$, ma trận dữ liệu được mở rộng thêm một cột toàn số 1: $\tilde{X} = [\mathbf{1}, X] \in \mathbb{R}^{N \times (D+1)}$ và vector tham số $\tilde{w} = [b, w_1, \dots, w_D]^\top$.

Nhóm thiết kế hai bộ giải tối ưu:

#### A. Bộ giải Phương trình Chuẩn tắc (Normal Equation OLS)
Cực tiểu hóa hàm mất mát bình phương tối thiểu (Ordinary Least Squares - OLS) có bổ sung điều chuẩn Ridge nhỏ ($\alpha_{\text{ridge}} = 10^{-5}$) nhằm chống suy biến ma trận và hiện tượng đa cộng tuyến:

$$
J(\tilde{w}) = \frac{1}{N} \left(\|\tilde{X} \tilde{w} - y\|_2^2 + \alpha_{\text{ridge}} \|w\|_2^2\right)
$$

Nghiệm giải tích đóng được xác định bằng giải thuật giả nghịch đảo Moore-Penrose:

$$
\tilde{w}^* = \left( \tilde{X}^\top \tilde{X} + \alpha_{\text{ridge}} I_0 \right)^{+} \tilde{X}^\top y
$$

Trong đó $I_0$ là ma trận đơn vị có phần tử đầu tiên ứng với hệ số chặn bằng 0 (không phạt điều chuẩn lên hệ số chặn $b$). Có thể cân nhắc biểu diễn ma trận thưa khi nhiều đặc trưng bằng 0; lợi ích cần được kiểm tra.

#### B. Bộ giải Hạ độ dốc (Gradient Descent)
Khởi tạo từng trọng số từ phân phối chuẩn có trung bình 0, độ lệch chuẩn 0,01 ($w_j^{(0)} \sim \mathcal{N}(0, 0.01^2)$) và cập nhật lặp qua $T$ vòng với tốc độ học $\eta$:

$$
\nabla_{\tilde{w}} J = \frac{2}{N} \tilde{X}^\top (\tilde{X} \tilde{w} - y) + \frac{2\alpha_{\text{ridge}}}{N} \begin{bmatrix} 0 \\ w \end{bmatrix}
$$

$$
\tilde{w}^{(t+1)} = \tilde{w}^{(t)} - \eta \cdot \nabla_{\tilde{w}} J
$$

Cả hai solver dùng cùng quy ước $\alpha_{\text{ridge}}$ và không phạt intercept. Khi Gradient Descent hội tụ, nghiệm phải khớp Normal Equation trong sai số số học cho phép.

---

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
- **Ma trận đặc trưng $X$:** Mảng ma trận 2D số thực kích thước $(N, D)$ bao gồm các đặc trưng số học đã chuẩn hóa Z-Score và các đặc trưng phân loại đã mã hóa One-Hot.
- **Vector nhãn mục tiêu $y$:** Vector số thực 1D kích thước $(N,)$ trong không gian log tự nhiên: $y = \log(1 + \text{sales})$.

### 2.2. Định dạng dữ liệu đầu ra (Output)
- **Dự báo trong không gian log:** Trả về vector dự báo liên tục $\hat{y}$ (có áp dụng cơ chế Kẹp biên Thích ứng).
- **Hàm chuyển đổi tiền tệ thực tế BRL:**

$$
\hat{y}_{\text{sales}} = \max\left(0.0, \exp(\hat{y}) - 1\right) \text{ BRL}
$$

---

## 3. Data Used (Tập dữ liệu sử dụng & Nguyên tắc chia tập)

- **Chiến lược phân chia:** Phân chia chuỗi thời gian tuần lịch (Temporal Split 80/10/10 tính theo số tuần, không theo số dòng):
  - **Tập Train (khoảng 80% số tuần đầu):** Dùng để học tham số $\tilde{w}$ và học tham số tiền xử lý.
  - **Tập Validation (khoảng 10% số tuần tiếp theo):** Dùng để tinh chỉnh siêu tham số và đánh giá mốc so sánh.
  - **Tập Test (khoảng 10% số tuần cuối cùng):** Đánh giá kiểm định độc lập.
- **Nguyên tắc chống rò rỉ dữ liệu:** Tiền xử lý (imputation, encoding, scaling) chỉ học tham số từ Train rồi áp dụng cho Validation/Test. Không sử dụng bất kỳ thông tin nào của tập tương lai để tính toán trọng số.

---

## 4. Evaluation (Phương pháp đánh giá)

Protocol đánh giá của nhóm chọn mô hình bằng Validation RMSE BRL trước khi đánh giá cuối cùng trên Test:
- **Hàm mất mát huấn luyện:** MSE trên không gian log tự nhiên.
- **Bộ chỉ số đo lường trên thang đo tiền tệ thực tế (BRL):**
  1. Mean Squared Error (MSE): $\text{BRL}^2$.
  2. Mean Absolute Error (MAE): $\text{BRL}$.
  3. Root Mean Squared Error (RMSE): $\text{BRL}$ (tiêu chí đánh giá chính).
  4. Hệ số xác định ($R^2$ Score).
  5. Weighted Absolute Percentage Error (WAPE).
  6. Điểm phạt thiếu/thừa giả định: Điểm phạt theo trọng số giả định, không phải chi phí vận hành đo được.

---

## 5. Design (Phương án giải và kiểm soát số học)

- **Hai cách giải:** Dự kiến đối chiếu nghiệm giả nghịch đảo với Gradient Descent trên cùng dữ liệu và quy ước điều chuẩn.
- **Tài nguyên tính toán:** Cân nhắc biểu diễn ma trận thưa khi nhiều đặc trưng bằng 0; cần kiểm tra mức tiết kiệm thay vì giả định trước.
- **Giới hạn dự báo:** Dự kiến chặn dự báo log trước phép mũ, đồng thời đánh giá nguy cơ dự báo thiếu ở doanh thu ngoài miền Train.

---

## 6. Thiết Kế Cơ Chế Ổn Định Số Học & Kẹp Biên (Numerical Stability & Bounding Design)

### 6.1. Kiểm soát đòn bẩy hàm mũ khi nghịch đảo Log
Mô hình hồi quy tuyến tính tối ưu hóa hàm mất mát trên không gian log tự nhiên: $\hat{y} = w^\top x + b$. Do siêu phẳng tuyến tính không có biên chặn trên tự nhiên ($\hat{y} \in (-\infty, +\infty)$), khi áp dụng hàm chuyển đổi ngược về thang tiền tệ BRL qua $\hat{y}_{\text{sales}} = \exp(\hat{y}) - 1$, các giá trị dự báo log lớn có thể bị khuếch đại theo cấp số nhân, dẫn đến nguy cơ sai số bình phương bị phóng đại rất lớn.

### 6.2. Thiết kế Cơ chế Kẹp biên Thích ứng (Adaptive Numerical Clamping)
Để kiểm soát số học khi dự báo, nhóm dự kiến khảo sát cơ chế kẹp biên dựa trên miền giá trị quan sát của tập Train:
- **Xác định biên an toàn từ Train:** Trong quá trình huấn luyện, ghi nhận giá trị nhãn nhỏ nhất $y_{\min}$ và lớn nhất $y_{\max}$ của tập Train, cùng độ trải $\text{span} = \max(1, y_{\max} - y_{\min})$.
- **Quy tắc kẹp biên:** Khi thực hiện dự báo, mọi giá trị dự báo log vượt khỏi khoảng dung sai cho phép sẽ được kẹp lại:

$$
\hat{y}_{\text{safe}} = \text{clip}\left(\hat{y}, y_{\min} - 0.5 \cdot \text{span}, y_{\max} + 0.5 \cdot \text{span}\right)
$$

Cận rộng vẫn có thể cho sai số BRL lớn sau phép mũ; cận hẹp có thể hạn chế dự báo doanh thu cao. Đây là đánh đổi cần khảo sát trên Validation, không bảo đảm chất lượng dự báo.

### 6.3. Kế hoạch kiểm tra và chẩn đoán
- **Hai thang đo:** Quan sát sai số log để hiểu quá trình học, nhưng chọn mô hình bằng Validation RMSE trên BRL.
- **Ổn định số học:** Xem số điều kiện ma trận Gram, thử mức điều chuẩn đặt trước và kiểm tra sự nhất quán của hai cách giải.
- **Ngoại suy:** Xem mẫu sai số lớn, tỷ lệ dự báo bị chặn và đóng góp vào tổng sai số bình phương.
- **Giữ vai trò Test:** Khảo sát cấu hình trên Train/Validation; đánh giá Test sau lựa chọn và ghi nhận hạn chế.
