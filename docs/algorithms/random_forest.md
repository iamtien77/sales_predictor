# 🌲 Rừng Ngẫu Nhiên Hồi Quy (Random Forest Regressor From-Scratch)

> **Vị trí trong bài lab:** Phương pháp học kết hợp (Ensemble Bagging) nhiều cây quyết định, được nhóm thiết kế nhằm giảm phương sai dự báo và tăng độ ổn định.
> **Mục tiêu trong bài lab:** Thử nghiệm lấy mẫu bootstrap và chọn ngẫu nhiên đặc trưng xem có giúp kết quả dự báo ổn định và chính xác hơn một cây đơn lẻ hay không.

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Một cây quyết định đơn lẻ (Decision Tree) dù có khả năng nắm bắt quan hệ phi tuyến rất tốt nhưng thường có phương sai cao (High Variance) và nhạy cảm với các biến động ngẫu nhiên trong dữ liệu mẫu.

Mục đích của thuật toán **Random Forest Regressor** trong thiết kế của nhóm:
- Ứng dụng kỹ thuật **Ensemble Bagging (Bootstrap Aggregating)** kết hợp lấy mẫu thuộc tính ngẫu nhiên (Feature Subsampling) để xây dựng một tập hợp gồm $B$ cây hồi quy độc lập.
- Giảm thiểu phương sai của mô hình thông qua phép lấy trung bình cộng dự báo từ nhiều cây, giúp hạn chế rủi ro quá khớp trên tập dữ liệu lớn.
- Tăng tính vững chắc (Robustness) khi dự báo doanh thu cho hàng chục nghìn mã sản phẩm có độ phân tán cao.

### 1.2. Cơ sở toán học & Nguyên lý Ensemble Bagging
Thuật toán hoạt động dựa trên ba nguyên lý cốt lõi:

#### A. Lấy mẫu Bootstrap có hoàn lại (Bootstrap Sampling)
Từ tập huấn luyện gốc $D$ có $N$ mẫu, với mỗi cây $b \in \{1, 2, \dots, B\}$, thuật toán rút ngẫu nhiên có hoàn lại $N$ mẫu để tạo ra tập con $D_b$. Về mặt lý thuyết, xác suất một mẫu được chọn vào $D_b$ xấp xỉ:

$$
P = 1 - \left(1 - \frac{1}{N}\right)^N \approx 1 - \frac{1}{e} \approx 63,2\%
$$

Khoảng $36,8\%$ số mẫu còn lại không được chọn tạo thành tập ngoài mẫu (Out-Of-Bag - OOB).

#### B. Phân nhánh ngẫu nhiên đặc trưng (Feature Subsampling)
Tại mỗi nút phân nhánh của từng cây con, thay vì tìm kiếm trên toàn bộ $d$ đặc trưng, thuật toán chỉ chọn ngẫu nhiên một tập con gồm:

$$
k = \lfloor \sqrt{d} \rfloor \text{ đặc trưng}
$$

Điều này làm giảm độ tương quan (Correlation) $\rho$ giữa các cây con, giúp phương sai của giá trị trung bình giảm đi đáng kể theo công thức phương sai tổ hợp:

$$
\text{Var}\left(\frac{1}{B} \sum_{b=1}^{B} T_b(x)\right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2
$$

Khi $\rho \to 0$ (các cây càng ít tương quan với nhau), phương sai toàn hệ thống càng giảm mạnh khi số lượng cây $B$ tăng lên.

#### C. Dự đoán trung bình cộng
Kết quả dự báo cuối cùng trong không gian log là trung bình cộng dự báo của $B$ cây con:

$$
\hat{y}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
$$

---

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
- **Ma trận đặc trưng $X$:** Mảng ma trận 2D số thực kích thước $(N, D)$ gồm các đặc trưng số học đã chuẩn hóa và các đặc trưng phân loại đã mã hóa One-Hot.
- **Vector nhãn mục tiêu $y$:** Vector số thực 1D kích thước $(N,)$ trong không gian log tự nhiên: $y = \log(1 + \text{sales})$.

### 2.2. Định dạng dữ liệu đầu ra (Output)
- **Dự báo trong không gian log:** Trả về vector dự báo liên tục $\hat{y}$ (trung bình cộng log của các cây con).
- **Hàm chuyển đổi tiền tệ thực tế BRL:**

$$
\hat{y}_{\text{sales}} = \max\left(0.0, \exp(\hat{y}) - 1\right) \text{ BRL}
$$

---

## 3. Data Used (Dữ liệu sử dụng & Ràng buộc thời gian)

- **Phân chia Temporal Split 80/10/10:**
  - Tập Huấn luyện (Train Set): Khoảng 80% số tuần đầu, dùng để xây dựng các tập con bootstrap và huấn luyện các cây con.
  - Tập Kiểm định (Validation Set): Khoảng 10% số tuần kế tiếp, dùng để tinh chỉnh siêu tham số số lượng cây và độ sâu.
  - Tập Kiểm thử Độc lập (Test Set): Khoảng 10% số tuần cuối cùng, đánh giá kiểm định độc lập.
- **Bảo toàn tuần doanh thu bằng 0:** Giữ nguyên các tuần `sales = 0.0` BRL để phản ánh đúng cấu trúc thưa thớt của thị trường.
- **Nguyên tắc chống rò rỉ:** Preprocessing chỉ fit trên Train; toàn bộ đặc trưng trễ/cuộn chỉ lấy từ thông tin đã xảy ra trước thời điểm dự báo.

---

## 4. Evaluation (Phương pháp đánh giá)

- **Hàm mục tiêu cây con:** Tối đa hóa lượng giảm phương sai (Variance Reduction) tại mỗi nút con.
- **Tiêu chí tối ưu lưới trên Validation:** Root Mean Squared Error đo bằng tiền tệ BRL (Validation RMSE BRL).
- **Bộ chỉ số đo lường trên thang đo tiền tệ thực tế (BRL):**
  1. Mean Squared Error (MSE): $\text{BRL}^2$.
  2. Mean Absolute Error (MAE): $\text{BRL}$.
  3. Root Mean Squared Error (RMSE): $\text{BRL}$ (tiêu chí chính).
  4. Hệ số xác định ($R^2$ Score).

---

## 5. Status (Trạng thái thiết kế & Tối ưu kỹ thuật)

- **Mô-đun hóa cây thành phần:** Tái sử dụng thiết kế khối cây hồi quy cơ sở làm thành phần con cho rừng ngẫu nhiên.
- **Quản lý trạng thái ngẫu nhiên độc lập:** Mỗi cây con được cấp một hạt giống ngẫu nhiên (seed) độc lập từ bộ tạo số ngẫu nhiên chính, đảm bảo tính tái lập (Reproducibility) và đa dạng hóa quá trình lấy mẫu bootstrap.
- **Trích xuất Feature Importance tổng hợp:**
  - Lấy trung bình cộng độ quan trọng đặc trưng của $B$ cây con:

$$
\text{Importance}(j) = \frac{1}{B} \sum_{b=1}^{B} \text{Importance}_b(j)
$$

---

## 6. Kế Hoạch Tinh Chỉnh Siêu Tham Số (Hyperparameter Tuning Plan)

### 6.1. Không gian tìm kiếm siêu tham số dự kiến
Nhóm thiết lập không gian tìm kiếm siêu tham số trên tập Validation nhằm xác định cấu hình phù hợp:
- Số lượng cây (`n_estimators`): Khảo sát các mức $\{10, 20, 30\}$.
- Độ sâu tối đa của cây con (`max_depth`): Khảo sát các mức $\{5, 8\}$.
- Số mẫu tối thiểu để chia nút (`min_samples_split`): Khảo sát $\{10, 20\}$.
- Tỷ lệ lấy mẫu đặc trưng (`max_features`): Khảo sát quy tắc căn bậc hai $k = \lfloor \sqrt{d} \rfloor$.

### 6.2. Tiêu chí đánh giá và lựa chọn cấu hình
- Toàn bộ các cấu hình trong không gian tìm kiếm được đánh giá độc lập trên tập Validation.
- Cấu hình có sai số **Validation RMSE (BRL)** thấp nhất sau khi chuyển đổi ngược từ logarit sẽ được nhóm lựa chọn để đại diện cho họ mô hình Random Forest.
- Tuyệt đối không sử dụng tập Test để can thiệp hay định hướng quá trình lựa chọn siêu tham số.
