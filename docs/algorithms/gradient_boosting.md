# ⚡ Tăng Cường Độ Dốc Hồi Quy (Gradient Boosting Regressor From-Scratch)

> **Vị trí trong bài lab:** Phương pháp học kết hợp tuần tự (Ensemble Boosting), được nhóm thiết kế nhằm huấn luyện chuỗi các cây quyết định nông bù trừ vào phần dư sai số.
> **Mục tiêu trong bài lab:** Thử nghiệm cơ chế học sửa sai từng bước kết hợp tốc độ học (learning rate) để xem có cải thiện dần sai số so với cây đơn lẻ hay không.

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong khi Bagging (như Random Forest) tập trung vào việc giảm phương sai bằng cách huấn luyện các cây độc lập, **Gradient Boosting** hướng tới mục tiêu tối ưu hóa mức độ khớp hàm bằng cách học tuần tự trên sai số còn tồn đọng của các thế hệ cây trước.

Mục đích của thuật toán Gradient Boosting trong thiết kế của nhóm:
- Học xấp xỉ hàm phi tuyến tính phức tạp bằng chuỗi cộng gộp các cây quyết định nông (*weak learners* / *shallow trees*).
- Học phần dư của mô hình tích lũy tại mỗi bước lặp; với mất mát bình phương, phần dư chính là gradient âm.
- Ứng dụng hệ số học co hẹp (*Learning Rate Shrinkage*) để kiểm soát tốc độ hội tụ, giảm thiểu nguy cơ quá khớp trên dữ liệu chuỗi thời gian bán lẻ có độ nhiễu cao.

### 1.2. Cơ sở toán học & Thuật toán huấn luyện tuần tự

#### A. Hàm mất mát tối ưu (Loss Function)
Mô hình hoạt động trong không gian log tự nhiên với hàm mất mát bình phương (Squared Error Loss):

$$
L(y, F(x)) = \frac{1}{2} \left( y - F(x) \right)^2
$$

Đạo hàm riêng (Gradient) âm của hàm mất mát theo giá trị dự báo hiện tại chính là phần dư (Residual):

$$
r_i = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}} = y_i - F_{m-1}(x_i)
$$

#### B. Quy trình thuật toán tuần tự $M$ vòng lặp
1. **Khởi tạo giá trị dự báo cơ sở $F_0(x)$:** Giá trị tối ưu là trung bình cộng của nhãn huấn luyện trên tập Train:

$$
   F_0(x) = \arg\min_{\gamma} \sum_{i=1}^{N} L(y_i, \gamma) = \frac{1}{N} \sum_{i=1}^{N} y_i = \bar{y}_{\text{train}}
$$

2. **Vòng lặp $m = 1, 2, \dots, M$ (với $M$ là số lượng cây số cây):**
   - **Bước 2.1 (Tính phần dư):** Với mỗi mẫu $i \in \{1, \dots, N\}$, tính phần dư bằng negative gradient:

$$
     r_{im} = y_i - F_{m-1}(x_i)
$$

   - **Bước 2.2 (Huấn luyện cây con trên phần dư):** Khớp một cây quyết định hồi quy nông $h_m(x)$ (thiết kế với độ sâu tối đa 3) trên tập dữ liệu phần dư $\{(x_i, r_{im})\}_{i=1}^N$.
   - **Bước 2.3 (Cập nhật mô hình tích lũy với hệ số co hẹp $\eta$):**

$$
     F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)
$$

     Trong đó $\eta \in (0, 1]$ là tốc độ học (hệ số co hẹp), giúp kiểm soát đóng góp của từng cây con và chống phân kỳ.

3. **Dự báo cuối cùng:**

$$
   \hat{y}(x) = F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)
$$

---

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
- **Ma trận đặc trưng $X$:** Mảng ma trận 2D số thực kích thước $(N, D)$ gồm các đặc trưng số học đã chuẩn hóa Z-Score và các cột danh mục sản phẩm đã mã hóa One-Hot.
- **Vector nhãn mục tiêu $y$:** Vector số thực 1D kích thước $(N,)$ trong không gian log tự nhiên: $y = \log(1 + \text{sales})$.

### 2.2. Định dạng dữ liệu đầu ra (Output)
- **Dự báo trong không gian log:** Trả về vector dự báo liên tục $\hat{y} = F_M(X)$.
- **Hàm chuyển đổi tiền tệ thực tế BRL:**

$$
\hat{y}_{\text{sales}} = \max\left(0.0, \exp(\hat{y}) - 1\right) \text{ BRL}
$$

---

## 3. Data Used (Dữ liệu sử dụng & Ràng buộc thời gian)

- **Phân chia Temporal Split 80/10/10:**
  - Tập Huấn luyện (Train Set): Khoảng 80% số tuần đầu, dùng để học tuần tự chuỗi cây trên phần dư.
  - Tập Kiểm định (Validation Set): Khoảng 10% số tuần kế tiếp, dùng để chọn số lượng cây $M$ và độ sâu cây.
  - Tập Kiểm thử Độc lập (Test Set): Khoảng 10% số tuần cuối, dùng để đánh giá cuối cùng.
- **Bảo toàn tuần doanh thu bằng 0:** Giữ nguyên các tuần có doanh thu bằng 0 BRL để phản ánh đúng cấu trúc dữ liệu thương mại điện tử thực tế.
- **Nguyên tắc chống rò rỉ:** Tham số tiền xử lý chỉ học từ Train; toàn bộ đặc trưng trễ/cuộn chỉ lấy từ thông tin đã xảy ra trước thời điểm dự báo.

---

## 4. Evaluation (Phương pháp đánh giá)

- **Hàm mất mát tối ưu:** L2 Loss (Mean Squared Error) trên không gian log tự nhiên: $\mathcal{L} = \frac{1}{N} \sum (y_i - \hat{y}_i)^2$.
- **Lựa chọn siêu tham số:** Đánh giá trên tập Validation độc lập bằng thang đo RMSE tiền tệ BRL.
- **Bộ chỉ số đo lường trên thang đo BRL:**
  1. Mean Squared Error (MSE): $\text{BRL}^2$.
  2. Mean Absolute Error (MAE): $\text{BRL}$.
  3. Root Mean Squared Error (RMSE): $\text{BRL}$ (tiêu chí chính).
  4. Hệ số xác định ($R^2$ Score).
  5. Weighted Absolute Percentage Error (WAPE).
  6. Điểm phạt thiếu/thừa giả định: Điểm phạt theo trọng số giả định, không phải chi phí vận hành đo được.

---

## 5. Design (Phương án học và theo dõi mô hình)

- **Phương án học:** Dùng cây hồi quy nông học phần dư rồi cộng dự báo cây mới với hệ số co hẹp.
- **Kiến trúc cây nông (độ sâu tối đa 3):** Độ sâu nông cho từng cây con giúp mỗi cây chỉ đóng vai trò "người học yếu" (*weak learner*), hạn chế quá khớp sớm.
- **Theo dõi hội tụ hàm mất mát & Độ quan trọng đặc trưng:** Lưu giá trị MSE phần dư qua từng vòng lặp $m$ và tính trung bình cộng lượng giảm phương sai của $M$ cây con:

$$
\text{Importance}(j) = \frac{1}{M} \sum_{m=1}^{M} \text{Importance}_{h_m}(j)
$$

---

## 6. Kế Hoạch Tinh Chỉnh Siêu Tham Số

### 6.1. Không gian tinh chỉnh siêu tham số dự kiến
Nhóm dự kiến bắt đầu với bốn cấu hình trên Validation:
- Số cây: $\{15, 30\}$.
- Độ sâu tối đa: $\{2, 3\}$.
- Tốc độ học: giữ mức $0.1$ trong phép so sánh này.

### 6.2. Tiêu chí lựa chọn và kế hoạch đối chiếu
Các cấu hình được so sánh bằng Validation RMSE trên BRL. Bảng sau là kế hoạch thử, chưa ghi kết quả:

| Số cây | Độ sâu tối đa | Tốc độ học | Chỉ số lựa chọn | Nội dung đối chiếu |
| --- | --- | --- | --- | --- |
| 15 | 2 | 0.1 | Validation RMSE | Mốc ít cây và nông |
| 15 | 3 | 0.1 | Validation RMSE | Ảnh hưởng của độ sâu |
| 30 | 2 | 0.1 | Validation RMSE | Ảnh hưởng của số cây |
| 30 | 3 | 0.1 | Validation RMSE | Kết hợp tăng số cây và độ sâu |

Ghi nhận thêm thời gian huấn luyện để đối chiếu với thay đổi sai số. Chọn cấu hình bằng Validation, không dùng Test để chọn lại.
