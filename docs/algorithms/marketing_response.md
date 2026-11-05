# 📢 Mô Hình Phản Hồi Tiếp Thị Phi Tuyến (Marketing Diminishing Returns Response Model)

> **Vị trí trong bài lab:** Phần mô phỏng kịch bản tiếp thị, được nhóm thiết kế nhằm khảo sát độ nhạy của doanh thu khi thay đổi mức ngân sách quảng cáo.
> **Lưu ý:** Đây chỉ là phần mô phỏng What-If bằng công thức toán giả định để quan sát xu hướng, không phải mô hình học từ dữ liệu thật vì Olist không có số liệu quảng cáo.

---

## 1. Purpose (Mục đích & Ý nghĩa bài toán)

### 1.1. Thách thức dữ liệu thực tế
Trong thực tế, việc ước lượng tác động của quảng cáo đến doanh số là một nhu cầu quen thuộc. Tuy nhiên, dữ liệu Olist chỉ lưu lịch sử đơn hàng, hoàn toàn không có số liệu về chi phí tiếp thị, lượt hiển thị hay lượt nhấp chuột của từng sản phẩm. Do đó, nhóm không thể huấn luyện một mô hình học máy để học tác động nhân quả từ dữ liệu này.

### 1.2. Hướng tiếp cận: Mô hình phản hồi theo quy luật lợi suất giảm dần
Để phục vụ việc thử nghiệm kịch bản giả định, nhóm chọn mô phỏng theo quy luật **Lợi suất giảm dần (Diminishing Marginal Returns)**:
- **Chế độ cơ sở (Baseline Only):** Giữ nguyên mức doanh thu được dự báo từ mô hình tuần; không giả định thêm mức tăng trưởng nào.
- **Chế độ phản hồi giảm biên (Calibrated Diminishing Returns):** Dùng hàm logarit để mô phỏng mức tăng trưởng doanh thu khi rót thêm ngân sách, với hệ số phản hồi $\beta$ và mức ngân sách chuẩn $R$ được đặt trước theo giả định.

---

## 2. Mathematical Formulation (Cơ sở Toán học & Công thức)

### 2.1. Hàm tăng trưởng theo ngân sách (Log-Response Uplift)
Với ngân sách tiếp thị giả định $A$, mức chi tham chiếu $R$ và hệ số phản hồi $\beta$:

$$
u(A) = 1 + \beta \ln\left(1 + \frac{A}{R}\right), \quad \text{với } A \ge 0, R > 0, \beta \ge 0
$$

Trong đó $R$ là ngân sách tham chiếu (cùng đơn vị BRL với $A$), và $\beta$ là hệ số điều chỉnh độ nhạy của doanh thu theo ngân sách.

### 2.2. Doanh thu dự báo theo kịch bản
Doanh thu theo kịch bản $\hat{S}(A)$ được tính bằng tích số giữa doanh thu cơ sở ban đầu $S_0$ và hệ số tăng trưởng $u(A)$:

$$
\hat{S}(A) = S_0 \cdot \left[1 + \beta \ln\left(1 + \frac{A}{R}\right)\right]
$$

### 2.3. Các chỉ số đo lường hiệu quả mô phỏng
Khi $A > 0$, $S_0 > 0$ và giả định biên lợi nhuận gộp $m \in [0, 1]$:
1. **Doanh thu tăng thêm (Incremental Sales):**

$$
\Delta S = \hat{S}(A) - S_0 = S_0 \cdot \beta \ln\left(1 + \frac{A}{R}\right)
$$

2. **Tỷ lệ tăng trưởng doanh số (Sales Growth %):**

$$
\text{Growth \%} = 100 \cdot \frac{\Delta S}{S_0} = 100 \cdot \beta \ln\left(1 + \frac{A}{R}\right)
$$

3. **Hệ số doanh thu trên chi phí (ROAS):**

$$
\text{ROAS} = \frac{\Delta S}{A}
$$

4. **Tỷ suất sinh lời tiếp thị ước tính (Marketing ROI %):**

$$
\text{Marketing ROI \%} = 100 \cdot \frac{m \cdot \Delta S - A}{A}
$$

---

## 3. Input / Output (Đầu vào & Đầu ra mô phỏng)

### 3.1. Tham số đầu vào
Mô hình mô phỏng tiếp nhận các tham số:
- `base_product`: Dòng dữ liệu của sản phẩm cần dự báo kịch bản.
- `ad_spend_levels`: Danh sách các mức ngân sách cần mô phỏng ($A \ge 0$, đơn vị BRL).
- `elasticity` ($\beta$): Hệ số phản hồi giả định (mặc định $\beta = 0.15$).
- `reference_spend` ($R$): Mức chi tham chiếu (mặc định $R = 100.0$ BRL).
- `gross_margin` ($m$): Tỷ lệ biên lợi nhuận gộp giả định (mặc định $m = 1.0$).

### 3.2. Cấu trúc kết quả mô phỏng
Bảng kết quả kịch bản bao gồm các cột thông tin:
- `Ad_Spend_BRL`: Mức ngân sách tiếp thị đưa vào.
- `Baseline_Sales_BRL`: Doanh thu dự báo cơ sở $S_0$.
- `Predicted_Sales_BRL`: Doanh thu dự báo theo kịch bản $\hat{S}(A)$.
- `Incremental_Sales_BRL`: Doanh thu tăng thêm $\Delta S$.
- `Sales_Growth_%`: Tỷ lệ tăng trưởng doanh thu so với cơ sở.
- `ROAS`: Doanh thu tăng thêm thu được trên mỗi đồng chi phí.
- `Marketing_ROI_%`: Tỷ suất sinh lời ước tính.

---

## 4. Mathematical Properties & Boundary Conditions (Tính chất Toán học & Điều kiện Biên)

Hàm mô phỏng được xây dựng thỏa mãn các tính chất tự nhiên:
1. **Khi không chi tiền quảng cáo ($A = 0$):** Hệ số tăng trưởng $u(0) = 1$, doanh thu tăng thêm $\Delta S(0) = 0$ BRL.
2. **Quy luật lợi suất giảm dần:** Đạo hàm bậc nhất và bậc hai của doanh thu tăng thêm theo ngân sách:

$$
\frac{\partial \Delta S}{\partial A} = \frac{S_0 \beta}{R + A} \ge 0, \qquad \frac{\partial^2 \Delta S}{\partial A^2} = -\frac{S_0 \beta}{(R + A)^2} \le 0
$$

   Điều này phản ánh thực tế rằng khi tiếp tục tăng ngân sách, doanh thu vẫn tăng nhưng tốc độ tăng chậm dần do thị trường dần bão hòa.
3. **Khi doanh thu cơ sở bằng 0 ($S_0 = 0$):** Nếu sản phẩm dự báo không bán được trong tuần, mô hình giữ nguyên doanh thu bằng 0 ở mọi mức ngân sách, phản ánh giả định rằng quảng cáo không thể tạo ra doanh số nếu sản phẩm không có nhu cầu cơ bản.

---

## 5. Scope & Notes (Phạm vi Áp dụng & Lưu ý)

- **Mục đích sử dụng:** Phục vụ trực quan hóa các kịch bản giả định What-If trên giao diện ứng dụng, giúp người dùng quan sát xu hướng biến thiên của doanh thu khi thay đổi ngân sách.
- **Lưu ý:** Nhóm nhấn mạnh rằng đây chỉ là phần mô phỏng What-If trên các tham số toán học giả định, không phải kết quả học máy từ dữ liệu Olist, và không dùng để đánh giá độ chính xác của bài lab.
