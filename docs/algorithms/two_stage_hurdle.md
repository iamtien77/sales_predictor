# 🎯 Mô Hình Hai Giai Đoạn (Two-Stage Hurdle Regressor From-Scratch)

> **Vị trí trong bài lab:** Kiến trúc học máy hai giai đoạn, được nhóm thiết kế nhằm xử lý hiện tượng dữ liệu có quá nhiều tuần bằng 0 (Zero-Inflated Demand).
> **Mục tiêu trong bài lab:** Tách riêng hai bài toán nhỏ: dự đoán xác suất tuần này có bán được hàng hay không, và nếu có thì ước lượng mức doanh thu tương ứng.

---

## 1. Purpose (Mục đích & Động lực lựa chọn)

### 1.1. Mục đích bài toán
Trong bài toán dự báo chuỗi thời gian bán lẻ theo từng cặp Sản phẩm – Tuần (Weekly Product Sales Panel) trên sàn thương mại điện tử Olist:
- **Đặc thù dữ liệu nhiều số 0:** Bảng tuần có thể chứa nhiều mẫu doanh thu bằng **0 BRL**. Thiết kế cần giữ các mẫu này để học cả tuần có bán và không bán.
- **Xung đột khi học chung:** Nếu huấn luyện một mô hình hồi quy duy nhất trên toàn bộ dữ liệu, nhiều mẫu 0 có thể kéo dự báo về gần 0, khiến mô hình khó ước lượng những tuần có doanh thu dương. Đây là lý do cần thử cách tách hai giai đoạn.
- **Ý tưởng phân tách bài toán:** Bài toán được chia thành hai câu hỏi:
  1. *Tuần này sản phẩm có bán được hàng không?* (bài toán phân loại nhị phân có/không).
  2. *Nếu bán được thì doanh thu là bao nhiêu?* (bài toán hồi quy trên các mẫu có phát sinh đơn).

### 1.2. Cơ sở toán học & Kiến trúc Hai giai đoạn (Hurdle Architecture)
Mô hình Hurdle tách bài toán thành hai giai đoạn:

```
                       ┌─────────────────────────┐
                       │  Ma trận đặc trưng x    │
                       └────────────┬────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌───────────────────────────────┐        ┌───────────────────────────────────────┐
│   Giai đoạn 1 (Occurrence)    │        │         Giai đoạn 2 (Severity)        │
│   Ước lượng Xác suất có đơn   │        │     Hồi quy Doanh thu có điều kiện    │
│    p(x) = P(sales > 0 | x)    │        │      s_pos(x) = max(0, exp(y+) - 1)   │
└──────────────┬────────────────┘        └───────────────────┬───────────────────┘
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │    Dự báo Hợp nhất        │
                        │ s_hat(x) = p(x) × s_pos(x) │
                        └───────────────────────────┘
```

#### A. Giai đoạn 1: Mô hình Ước lượng Xác suất phát sinh giao dịch (Occurrence Stage)
- **Tập dữ liệu huấn luyện:** Toàn bộ mẫu huấn luyện với nhãn nhị phân:

$$
  z_i = \mathbb{I}(y_i > 0) = \begin{cases} 1 & \text{nếu } y_i > 0 \\ 0 & \text{nếu } y_i = 0 \end{cases}
$$

- **Bộ phân loại:** Cây hồi quy học nhãn 0/1, với độ sâu và số mẫu tối thiểu tại lá xác định trước. Tại mỗi nút lá $L$, giá trị trung bình mẫu chính là xác suất thực nghiệm rơi vào lớp dương:

$$
  p(x) = \hat{P}(\text{sales} > 0 \mid x) = \frac{1}{|L|} \sum_{i \in L} z_i \in [0, 1]
$$

#### B. Giai đoạn 2: Mô hình Hồi quy Doanh thu có điều kiện (Severity Stage)
- **Tập dữ liệu huấn luyện:** Chỉ bao gồm các mẫu **thực sự phát sinh đơn hàng** ($y_i > 0$):

$$
  \mathcal{D}_{\text{pos}} = \{(x_i, y_i) \mid y_i > 0\}
$$

- **Bộ hồi quy:** Cây quyết định hồi quy với độ sâu và số mẫu tối thiểu tại lá xác định trước học trên nhãn log của tập mẫu dương, không đưa mẫu doanh thu bằng 0 vào nhánh hồi quy dương.
- **Giá trị dự báo tiền tệ có điều kiện:**

$$
  \hat{s}_{\text{pos}}(x) = \max\left(0.0, \exp(\hat{y}_{\text{pos}}(x)) - 1\right)
$$

#### C. Tổng hợp Kỳ vọng Toán học (Expected Revenue Combination)
Kỳ vọng doanh thu được xác định theo công thức xác suất toàn phần:

$$
\mathbb{E}[\text{sales} \mid x] = P(\text{sales} > 0 \mid x) \times \mathbb{E}[\text{sales} \mid \text{sales} > 0, x]
$$

Công thức kỳ vọng trên là cơ sở cho cách kết hợp hai nhánh. Tuy nhiên, vì nhánh dương học trên log, nghịch đảo dự báo của nó không mặc nhiên bằng kỳ vọng doanh thu dương. Giá trị dự báo đề xuất là:

$$
\hat{s}(x) = p(x) \cdot \hat{s}_{\text{pos}}(x)
$$

Và được chuyển đổi về không gian log để giữ giao diện thống nhất với các mô hình khác:

$$
\hat{y} = \log\left(1 + p(x) \cdot \hat{s}_{\text{pos}}(x)\right)
$$

---

## 2. Input / Output (Đầu vào & Đầu ra)

### 2.1. Định dạng dữ liệu đầu vào (Input)
- **Ma trận đặc trưng $X$:** Mảng ma trận 2D số thực kích thước $(N, D)$ gồm các đặc trưng số học (trễ, cuộn, giá, lịch) và các đặc trưng phân loại ngành hàng.
- **Vector nhãn mục tiêu $y$:** Vector số thực 1D kích thước $(N,)$ trong không gian log tự nhiên: $y = \log(1 + \text{sales})$.

### 2.2. Định dạng dữ liệu đầu ra (Output)
- **Dự báo hợp nhất trong không gian log:** Trả về vector $\hat{y} = \log(1 + \hat{s}(x))$.
- **Đầu ra mở rộng:**
  - Xác suất phát sinh giao dịch: $p(x) \in [0, 1]$.
  - Mức doanh thu BRL dự kiến trong tình huống có người mua: $\hat{s}_{\text{pos}}(x)$.
- **Chuyển đổi tiền tệ thực tế BRL:**

$$
\hat{y}_{\text{money}} = \max\left(0.0, \exp(\hat{y}) - 1\right) \text{ BRL}
$$

---

## 3. Data Used (Tập dữ liệu sử dụng & Đặc thù phân phối)

- **Chiến lược phân chia:** Phân chia chuỗi thời gian theo tuần lịch (Temporal Split 80/10/10 tính theo số tuần):
  - **Tập Train (khoảng 80% số tuần đầu):** Dùng để học cây Occurrence trên toàn bộ tập và học cây Severity trên tập mẫu dương.
  - **Tập Validation (khoảng 10% số tuần tiếp theo):** Dùng để tinh chỉnh siêu tham số và đánh giá mốc so sánh.
  - **Tập Test (khoảng 10% số tuần cuối):** Đánh giá kiểm định độc lập.
- **Nguyên tắc chống rò rỉ:** Chia theo thời gian trước khi học thống kê điền thiếu, mã hóa và chuẩn hóa. Tập con mẫu dương $\mathcal{D}_{\text{pos}}$ chỉ được trích xuất từ tập Train; tuyệt đối không lấy mẫu từ Validation hay Test để huấn luyện nhánh Severity.

---

## 4. Evaluation (Phương pháp đánh giá)

Hệ thống đánh giá trên thang đo tiền tệ thực tế BRL với đầy đủ bộ chỉ số:
1. **Root Mean Squared Error (RMSE):** Đo lường căn của trung bình bình phương sai số trên đơn vị BRL (tiêu chí chính).
2. **Mean Squared Error (MSE):** Thang đo phạt các sai số cực đoan ($\text{BRL}^2$).
3. **Mean Absolute Error (MAE):** Đo lường sai lệch tuyệt đối trung bình (BRL).
4. **Hệ số xác định ($R^2$ Score):** Đo lường tỷ lệ phương sai doanh thu được giải thích so với mốc trung bình.
5. **Weighted Absolute Percentage Error (WAPE):** dùng doanh thu thực tế $s_i$ và dự báo BRL $\hat{s}_i$; cần ghi rõ quy ước nếu tổng doanh thu bằng 0.

$$
\text{WAPE} = \frac{\sum_{i=1}^N |s_i - \hat{s}_i|}{\sum_{i=1}^N s_i}
$$

---

## 5. Status (Trạng thái thiết kế & Tối ưu kỹ thuật)

- **Cơ chế Fallback an toàn:** Nếu số lượng mẫu dương trong tập huấn luyện quá nhỏ so với yêu cầu tối thiểu để học cây, mô hình tự động kích hoạt cơ chế dự phòng bằng trung bình log của mẫu dương; nếu không có mẫu dương nào thì gán bằng 0 và dự báo kết hợp bằng 0.
- **Thiết kế phân nhánh độc lập:** Hai giai đoạn Occurrence và Severity được mô-đun hóa độc lập, cho phép điều chỉnh độ sâu và siêu tham số riêng biệt phù hợp với từng nhiệm vụ (phân loại nhị phân vs hồi quy mẫu dương).

---

## 6. Kế Hoạch Tinh Chỉnh & Cân Nhắc Lý Thuyết (Tuning Plan & Theoretical Considerations)

### 6.1. Thiết lập siêu tham số cho hai giai đoạn
Trong phạm vi bài lab, nhóm dự kiến dùng một cấu hình cây định trước cho mỗi giai đoạn rồi so sánh mô hình với các ứng viên bằng Validation RMSE:
- **Giai đoạn 1 (Occurrence):** Dự kiến dùng cây hồi quy học nhãn 0/1, giới hạn độ sâu và số mẫu tại lá từ trước nhằm ước lượng xác suất thực nghiệm $p(x) \in [0, 1]$ đồng thời giới hạn độ phức tạp của cây.
- **Giai đoạn 2 (Severity):** Dự kiến dùng cây hồi quy với độ sâu và số mẫu tối thiểu tại lá xác định trước học trên tập con mẫu dương $\mathcal{D}_{\text{pos}}$ để mô hình hóa biên độ doanh thu khi có giao dịch.

### 6.2. Cân nhắc lý thuyết về Bất đẳng thức Jensen
- Do nhánh Severity học trong không gian logarit $\hat{y}_{\text{pos}} = \overline{\log(1 + s)}$, theo bất đẳng thức Jensen:

$$
\exp\left(\overline{\log(1 + s)}\right) - 1 \le \overline{s}
$$

- Phép nghịch đảo hàm mũ tương ứng với trung bình nhân của doanh thu cộng một, sau đó trừ một; không phải trung bình số học doanh thu. Nhóm xác định đây là một đặc thù toán học cần lưu ý khi diễn giải kết quả đầu ra trên thang đo tiền tệ BRL.
