# ⚙️ Giai Đoạn 2: Kỹ Nghệ Hệ Thống 89 Đặc Trưng & Kiểm Toán Chống Rò Rỉ (Phase 2 Playbook)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn thực nghiệm của dự án. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy hoàn chỉnh trên toàn bộ dữ liệu.

> **Mục tiêu kỹ thuật:** Thiết kế và xây dựng hệ thống 89 đặc trưng dự báo toàn diện, phản ánh đa chiều từ quán tính tiêu dùng, thống kê trượt, vòng đời sản phẩm, lịch sử đánh giá khách hàng đến chu kỳ mùa vụ; thiết lập cơ chế kiểm toán nghiêm ngặt chống rò rỉ dữ liệu (Anti-Leakage Audit) và tối ưu hóa ma trận thưa.  
> **Hàm điều phối:** `WeeklySalesFeatureEngineer` tích hợp trong `CustomDataPipeline`, xuất trực quan hóa qua `step_4_run_eda()` trong [`main.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/main.py).  
> **Mã nguồn cốt lõi:** [`src/data/feature_engineering.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/feature_engineering.py), [`src/evaluation/visualization.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/visualization.py).

---

### Bước 1. Thiết Kế Nhóm Đặc Trưng Quán Tính Trễ & Thống Kê Trượt (Lag & Rolling)
- **Động lực kinh tế học hành vi:** Doanh số bán lẻ luôn tồn tại tính quán tính tiêu dùng (Demand Autoregressive). Nhu cầu của tuần kế tiếp chịu sự chi phối mạnh mẽ từ xu hướng ngắn hạn và mức độ biến động trong các tuần vừa qua.
- **Quy tắc chặn thời gian nghiêm ngặt:** Toàn bộ các đặc trưng quá khứ chỉ được tính toán trên các tuần đã kết thúc trọn vẹn **trước mốc bắt đầu của tuần đích ($t$)**:
  - `last_week_sales`: Doanh thu của tuần hoàn tất liền trước ($t-1$).
  - `sales_lag_2_weeks`: Doanh thu trễ 2 tuần lịch ($t-2$).
  - `sales_lag_4_weeks`: Doanh thu trễ 4 tuần lịch ($t-4$), đại diện cho chu kỳ mua sắm hàng tháng.
  - `last_week_units`: Tổng số lượng sản phẩm bán được trong tuần liền trước (đo lường trực tiếp về mặt sản lượng vật lý).
- **Đặc trưng thống kê trượt trong cửa sổ 4 tuần:**
  - *Doanh thu trung bình trượt (Rolling Mean):* Đo lường mức nền tảng doanh số ổn định:

$$
\bar{s}_{4w} = \frac{1}{4} \sum_{k=1}^4 \mathrm{sales}(t-k)
$$

  - *Độ biến động doanh thu trượt (Rolling Standard Deviation):* Đo lường mức độ bất định và rủi ro dao động của nhu cầu thị trường:

$$
\sigma_{4w} = \sqrt{\frac{1}{3} \sum_{k=1}^4 \left(\mathrm{sales}(t-k) - \bar{s}_{4w}\right)^2}
$$

---

### Bước 2. Kỹ Nghệ Đặc Trưng Vòng Đời & Giá Bán Lịch Sử (Pricing & Lifecycle)
- **Giá bán lịch sử (`price`):**
  - *Rủi ro rò rỉ dữ liệu:* Nếu sử dụng giá bán phát sinh trong chính tuần cần dự báo, mô hình sẽ phạm lỗi nhìn trước tương lai (Look-ahead Bias).
  - *Giải pháp thiết kế:* Thuộc tính giá được xác định bằng **giá bán trung bình của sản phẩm trong tuần có phát sinh giao dịch gần nhất trước tuần đích**. Áp dụng kỹ thuật điền tiến theo thời gian (Forward-fill). Đối với tuần đầu tiên sau khi xuất hiện đơn hàng, giá bán chính là giá của lần mở bán đầu tiên.
- **Độ trưởng thành sản phẩm (`weeks_since_first_sale`):**
  - Tính bằng số tuần lịch trôi qua kể từ thời điểm sản phẩm phát sinh đơn hàng đầu tiên trong hệ thống.
  - *Ý nghĩa học máy:* Giúp các thuật toán dạng cây phân nhánh hiệu quả giữa hai nhóm hành vi: sản phẩm mới mở bán (doanh số biến động mạnh, cần tích lũy độ nhận diện) và sản phẩm lâu năm (doanh số đã đi vào quỹ đạo bão hòa).

---

### Bước 3. Kỹ Nghệ Đặc Trưng Đánh Giá Khách Hàng Chống Rò Rỉ (Anti-Leakage Review)
- **Phân tích rủi ro rò rỉ thời gian thực tế:**
  - Trong thương mại điện tử, quy trình trải nghiệm của khách hàng diễn ra theo trình tự: *Đặt hàng $\to$ Vận chuyển (1–3 tuần) $\to$ Nhận hàng $\to$ Viết đánh giá phản hồi*.
  - Nếu gom tất cả đánh giá thuộc về đơn hàng trong tuần $t$ để dự báo doanh số đầu tuần $t$, mô hình sẽ sử dụng thông tin phát sinh sau đó từ 1 đến 3 tuần.
- **Giải pháp kiểm toán điểm thời gian (Point-in-Time Audit):**
  - Chỉ tổng hợp các đánh giá có thời điểm gửi câu trả lời (`review_answer_timestamp`) hoàn tất **trước 00:00:00 thứ Hai của tuần đích $t$**.
  - `past_review_count`: Tổng số lượt đánh giá hợp lệ tích lũy của sản phẩm trước tuần $t$.
  - `past_review_rating`: Điểm sao trung bình lũy kế (thang điểm 1 đến 5 sao) tính đến trước tuần $t$. Trường hợp sản phẩm chưa có đánh giá nào được gán giá trị mặc định là 0.

---

### Bước 4. Kỹ Nghệ Đặc Trưng Hành Vi Người Mua & Độ Phủ Địa Lý (Customer Behavior)
- **Tổng chi tiêu lịch sử (`customer_past_spend`):** Mức chi tiêu trung bình trong quá khứ của nhóm khách hàng đã mua sản phẩm ở tuần gần nhất, phản ánh phân khúc thu nhập và sức mua của tệp khách hàng mục tiêu.
- **Tần suất đặt hàng (`customer_past_orders`):** Đo lường mức độ trung thành của nhóm khách hàng mua sản phẩm.
- **Độ phân tán địa lý (`customer_state_count`):** Số lượng bang khác nhau tại Brazil mà sản phẩm đã tiếp cận được trong tuần bán gần nhất. Sản phẩm có độ phủ địa lý rộng thường có chuỗi cung ứng ổn định và ít bị phụ thuộc vào các cú sốc logistics cục bộ.

---

### Bước 5. Kỹ Nghệ Biến Chu Kỳ Thời Gian & Mùa Mua Sắm Cao Điểm (Seasonality)
- **Mùa mua sắm cao điểm (`is_peak_shopping_season`):** Biến chỉ báo nhị phân (0 hoặc 1) nhận diện tuần lễ hội giảm giá Black Friday (tuần cuối tháng 11) và mùa mua sắm Giáng sinh / Tết dương lịch (tháng 12).
- **Tuần lễ hội quốc gia (`is_holiday_week`):** Nhận diện các tuần có các ngày nghỉ lễ lớn tại Brazil (Lễ hội Carnival, Ngày Độc Lập, Ngày Quốc Tế Lao Động...).
- **Mã hóa chu kỳ lượng giác (Trigonometric Calendar Encoding):**
  - Các biến số lịch thông thường (tháng 1 đến 12) khi đưa vào mô hình hồi quy tuyến tính sẽ bị ngắt quãng giữa tháng 12 ($12$) và tháng 1 ($1$), dù về mặt thời gian chúng là hai tháng kế tiếp nhau.
  - Hệ thống áp dụng phép chiếu lượng giác lên vòng tròn đơn vị:

$$
\mathrm{month\_sin} = \sin\left(\frac{2\pi \times \mathrm{month}}{12}\right), \quad \mathrm{month\_cos} = \cos\left(\frac{2\pi \times \mathrm{month}}{12}\right)
$$

  - Phép biến đổi này bảo toàn tính liên tục tô-pô của không gian thời gian chu kỳ.

---

### Bước 6. Mã Hóa 73 Danh Mục Ngành Hàng & Tối Ưu Ma Trận Thưa (Sparse CSR Matrix)
- **Mã hóa biến danh mục:** 73 biến chỉ báo One-Hot đại diện cho toàn bộ các nhóm ngành hàng Olist (`category_cama_mesa_banho`, `category_beleza_saude`, `category_informatica_acessorios`...).
- **Cấu trúc ma trận đặc trưng toàn diện:** Ghép nối đồng bộ thành ma trận đặc trưng $X \in \mathbb{R}^{N \times 89}$ gồm:
  - 16 đặc trưng số học liên tục (quán tính, thống kê trượt, giá, vòng đời, review, địa lý, mùa vụ).
  - 73 đặc trưng phân loại nhị phân.
- **Tối ưu hóa bộ nhớ với SciPy CSR Matrix:**
  - Nếu lưu trữ ma trận dày đặc (Dense Matrix) với $1,2 \times 10^6$ dòng và 89 cột kiểu `float64`, bộ nhớ RAM đòi hỏi gần 1 GB chỉ riêng cho ma trận dữ liệu.
  - Do 73 cột danh mục chỉ chứa duy nhất một giá trị 1 trên mỗi dòng (độ thưa > 98%), việc nén ma trận sang định dạng thưa **SciPy CSR (Compressed Sparse Row)** giúp giảm hơn **85% dung lượng RAM**, tăng tốc độ tính toán tích vô hướng ma trận $X^T X$ lên gấp 4 lần trong quá trình huấn luyện Normal Equation.
