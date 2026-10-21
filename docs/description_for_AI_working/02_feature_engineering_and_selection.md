# ⚙️ Giai Đoạn 2: Kỹ Nghệ Hệ Thống 89 Đặc Trưng & Kiểm Toán Chống Rò Rỉ (Phase 2 Playbook)

> **Mục tiêu:** Thiết kế hệ thống 89 đặc trưng dự báo toàn diện bao quát quán tính doanh thu, thống kê trượt, lịch sử đánh giá khách hàng, mùa vụ và danh mục sản phẩm; thực hiện kiểm toán nghiêm ngặt chống rò rỉ dữ liệu (Anti-Leakage Audit).  
> **Hàm điều phối:** `WeeklySalesFeatureEngineer` tích hợp trong `CustomDataPipeline`, xuất đồ thị qua `step_4_run_eda()` trong [`main.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/main.py).  
> **Mã nguồn cốt lõi:** [`src/data/feature_engineering.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/feature_engineering.py), [`src/evaluation/visualization.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/visualization.py).

---

### Bước 1. Thiết kế Nhóm Đặc Trưng Quán Tính Trễ & Thống Kê Trượt (Lag & Rolling Features)
- Toàn bộ các đặc trưng lịch sử được tính toán hoàn tất **ngay trước tuần mục tiêu (target week)**:
  - `last_week_sales`: Doanh thu của tuần hoàn tất liền trước ($t-1$).
  - `sales_lag_2_weeks`: Doanh thu trễ 2 tuần lịch ($t-2$).
  - `sales_lag_4_weeks`: Doanh thu trễ 4 tuần lịch ($t-4$).
  - `sales_mean_4_weeks`: Doanh thu trung bình trượt 4 tuần gần nhất:
    
    $$\bar{S}_{4w} = \frac{1}{4} \sum_{k=1}^4 \text{sales}(t-k)$$

  - `sales_std_4_weeks`: Độ biến động doanh thu chuẩn hóa trong cửa sổ 4 tuần.
  - `last_week_units`: Số lượng đơn vị sản phẩm bán được trong tuần liền trước.

### Bước 2. Kỹ Nghệ Đặc Trưng Vòng Đời & Giá Bán Lịch Sử (Pricing & Lifecycle)
- `price`: Giá bán trung bình của sản phẩm trong **tuần có phát sinh giao dịch gần nhất trước target week**. Áp dụng kỹ thuật Forward-fill theo chiều thời gian. Tuyệt đối không sử dụng giá bán phát sinh trong tuần target để tạo biến độc lập.
- `weeks_since_first_sale`: Số tuần lịch tính từ thời điểm sản phẩm phát sinh đơn hàng đầu tiên trong hệ thống, đại diện cho độ trưởng thành (Product Maturity) và độ nhận diện thương hiệu.

### Bước 3. Kỹ Nghệ Đặc Trưng Phản Hồi Khách Hàng Chống Rò Rỉ (Anti-Leakage Review Features)
- Nhận định rủi ro rò rỉ dữ liệu: Trong thương mại điện tử, khách hàng chỉ để lại đánh giá sau khi nhận hàng (thường trễ từ 1 đến 3 tuần sau khi mua). Nếu lấy đánh giá của đơn hàng trong tuần target gán vào dự báo đầu tuần target thì mô hình sẽ bị rò rỉ dữ liệu tương lai.
- **Giải pháp kỹ thuật:**
  - `past_review_count`: Tổng số lượt đánh giá của sản phẩm có ngày trả lời (`review_answer_timestamp`) hoàn tất **trước mốc bắt đầu của tuần target**.
  - `past_review_rating`: Điểm sao trung bình lũy kế tính đến trước tuần target.

### Bước 4. Kỹ Nghệ Đặc Trưng Hành Vi Người Mua & Độ Phủ Địa Lý (Customer Behavior)
- `customer_past_spend`: Tổng chi tiêu lịch sử trung bình của nhóm khách hàng đã mua sản phẩm ở tuần giao dịch gần nhất.
- `customer_past_orders`: Tần suất đặt hàng tích lũy của nhóm khách mua trước thời điểm dự báo.
- `customer_state_count`: Số lượng bang khác nhau mà sản phẩm đã tiếp cận được trong tuần bán gần nhất, đo lường độ phủ thị trường.

### Bước 5. Kỹ Nghệ Biến Chu Kỳ Thời Gian & Mùa Mua Sắm Cao Điểm (Seasonality Features)
- `is_peak_shopping_season`: Biến nhị phân nhận diện tuần lễ hội Black Friday (tuần cuối tháng 11) và mùa mua sắm Giáng sinh / Năm mới (tháng 12).
- `is_holiday_week`: Nhận diện các tuần có ngày lễ quốc gia Brazil (Carnival, Độc lập, Lao động...).
- Biến đổi tuần hoàn lượng giác:
  
  $$\text{month\_sin} = \sin\left(\frac{2\pi \times \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \times \text{month}}{12}\right)$$

  Giúp mô hình hiểu được tính liên tục giữa tháng 12 và tháng 1 năm sau.

### Bước 6. Mã Hóa Vector Danh Mục Ngành Hàng & Ghép Nối 89 Chiều
- 73 biến chỉ báo One-Hot tương ứng với 73 nhóm ngành hàng Olist (`category_cama_mesa_banho`, `category_informatica_acessorios`, `category_beleza_saude`...).
- Ghép nối thành ma trận đặc trưng chuẩn hóa $X \in \mathbb{R}^{N \times 89}$ gồm 16 biến số học và 73 biến phân loại.
- Kiểm toán ma trận: Không có giá trị vô cùng (`inf`), không có giá trị khuyết thiếu (`NaN`), ma trận được chuyển sang định dạng thưa `scipy.sparse.csr_matrix` để giảm 85% dung lượng RAM.
