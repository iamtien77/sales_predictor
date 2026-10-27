# 📋 Giai Đoạn 1: Tiền Xử Lý Dữ Liệu & Khởi Tạo Weekly Panel (Phase 1 Playbook)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn cũ. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy full ngày 27/10/2026.

> **Mục tiêu:** Xây dựng quy trình chuẩn mực chuyển đổi từ dữ liệu giao dịch thô Olist sang bảng chuỗi thời gian Sản phẩm – Tuần (Weekly Product-Week Panel ~1,2 triệu mẫu), thực hiện kiểm toán dữ liệu và phân chia Temporal Split 80/10/10 chống rò rỉ thông tin tương lai.  
> **Hàm điều phối:** `step_1_load_and_explore()`, `step_2_clean_and_preprocess()`, `step_3_build_weekly_panel()`, `step_5_split_and_scale()` trong [`main.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/main.py).  
> **Mã nguồn cốt lõi:** [`src/data/loader.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/loader.py), [`src/data/builder.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/builder.py), [`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/preprocessing.py).

---

### Bước 1. Khám phá và Nạp 6 Bảng Dữ liệu Giao dịch Olist
- Sử dụng hàm `load_data()` trong [`src/data/loader.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/loader.py) để nạp và hợp nhất 6 bảng nguồn e-commerce Olist:
  1. `olist_orders_dataset.csv`: Thông tin đơn hàng và thời gian giao dịch (`order_purchase_timestamp`).
  2. `olist_order_items_dataset.csv`: Dòng chi tiết sản phẩm, giá bán (`price`) và cước phí (`freight_value`).
  3. `olist_products_dataset.csv`: Danh mục (`product_category_name`), trọng lượng, kích thước, ảnh.
  4. `olist_customers_dataset.csv`: Mã khách hàng, thành phố, bang (`customer_state`).
  5. `olist_order_payments_dataset.csv`: Hình thức thanh toán và số kỳ trả góp.
  6. `olist_order_reviews_dataset.csv`: Điểm số đánh giá (`review_score`) và ngày phản hồi.
- Kiểm tra tính nhất quán của các khóa nối (`order_id`, `product_id`, `customer_id`). Ghi nhận dữ liệu thô ban đầu đạt **112.650 dòng giao dịch sản phẩm**.

### Bước 2. Làm sạch Nguồn Item trước Aggregation (Data Quality Audit)
- Triển khai hàm `clean_weekly_source_data()` trong [`src/data/builder.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/builder.py).
- Nguyên tắc bảo toàn: Không cắt xén (capping) hay loại bỏ các giao dịch có giá trị cao nếu chúng là đơn hàng hợp lệ.
- Thực hiện kiểm toán dữ liệu:
  - Loại bỏ các dòng thiếu thông tin cốt lõi (`price <= 0`, thiếu `product_id` hoặc thiếu mốc thời gian).
  - Chuẩn hóa kiểu ngày tháng `datetime64[ns]` trên trường `order_purchase_timestamp`.
  - Khử các bản ghi có trạng thái hủy đơn (`canceled`, `unavailable`) không phát sinh doanh thu thực tế.
- Kết quả kiểm toán: Giữ lại **112.650 dòng sản phẩm hợp lệ** sẵn sàng cho bước tạo bảng thời gian.

### Bước 3. Xây dựng Bảng Chuỗi Thời Gian Sản Phẩm – Tuần (Weekly Panel Builder)
- Triển khai hàm `build_weekly_panel_data()` trong [`src/data/builder.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/builder.py).
- **Quy tắc Tuần Lịch:** Tuần được tính trọn vẹn từ thứ Hai (00:00:00) đến Chủ nhật (23:59:59). Cột `order_purchase_timestamp` trong weekly panel đại diện cho ngày thứ Hai của **tuần cần dự báo (target week)**.
- **Biến mục tiêu (Target Variable):**
  
  $$\text{sales} = \sum_{i \in \text{week}} \text{price}_i \quad (\text{đơn vị BRL})$$

- **Bảo toàn phân phối Zero-Sales:**
  - Với mỗi sản phẩm, thời điểm quan sát bắt đầu từ **tuần phát sinh đơn hàng đầu tiên** cho đến tuần cuối cùng của toàn hệ thống.
  - Các tuần không có đơn hàng trong khoảng thời gian này được điền đầy đủ với `sales = 0.0 BRL`. Không tạo tuần trước lần bán đầu tiên để tránh giả định sai về thời điểm ra mắt sản phẩm.
- **Quy mô tập dữ liệu bảng:** Thu được **1.198.182 bản ghi Sản phẩm – Tuần**, trong đó hơn **90% số tuần có doanh thu bằng 0 BRL**, phản ánh chân thực bài toán nhu cầu ngắt quãng (Intermittent Demand) trong bán lẻ.

### Bước 4. Phân chia Chuỗi Thời Gian Chống Rò Rỉ Dữ Liệu (Temporal Split 80/10/10)
- Sử dụng hàm `weekly_split_data()` trong [`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/preprocessing.py).
- Chia dữ liệu theo tiến trình thời gian tuần target độc lập (không chia ngẫu nhiên theo dòng sản phẩm):
  - **Tập Train (80% số tuần đầu):** **655.066 dòng**.
  - **Tập Validation (10% số tuần kế tiếp):** **249.399 dòng**.
  - **Tập Test (10% số tuần cuối cùng):** **293.717 dòng**.
- Bảo đảm 100% sản phẩm trong cùng một tuần lịch nằm trọn vẹn ở một phân vùng dữ liệu, triệt tiêu hoàn toàn hiện tượng rò rỉ thông tin tương lai vào quá khứ (Data Leakage).

### Bước 5. Xây dựng Pipeline Tiền Xử Lý Một Chiều (CustomDataPipeline)
- Đóng gói lớp `CustomDataPipeline` trong [`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/preprocessing.py):
  1. `CustomSimpleImputer`: Điền trung vị (Median) cho biến số học, yếu vị (Mode) cho biến danh mục.
  2. `CustomOneHotEncoder`: Mã hóa 73 danh mục sản phẩm Olist, các danh mục lạ xuất hiện ở tương lai được gán vector 0.
  3. `CustomStandardScaler`: Chuẩn hóa Z-score dựa trên trung bình $\mu$ và độ lệch chuẩn $\sigma$.
- **Nguyên tắc đóng băng thống kê:** Pipeline **chỉ được gọi `fit()` duy nhất trên tập Train**. Tập Validation và Test chỉ được gọi `transform()`.
- Chuyển đổi nhãn mục tiêu sang không gian log tự nhiên: $y = \log(1 + \text{sales})$. Lưu trữ ma trận đệm vào `data/processed/` gồm `X_train.npz`, `y_train.npy`, `X_val.npz`, `y_val.npy`, `X_test.npz`, `y_test.npy`.
