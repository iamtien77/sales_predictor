# 📋 Giai Đoạn 1: Tiền Xử Lý Dữ Liệu & Khởi Tạo Weekly Panel (Phase 1 Playbook)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn thực nghiệm của dự án. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy hoàn chỉnh trên toàn bộ dữ liệu.

> **Mục tiêu kỹ thuật:** Xây dựng quy trình chuẩn mực chuyển đổi từ dữ liệu giao dịch thô thương mại điện tử Olist sang bảng chuỗi thời gian Sản phẩm – Tuần (Weekly Product-Week Panel ~1,2 triệu mẫu), thực hiện kiểm toán chất lượng dữ liệu và phân chia Temporal Split 80/10/10 triệt tiêu hoàn toàn rò rỉ thông tin tương lai.  
> **Hàm điều phối:** `step_1_load_and_explore()`, `step_2_clean_and_preprocess()`, `step_3_build_weekly_panel()`, `step_5_split_and_scale()` trong [`main.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/main.py).  
> **Mã nguồn cốt lõi:** [`src/data/loader.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/loader.py), [`src/data/builder.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/builder.py), [`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/preprocessing.py).

---

### Bước 1. Khám Phá & Nạp 6 Bảng Dữ Liệu Giao Dịch Olist
- Sử dụng hàm `load_data()` trong [`src/data/loader.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/loader.py) để nạp và hợp nhất 6 bảng nguồn thương mại điện tử Olist qua cơ chế kết nối khóa ngoại:
  1. `olist_orders_dataset.csv`: Thông tin trạng thái đơn hàng và mốc thời gian giao dịch (`order_purchase_timestamp`).
  2. `olist_order_items_dataset.csv`: Chi tiết từng dòng sản phẩm, giá bán thực tế (`price`) và cước phí vận chuyển (`freight_value`).
  3. `olist_products_dataset.csv`: Đặc trưng danh mục hàng hóa (`product_category_name`), trọng lượng, kích thước đóng gói và số lượng ảnh.
  4. `olist_customers_dataset.csv`: Định danh khách hàng, thành phố và bang nhận hàng (`customer_state`).
  5. `olist_order_payments_dataset.csv`: Phương thức thanh toán (thẻ tín dụng, boleto, voucher) và số kỳ trả góp.
  6. `olist_order_reviews_dataset.csv`: Điểm số đánh giá phản hồi (`review_score`) và ngày gửi câu trả lời (`review_answer_timestamp`).
- **Kiểm toán khóa quan hệ:** Xác thực tính toàn vẹn tham chiếu giữa `order_id`, `product_id`, và `customer_id`. Tổng số bản ghi giao dịch mức dòng sản phẩm (item-level) đạt **112.650 dòng**.

---

### Bước 2. Kiểm Toán Chất Lượng & Làm Sạch Nguồn Item Trước Aggregation
- Triển khai hàm `clean_weekly_source_data()` trong [`src/data/builder.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/builder.py).
- **Nguyên tắc bảo toàn phân phối:** Tuyệt đối không áp dụng kỹ thuật kẹp biên (capping) hay loại bỏ tùy tiện các giao dịch có giá trị lớn nếu chúng là giao dịch thực tế hợp lệ. Việc cắt xén cơ học sẽ làm sai lệch bản chất phân phối doanh thu lệch phải cực đoan (Heavy-Tailed Distribution) vốn có của thị trường bán lẻ.
- **Tiêu chuẩn kiểm toán và lọc bản ghi:**
  - Loại bỏ các bản ghi không hợp lệ về mặt kinh tế: `price <= 0`, thiếu mã định danh `product_id` hoặc thiếu trường thời gian mua hàng.
  - Chuẩn hóa kiểu dữ liệu thời gian thống nhất sang `datetime64[ns]` trên trường `order_purchase_timestamp`.
  - Khử các đơn hàng bị hủy hoặc không sẵn sàng phục vụ (`canceled`, `unavailable`) do chúng không tạo ra dòng tiền doanh thu thực tế cho người bán.
- **Kết quả nghiệm thu dữ liệu làm sạch:** Giữ lại trọn vẹn **112.650 dòng sản phẩm hợp lệ**, đảm bảo không mất mát thông tin trước khi chuyển sang bước tổng hợp bảng chuỗi thời gian.

---

### Bước 3. Xây Dựng Bảng Chuỗi Thời Gian Sản Phẩm – Tuần (Weekly Panel Builder)
- Triển khai hàm `build_weekly_panel_data()` trong [`src/data/builder.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/builder.py).
- **Quy tắc Tuần Lịch chuẩn hóa:** Chu kỳ dự báo được cố định theo tuần lịch dương, bắt đầu từ 00:00:00 thứ Hai đến 23:59:59 Chủ Nhật. Cột mốc thời gian `order_purchase_timestamp` trong bảng đại diện cho điểm gốc (Forecast Origin) lúc đầu tuần cần dự báo (target week).
- **Định nghĩa toán học của biến mục tiêu:**
  Tổng doanh thu bán hàng của sản phẩm $p$ trong tuần $t$ được tính từ tổng giá trị `price` của các đơn vị hàng bán ra (loại trừ `freight_value` vì cước vận chuyển thuộc về đơn vị logistics, không phản ánh nhu cầu sản phẩm):

$$
s_{p,t} = \sum_{i \in I_{p,t}} \mathrm{price}_i \quad (\text{đơn vị: Brazilian Real - BRL})
$$

- **Cơ chế xử lý nhu cầu ngắt quãng (Intermittent Demand) & Triệt tiêu Survivorship Bias:**
  - *Vấn đề thiên lệch sinh tồn:* Nếu chỉ gom dữ liệu trên các tuần có phát sinh đơn hàng, mô hình sẽ không bao giờ nhìn thấy các tuần thất bại khi sản phẩm không bán được hàng. Điều này làm mô hình luôn dự báo doanh thu dương và thất bại hoàn toàn trong thực tế.
  - *Giải pháp:* Với mỗi sản phẩm, hệ thống khởi tạo lưới thời gian đầy đủ mọi tuần liên tục kể từ **tuần phát sinh đơn hàng đầu tiên** cho đến tuần cuối cùng của kỳ quan sát. Toàn bộ các tuần không phát sinh giao dịch được gán nhãn mục tiêu bằng đúng $0.0\text{ BRL}$.
  - *Bảo vệ giả định mở bán:* Không tạo các tuần trống trước thời điểm phát sinh giao dịch đầu tiên của sản phẩm để tránh giả định sai lầm về ngày niêm yết khi thiếu catalog gốc.
- **Quy mô tập dữ liệu bảng:** Thu được **1.198.182 bản ghi Sản phẩm – Tuần**, trong đó tỷ lệ tuần có doanh thu bằng 0 BRL vượt quá **90%**, phản ánh chân thực thách thức của bài toán lạm phát số không (Zero-Inflation) trong thương mại điện tử.

---

### Bước 4. Phân Chia Chuỗi Thời Gian Chống Rò Rỉ Dữ Liệu (Temporal Split 80/10/10)
- Triển khai hàm `weekly_split_data()` trong [`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/preprocessing.py).
- **Bản chất chống rò rỉ thời gian (Anti-Leakage Principle):**
  - Trong học máy chuỗi thời gian, việc sử dụng kỹ thuật K-Fold Cross-Validation ngẫu nhiên hoặc xáo trộn ngẫu nhiên các dòng bản ghi sẽ dẫn đến việc thông tin tương lai được dùng để giải thích cho quá khứ (Look-ahead Bias / Data Leakage).
  - Hệ thống quy định phân chia tập dữ liệu nghiêm ngặt theo trục thời gian thực tế dựa trên số lượng tuần lịch phân biệt:
    1. **Tập Huấn luyện (Train Set - 80% số tuần đầu):** **655.066 bản ghi**, dùng để học trọng số và tham số tiền xử lý.
    2. **Tập Kiểm định (Validation Set - 10% số tuần kế tiếp):** **249.399 bản ghi**, dùng độc lập để tinh chỉnh siêu tham số và chọn mô hình.
    3. **Tập Kiểm tra (Test Set - 10% số tuần cuối cùng):** **293.717 bản ghi**, tập kiểm định mù đại diện cho tương lai, chỉ được đánh giá một lần duy nhất.
- **Ranh giới tuyệt đối:** Toàn bộ các sản phẩm trong cùng một tuần lịch bắt buộc phải thuộc về cùng một phân vùng dữ liệu. Không có bất kỳ tuần nào xuất hiện đồng thời ở hai tập khác nhau.

---

### Bước 5. Xây Dựng Pipeline Tiền Xử Lý Một Chiều (CustomDataPipeline)
- Đóng gói lớp `CustomDataPipeline` trong [`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/preprocessing.py) tích hợp 3 bộ biến đổi một chiều:
  1. `CustomSimpleImputer`: Điền giá trị trung vị (Median) cho các biến số học (chống ảnh hưởng của ngoại lai) và yếu vị (Mode) cho các biến danh mục.
  2. `CustomOneHotEncoder`: Mã hóa nhị phân 73 nhóm ngành hàng Olist; thiết lập cơ chế phòng vệ tự động gán vector toàn số 0 cho các nhãn danh mục mới xuất hiện trong tương lai.
  3. `CustomStandardScaler`: Chuẩn hóa Z-score đưa các biến số học về phân phối có trung bình bằng 0 và phương sai bằng 1:

$$
z = \frac{x - \mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}
$$

- **Nguyên tắc Đóng Băng Thống Kê (Statistical Freeze):**
  - Bộ tham số thống kê ($\mu_{\mathrm{train}}$, $\sigma_{\mathrm{train}}$, median, mode, từ điển danh mục) **chỉ được ước lượng duy nhất một lần trên tập Train** qua phương thức `fit()`.
  - Tập Validation và Test tuyệt đối không được gọi `fit()` mà chỉ được áp dụng `transform()` với các tham số đã đóng băng từ Train.
- **Biến đổi không gian nhãn mục tiêu:** Nhãn huấn luyện được chuyển sang không gian log tự nhiên để ổn định phương sai:

$$
y = \log(1 + s)
$$

- **Quản lý Cache tối ưu:** Lưu trữ ma trận đệm vào thư mục `data/processed/` gồm `X_train.npz`, `y_train.npy`, `X_val.npz`, `y_val.npy`, `X_test.npz`, `y_test.npy` kèm chữ ký kiểm toán (schema fingerprint) để tái sử dụng tức thì giữa các vòng thực nghiệm.
