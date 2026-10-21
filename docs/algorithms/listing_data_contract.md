# Cold-start trước lần bán đầu: dữ liệu bắt buộc

Olist hiện tại không có ngày niêm yết, giá niêm yết theo thời điểm hoặc catalog
đầy đủ các sản phẩm chưa bán. Không thể suy ra những dữ liệu này từ lần mua đầu.
Do đó **chưa có kết quả thực nghiệm cold-start trước bán trên dữ liệu thật**.

Luồng mới nhận dữ liệu ngoài, tách hoàn toàn khỏi proxy sau lần bán đầu:

- `catalog.csv`: một dòng mỗi `product_id`, gồm `listing_date`, `listed_price`,
  `features_available_at`, `product_category_name`; có thể thêm năm thuộc tính số
  của mô hình nội dung (khối lượng, số ảnh, độ dài tên, mô tả; giá lấy từ catalog).
- Catalog phải gồm cả sản phẩm chưa bán. Các thuộc tính là snapshot đã sẵn có
  trước hoặc tại `listing_date`, không lấy từ bảng cập nhật sau này.
- `transactions.csv`: `product_id`, `purchased_at`, `revenue` (doanh thu từng dòng).
- `observed-until`: mốc kết thúc quan sát được bên cung cấp xác nhận bao phủ đầy đủ;
  không lấy thời điểm bán cuối của từng sản phẩm làm mốc.

Luồng giữ sản phẩm không bán với nhãn **0**, loại cửa sổ chưa đủ 7 ngày, từ chối
thuộc tính có timestamp sau ngày niêm yết. Target là doanh thu trong
`[listing_date, listing_date + 7 days)`, giá là `listed_price`, không phải giá tương lai.

```powershell
.venv/Scripts/python.exe -m src.evaluation.listing_launch --catalog catalog.csv --transactions transactions.csv --observed-until 2026-10-21 --output results/listing_launch
```

Chia ba fold theo thời gian với purge nhãn Train, so sánh cây, trung vị danh mục,
baseline 0 và baseline một đơn vị. Xuất metric, manifest và model riêng; không
thay mô hình Olist đang triển khai. Không dùng Test chọn mô hình hay tham số.

Các test tổng hợp chỉ kiểm tra nhãn 0, mốc thời gian, schema và tích hợp, không
được dùng làm bằng chứng về độ chính xác trên thị trường. Với dữ liệu hiện có,
phần còn thiếu là **dữ liệu catalog thực**, không phải thêm một thuật toán.
