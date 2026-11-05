# Tổng Quan Về Các Mô Hình Trong Bài Lab Học Máy

Tài liệu này tổng hợp danh mục và mô tả ngắn gọn về các mô hình mà nhóm sẽ thử nghiệm trong bài lab môn Học Máy. Bản thiết kế được xây dựng ngay từ ban đầu nhằm xác định rõ mục tiêu so sánh và quy trình làm bài, hoàn toàn không phụ thuộc vào code hay kết quả cụ thể.

---

## 1. Danh mục Tài liệu Thiết kế Thuật toán

| Tài liệu thiết kế | Mục tiêu trong bài lab |
| --- | --- |
| [Kế hoạch Tổng thể (5W1H)](design.md) | Kế hoạch tổng thể theo 5W1H, cấu trúc dự kiến và quy tắc chống rò rỉ dữ liệu. |
| [Mô hình Đối chứng Cơ sở (Baselines)](baselines.md) | Thiết lập 4 mốc chuẩn cơ sở: Zero, Revenue Mean, Log Mean và Previous Week để đối chiếu. |
| [Hồi quy Tuyến tính Điều chuẩn](linear_regression.md) | Mô hình tham số cơ sở giải bằng Normal Equation OLS có Ridge và Gradient Descent. |
| [Cây Quyết định Hồi quy](decision_tree.md) | Khối cây cơ sở phi tuyến tối ưu hóa mức giảm phương sai (Variance Reduction). |
| [Rừng Ngẫu nhiên Hồi quy](random_forest.md) | Phương pháp học kết hợp Bagging kết hợp lấy mẫu bootstrap và ngẫu nhiên đặc trưng. |
| [Tăng cường Độ dốc Hồi quy](gradient_boosting.md) | Phương pháp học kết hợp Boosting tối ưu hóa tuần tự trên phần dư của hàm mất mát. |
| [Mô hình Hai Giai đoạn (Two-Stage Hurdle)](two_stage_hurdle.md) | Kiến trúc chuyên biệt xử lý dữ liệu có nhiều số 0, tách xác suất và mức doanh thu bán. |
| [Proxy Doanh thu 7 Ngày Đầu](cold_start.md) | Ước lượng doanh thu khởi đầu dựa trên giao dịch đầu tiên đã biết và thuộc tính sản phẩm. |
| [Mô phỏng Kịch bản Tiếp thị](marketing_response.md) | Mô hình độ nhạy doanh thu theo giả định hàm logarit lợi suất giảm dần để chạy What-If. |
| [Quy chuẩn Dữ liệu Niêm yết](listing_data_contract.md) | Đặc tả yêu cầu dữ liệu bắt buộc cho bài toán dự báo cold-start trước khi ra mắt. |
| [Kế hoạch Kiểm tra Mô hình](experimental_validation.md) | Cách kiểm tra tính ổn định của mô hình qua nhiều mốc thời gian và lưu ý khi làm bài. |

---

## 2. Khung Đánh giá & Tiêu chuẩn So sánh Mô hình

Toàn bộ các mô hình được nhóm so sánh trên cùng một điều kiện:
- **Nguyên tắc chống rò rỉ dữ liệu (Data Leakage):** Phân chia tập dữ liệu theo thời gian (Temporal Split 80/10/10) TRƯỚC khi tiến hành tiền xử lý. Toàn bộ tham số thống kê chỉ được fit trên tập Train.
- **Tiêu chuẩn chọn mô hình:** Sử dụng **Root Mean Squared Error trên thang đo tiền tệ thực tế (Validation RMSE BRL)** sau khi chuyển đổi ngược từ không gian logarit: $\hat{s} = \max(0, \exp(\hat{y}) - 1)$.
- **Bộ chỉ số đo lường toàn diện:**
  1. **RMSE (BRL):** Tiêu chí đánh giá chính, phạt nặng các sai số dự báo lệch lớn.
  2. **MAE (BRL):** Đo lường độ lệch tuyệt đối trung bình trên từng đơn vị sản phẩm–tuần.
  3. **MSE ($\text{BRL}^2$):** Phản ánh phương sai sai số bình phương.
  4. **$R^2$ Score:** Đo lường tỷ lệ phương sai doanh thu được giải thích so với mốc trung bình.
  5. **WAPE:** Sai số phần trăm có trọng số, tránh lỗi chia cho 0 trên dữ liệu thưa.

---

## 3. Đặc Điểm Kiến Trúc Của Các Thuật Toán Dự Kiến

Nhóm thiết kế và khảo sát các họ thuật toán với các cơ chế toán học bổ trợ lẫn nhau:
1. **Baselines Cơ sở:** Bốn mô hình tham chiếu (Zero, Revenue Mean, Log Mean, Previous Week) thiết lập mốc đối chứng tối thiểu, làm thước đo để xem mô hình học máy có thực sự học được quy luật hay không.
2. **Linear Regression:** Mô hình tham số tuyến tính có điều chuẩn Ridge nhằm kiểm tra giả định tương quan tuyến tính của các biến trễ và biến danh mục, tích hợp cơ chế kẹp biên log để ổn định số học.
3. **Cây Quyết định (Decision Tree):** Mô hình phi tuyến dạng phân tách nhị phân theo độ giảm phương sai, tạo ra các quy tắc quyết định dạng If-Else có khả năng giải thích cao.
4. **Học Kết hợp (Random Forest & Gradient Boosting):** Hai hướng tiếp cận Ensemble tương phản: Bagging (lấy mẫu ngẫu nhiên độc lập để giảm phương sai) và Boosting (học tuần tự trên phần dư để giảm sai số).
5. **Mô hình Hai Giai đoạn (Two-Stage Hurdle):** Kiến trúc phân tách bài toán nhiều số 0 thành hai giai đoạn độc lập: phân loại xác suất phát sinh đơn hàng và hồi quy quy mô doanh thu khi có giao dịch.

---

## 4. Các Phần Thử Nghiệm Mở Rộng

- **Luồng Cold-start Proxy:** Thử nghiệm ước lượng doanh thu 7 ngày đầu sau giao dịch đầu tiên đã biết. Nhóm thiết lập Baseline Giá bán đơn vị và Baseline Trung vị danh mục làm chuẩn đối sánh bắt buộc.
- **Luồng Mô phỏng Kịch bản Tiếp thị:** Thiết lập hàm phản hồi phi tuyến $u(A) = 1 + \beta \ln(1 + A/R)$ để khảo sát độ nhạy doanh thu theo ngân sách quảng cáo dựa trên giả định toán học, phục vụ trực quan hóa What-If.
- **Quy chuẩn Dữ liệu Niêm yết:** Xác lập các tiêu chuẩn cần thiết về bảng catalog thực tế để phục vụ bài toán dự báo trước khi sản phẩm ra mắt sau này.

---

## 5. Quy Trình Các Bước Thực Hiện Bài Lab

```mermaid
flowchart TD
    A["Dữ liệu thô Olist (Orders, Items, Products, Reviews)"] --> B["Tổng hợp Product–Week Panel"]
    B --> C["Tính toán đặc trưng trễ & cuộn (chỉ dùng quá khứ)"]
    C --> D["Chia dữ liệu theo thời gian (Temporal Split 80/10/10)"]
    D --> E["Tập Train (80% số tuần đầu)"]
    D --> F["Tập Validation (10% số tuần tiếp theo)"]
    D --> G["Tập Test (10% số tuần cuối cùng)"]
    E --> H["Fit các bộ tiền xử lý (Imputer, Scaler, Encoder)"]
    H --> I["Đóng băng tham số tiền xử lý"]
    I -.->|Chỉ áp dụng Transform| F
    I -.->|Chỉ áp dụng Transform| G
    H --> J["Huấn luyện 4 Baselines & Các Mô hình Ứng viên"]
    J --> K["Đánh giá & Chọn mô hình tối ưu trên Validation (RMSE BRL)"]
    K --> L["Đánh giá độc lập lần cuối trên Test"]
    L --> M["Tổng kết và hoàn thành bài lab"]
```
