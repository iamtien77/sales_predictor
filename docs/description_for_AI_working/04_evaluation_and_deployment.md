# 📊 Giai Đoạn 4: Nghiệm Thu Độc Lập, Phân Tích Sai Số & Triển Khai Kinh Doanh (Phase 4 Playbook)

> **Mục tiêu:** Thực hiện kiểm định khách quan trên 293.717 dòng Test theo thang đo tiền tệ thực tế BRL, mổ xẻ phần dư Top 20 ca dự báo lệch nhất, đóng gói artifact và triển khai ứng dụng Streamlit dự báo đa kịch bản.  
> **Hàm điều phối:** `step_8_evaluate_and_compare()`, `step_9_error_analysis()`, `step_10_persistence_and_demo()` trong [`main.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/main.py).  
> **Mã nguồn cốt lõi:** [`src/evaluation/metrics.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/metrics.py), [`src/evaluation/error_analysis.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/error_analysis.py), [`src/models/persistence.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/persistence.py), [`src/deployment/predictor.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/deployment/predictor.py), [`app.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/app.py).

---

### Bước 1. Quy Trình Nghiệm Thu Độc Lập trên Test Set (293.717 Dòng)
- Nguyên tắc quy đổi tiền tệ thực tế: Dự báo log của mô hình được chuyển đổi ngược về thang đo BRL và chặn dưới bằng 0:
  
  $$\hat{y}_{\text{money}} = \max\left(0.0, \exp(\hat{y}) - 1\right) = \max\left(0.0, \text{expm1}(\hat{y})\right)$$

- Tính toán song song bộ chỉ số khoa học và chỉ số nghiệp vụ bán lẻ:
  1. **MSE (BRL²):** Đo lường hàm mất mát bình phương.
  2. **MAE (BRL):** Đo sai lệch tuyệt đối trung bình.
  3. **RMSE (BRL):** Chỉ số đo lường sai số tiêu chuẩn trong kinh doanh.
  4. **$R^2$ Score:** Đo lường tỷ lệ phương sai thực tế giải thích được.
  5. **WAPE (Weighted Absolute Percentage Error):**
     
     $$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$$
     
     Khắc phục nhược điểm chia cho 0 của MAPE khi doanh thu thực tế bằng 0 BRL.
  6. **Supply Chain Cost Per Item (BRL):** Ma trận chi phí tổn thất chuỗi cung ứng mô phỏng chi phí lưu kho khi dư hàng (Overstock penalty = 2 BRL) và tổn thất cơ hội khi đứt hàng (Understock penalty = 10 BRL).
- Lưu bảng tổng hợp xếp hạng ra file [`results/metrics/model_comparison.csv`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/results/metrics/model_comparison.csv).

### Bước 2. Xuất Bản Hệ Thống Biểu Đồ Đánh Giá Khoa Học
- Triển khai trong [`src/evaluation/visualization.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/visualization.py) với độ phân giải cao `dpi=300`:
  - `model_comparison.png`: Biểu đồ cột đối sánh trực quan MSE, MAE, RMSE và $R^2$ giữa toàn bộ các mô hình.
  - `actual_vs_predicted.png`: Biểu đồ phân tán so sánh giá trị Thực tế vs Dự báo.
  - `residuals_plot.png`: Đồ thị phân phối sai số phần dư $r = y - \hat{y}$ và kiểm định tính ngẫu nhiên.
  - `feature_importance.png`: Đồ thị Top 15 đặc trưng quan trọng nhất của mô hình cây.

### Bước 3. Phân Tích Phần Dư Chuyên Sâu & Top 20 Ca Lệch Nhất (Error Analysis)
- Triển khai hàm `extract_top_worst_predictions()` trong [`src/evaluation/error_analysis.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/error_analysis.py).
- Trích xuất Top 20 bản ghi có sai số tuyệt đối lớn nhất $|y - \hat{y}|$ trên tập Test.
- Mổ xẻ 3 nguyên nhân kinh tế và thống kê cốt lõi:
  1. *Đột biến đơn hàng sỉ (B2B Bulk Orders):* Khách hàng doanh nghiệp bất ngờ đặt mua số lượng lớn sản phẩm giá trị cao vào một tuần bất thường, vượt khỏi quy luật trượt của khách hàng bán lẻ cá nhân.
  2. *Hiện tượng trễ pha do chiến dịch khuyến mãi chớp nhoáng (Flash Sales):* Doanh số bùng nổ trong 1 tuần rồi trở về 0 ngay tuần sau đó.
  3. *Thiếu dữ liệu tồn kho thực tế:* Tập dữ liệu Olist không có thông tin về tình trạng còn hàng hay hết hàng (Out-of-Stock), khiến mô hình hiểu nhầm việc hết hàng là sản phẩm không có nhu cầu.

### Bước 4. Đóng Gói Model Persistence & Kiểm Tra Tái Lập (Model Persistence)
- Triển khai hàm `save_artifacts()` trong [`src/models/persistence.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/persistence.py).
- Lưu gói artifact toàn diện vào [`results/saved_models/best_sales_model.pkl`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/results/saved_models), chứa:
  - Mô hình tối ưu đã huấn luyện (`model`).
  - Pipeline tiền xử lý đã fit hoàn chỉnh (`pipeline`).
  - Danh sách 89 tên đặc trưng (`feature_names`).
  - Metadata phiên bản, thời gian huấn luyện và độ đo thực nghiệm.
- Kiểm tra tính tái lập: Nạp lại artifact qua `load_artifacts()` và xác thực độ lệch sai số trên tập mẫu bằng đúng $0.00000000$.

### Bước 5. Mô Phỏng Kịch Bản Kinh Doanh & Vận Hành Streamlit App 5 Tabs
- Triển khai các kịch bản kinh doanh trong [`src/deployment/predictor.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/deployment/predictor.py):
  1. `predict_scenario_new_product()`: Dự báo doanh thu ra mắt sản phẩm mới (tích hợp cả zero-history proxy và mô hình Cold-Start).
  2. `predict_scenario_marketing_campaign()`: Mô phỏng độ nhạy ngân sách quảng cáo với 2 chế độ (`baseline_only` và `calibrated_diminishing_returns`).
  3. `calculate_discount_scenario()`: Bảng tính doanh thu theo sản lượng và chiết khấu giả định số học chuẩn xác.
- Vận hành giao diện Web tương tác trực quan 5 Thẻ chức năng qua [`app.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/app.py):
  - Khởi chạy: `.venv/Scripts/python.exe -m streamlit run app.py`.
  - Thẻ 1: *Một sản phẩm* (Dự báo tức thời cho 1 sản phẩm tuần đích).
  - Thẻ 2: *Nhiều sản phẩm từ CSV* (Batch prediction lên đến 10.000 dòng, xuất file CSV kết quả).
  - Thẻ 3: *Mô phỏng giả định giảm giá* (Tính toán doanh thu và sản lượng giữ doanh thu tham chiếu).
  - Thẻ 4: *Sản phẩm mới (Cold-Start)* (Dự báo tuần ra mắt dựa trên đặc trưng nội dung).
  - Thẻ 5: *Mô phỏng Tiếp thị (Marketing Uplift)* (Thanh trượt AED, mô phỏng ROAS và đồ thị đường cong bão hòa).

### Bước 6. Mô Hình Hồi Quy Sản Phẩm Mới Mở Bán (Cold-Start Content-based Regressor)
- Triển khai trong [`src/models/cold_start.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/cold_start.py):
  - Khai thác dữ liệu 32.216 sản phẩm Olist tuần đầu tiên.
  - Xây dựng `CategoryMedianColdStartBaseline` (đối chuẩn trung vị danh mục).
  - Xây dựng `ColdStartProductRegressor` From-Scratch học trên 6 đặc tính nội dung (giá niêm yết, trọng lượng, số ảnh, độ dài tiêu đề, độ dài mô tả, target-encoded category).
  - Đánh giá độc lập trên sản phẩm chưa thấy: $\text{MAE} = 41.60\text{ BRL}$ (vượt trội so với Baseline $119.68\text{ BRL}$) và $R^2 = +0.5301$.

### Bước 7. Động Cơ Mô Phỏng Tiếp Thị Kinh Tế Lượng (Marketing Uplift & ROAS Engine)
- Triển khai trong [`src/deployment/predictor.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/deployment/predictor.py) qua hàm `simulate_marketing_response()`:
  - Ứng dụng hàm bão hòa phi tuyến Log-Response với hệ số co giãn quảng cáo AED:
    $$\Delta \text{Sales}(S) = S_{\text{baseline}} \times \epsilon \times \ln\left(1 + \frac{S}{S_{\text{ref}}}\right)$$
  - Kiểm chứng định luật lợi biên giảm dần (Diminishing Marginal Returns): $\Delta S$ ở các mức ngân sách cao tăng chậm dần, ngăn chặn ngoại suy phi thực tế.
  - Cung cấp trọn bộ chỉ số kinh doanh: `Incremental_Sales_BRL`, `Sales_Growth_%`, `ROAS` và `Marketing_ROI_%`.

