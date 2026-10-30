# 📊 Giai Đoạn 4: Nghiệm Thu Độc Lập, Phân Tích Sai Số & Triển Khai Kinh Doanh (Phase 4 Playbook)

> **Ghi chú lịch sử phát triển:** Nội dung và số liệu dưới đây phản ánh các giai đoạn thực nghiệm của dự án. Xem [kết quả hiện tại](../algorithms/README.md) và [README chính](../../README.md) cho lần chạy hoàn chỉnh trên toàn bộ dữ liệu.

> **Mục tiêu kỹ thuật:** Thực hiện quy trình nghiệm thu độc lập trên 293.717 dòng Test theo thang đo tiền tệ thực tế BRL, thiết lập ma trận tổn thất tài chính chuỗi cung ứng, mổ xẻ nguyên nhân kinh tế của Top 20 ca dự báo lệch nhất, đóng gói artifact và triển khai hệ thống suy luận đa kịch bản kinh doanh.  
> **Hàm điều phối:** `step_8_evaluate_and_compare()`, `step_9_error_analysis()`, `step_10_persistence_and_demo()` trong [`main.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/main.py).  
> **Mã nguồn cốt lõi:** [`src/evaluation/metrics.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/metrics.py), [`src/evaluation/error_analysis.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/error_analysis.py), [`src/models/persistence.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/persistence.py), [`src/deployment/predictor.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/deployment/predictor.py), [`app.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/app.py).

---

### Bước 1. Quy Trình Nghiệm Thu Độc Lập trên Test Set (293.717 Dòng)
- **Nguyên tắc quy đổi tiền tệ thực tế:**
  Dự báo log của mô hình được chuyển đổi ngược về thang đo tiền tệ BRL và kẹp chặn dưới bằng 0 để triệt tiêu các giá trị âm phi logic kinh tế:

$$
\hat{s} = \max\left(0.0, \; \exp(\hat{y}) - 1.0\right)
$$

- **Bộ chỉ số đo lường khoa học & kinh doanh:**
  1. **MSE ($\text{BRL}^2$):** Đo lường hàm mất mát bình phương trung bình trên thang tiền tệ.
  2. **MAE (BRL):** Đo độ lệch giá trị tuyệt đối trung bình giữa dự báo và thực tế.
  3. **RMSE (BRL):** Thước đo sai số tiền tệ chuẩn mực, phạt mạnh các lỗi lệch lớn trong kế hoạch nhập hàng.
  4. **$R^2$ Score:** Tỷ lệ phương sai thực tế của doanh số tuần mà mô hình giải thích được trên tập kiểm tra độc lập.
  5. **WAPE (Weighted Absolute Percentage Error):**
     Khắc phục thảm họa chia cho 0 của chỉ số MAPE truyền thống trên dữ liệu thưa (>90% số tuần có doanh số thực tế bằng 0 BRL):

$$
\mathrm{WAPE} = \frac{\sum_{i=1}^N |s_i - \hat{s}_i|}{\sum_{i=1}^N s_i}
$$

  6. **Ma trận Chi Phí Tổn Thất Chuỗi Cung Ứng (Supply Chain Cost Matrix):**
     Mô phỏng bài toán kinh tế bất đối xứng trong bán lẻ:
     - *Phạt tồn kho thừa (Overstock Penalty):* $2.0\text{ BRL}$ trên mỗi BRL dự báo dư thừa (chi phí lưu kho, chi phí vốn và hao mòn sản phẩm).
     - *Phạt đứt hàng (Understock Penalty):* $10.0\text{ BRL}$ trên mỗi BRL dự báo thiếu hụt (tổn thất cơ hội bán hàng, mất khách hàng vào tay đối thủ).

$$
\mathrm{Cost} = \frac{1}{N} \sum_{i=1}^N \left( 2.0 \cdot \max(0, \hat{s}_i - s_i) + 10.0 \cdot \max(0, s_i - \hat{s}_i) \right)
$$

- Lưu bảng so sánh xếp hạng tổng hợp ra tệp [`results/metrics/model_comparison.csv`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/results/metrics/model_comparison.csv).

---

### Bước 2. Xuất Bản Hệ Thống Biểu Đồ Đánh Giá Khoa Học (300 DPI)
- Triển khai trong [`src/evaluation/visualization.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/visualization.py) với chuẩn đồ họa xuất bản:
  - `model_comparison.png`: Biểu đồ cột đối sánh trực quan MSE, MAE, RMSE và $R^2$ giữa toàn bộ các mô hình From-Scratch và Baselines.
  - `actual_vs_predicted.png`: Biểu đồ phân tán đối chiếu giữa Doanh số Thực tế và Doanh số Dự báo.
  - `residuals_plot.png`: Đồ thị phân phối phần dư $r = s - \hat{s}$ và kiểm định tính ngẫu nhiên (chống hiện tượng Heteroscedasticity).
  - `feature_importance.png`: Đồ thị Top 15 đặc trưng quan trọng nhất của mô hình cây, làm rõ đóng góp của biến giá, biến trễ và danh mục hàng.

---

### Bước 3. Phân Tích Phần Dư Chuyên Sâu & Mổ Xẻ Top 20 Ca Sai Số Lớn Nhất
- Triển khai hàm `extract_top_worst_predictions()` trong [`src/evaluation/error_analysis.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/error_analysis.py).
- Trích xuất Top 20 đơn vị bản ghi có sai số tuyệt đối $|s - \hat{s}|$ lớn nhất trên tập Test và phân tích 3 nguyên nhân kinh tế cốt lõi:
  1. *Đột biến đơn hàng sỉ (B2B Bulk Orders):* Khách hàng doanh nghiệp bất ngờ đặt mua số lượng lớn sản phẩm giá trị cao trong một tuần đơn lẻ, vượt khỏi quy luật trượt của nhóm người tiêu dùng cá nhân thông thường.
  2. *Hiện tượng trễ pha do chiến dịch khuyến mãi chớp nhoáng (Flash Sales):* Doanh số bùng nổ cục bộ trong 1 tuần rồi trở về 0 ngay tuần kế tiếp, khiến mô hình dựa vào trễ tuần trước dự báo thừa (Over-prediction).
  3. *Hiện tượng Đứt Hàng Ẩn (Latent Stock-Out Bias):* Dữ liệu Olist ghi nhận giao dịch bán chứ không có dữ liệu tồn kho kho bãi thực tế (Out-of-Stock). Một tuần có doanh số bằng 0 có thể do không có nhu cầu, hoặc do người bán đã hết sạch hàng trong kho. Đây là một ranh giới dữ liệu cần lưu ý khi diễn giải mô hình.

---

### Bước 4. Đóng Gói Model Persistence & Kiểm Tra Tái Lập (Reproducibility)
- Triển khai hàm `save_artifacts()` trong [`src/models/persistence.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/persistence.py).
- Lưu gói artifact toàn diện vào [`results/saved_models/best_sales_model.pkl`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/results/saved_models), chứa:
  - Mô hình chiến thắng đã huấn luyện hoàn chỉnh (`model`).
  - Pipeline tiền xử lý đã fit tham số (`pipeline`).
  - Danh sách 89 tên đặc trưng cố định thứ tự (`feature_names`).
  - Metadata phiên bản, thời điểm lưu trữ và điểm số kiểm định.
- **Kiểm định tính tái lập 100%:** Nạp lại artifact qua `load_artifacts()` và chạy kiểm thử assertion: độ lệch giữa kết quả dự báo trước và sau khi lưu bằng đúng $0.00000000$.

---

### Bước 5. Mô Phỏng Kịch Bản Kinh Doanh & Vận Hành Streamlit App 5 Tabs
- Triển khai các kịch bản kinh doanh trong [`src/deployment/predictor.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/deployment/predictor.py):
  1. `predict_scenario_new_product()`: Dự báo doanh thu ra mắt sản phẩm mới (kết hợp cả proxy lịch sử tuần 0 và mô hình Cold-Start).
  2. `predict_scenario_marketing_campaign()`: Mô phỏng độ nhạy ngân sách quảng cáo với cơ chế lợi biên giảm dần.
  3. `calculate_discount_scenario()`: Bảng tính toán doanh thu và sản lượng bù đắp theo tỷ lệ chiết khấu giá bán.
- Vận hành giao diện Web tương tác trực quan 5 Thẻ chức năng qua [`app.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/app.py):
  - Khởi chạy ứng dụng: `.venv/Scripts/python.exe -m streamlit run app.py`.
  - *Thẻ 1:* **Một sản phẩm** (Dự báo tức thời cho 1 sản phẩm tuần đích kèm biểu đồ lịch sử).
  - *Thẻ 2:* **Nhiều sản phẩm từ CSV** (Dự báo hàng loạt theo lô lên đến 10.000 dòng, xuất file CSV kết quả).
  - *Thẻ 3:* **Mô phỏng giả định giảm giá** (Tính toán doanh thu và sản lượng giữ doanh thu tham chiếu).
  - *Thẻ 4:* **Sản phẩm mới (Cold-Start)** (Dự báo tuần ra mắt dựa trên đặc trưng nội dung).
  - *Thẻ 5:* **Mô phỏng Tiếp thị (Marketing Uplift)** (Thanh trượt AED, mô phỏng ROAS và đồ thị đường cong bão hòa).

---

### Bước 6. Mô Hình Hồi Quy Sản Phẩm Mới Mở Bán (Cold-Start Content-Based)
- Triển khai trong [`src/models/cold_start.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/cold_start.py) giải quyết bài toán cho sản phẩm chưa từng có lịch sử:
  - Khai thác tập dữ liệu 32.216 sản phẩm Olist ở tuần đầu tiên phát sinh giao dịch.
  - Xây dựng `CategoryMedianColdStartBaseline` làm mốc đối chuẩn trung vị ngành hàng.
  - Xây dựng `ColdStartProductRegressor` From-Scratch học trên 6 đặc trưng nội dung: giá niêm yết ban đầu, trọng lượng sản phẩm, số lượng ảnh, độ dài tiêu đề, độ dài mô tả, và danh mục hàng đã mã hóa.
  - Đánh giá độc lập trên tập sản phẩm chưa từng xuất hiện: $\text{MAE} = 39,79\text{ BRL}$ (vượt trội so với Baseline trung vị ngành $126,65\text{ BRL}$) và $R^2 = +0,8142$.

---

### Bước 7. Động Cơ Mô Phỏng Tiếp Thị Kinh Tế Lượng (Marketing Uplift & ROAS Engine)
- Triển khai trong [`src/deployment/predictor.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/deployment/predictor.py) qua hàm `simulate_marketing_response()`:
- **Mô hình đường cong bão hòa Log-Response:**
  Ứng dụng hàm phản hồi phi tuyến có hệ số co giãn quảng cáo AED ($\epsilon$ - Advertising Elasticity of Demand):

$$
\Delta \mathrm{Sales}(S) = S_{\mathrm{baseline}} \times \epsilon \times \ln\left(1 + \frac{S}{S_{\mathrm{ref}}}\right)
$$

- **Kiểm chứng định luật lợi biên giảm dần (Diminishing Marginal Returns):**
  - Khi ngân sách tiếp thị $S$ tăng lên các mốc rất lớn, doanh số gia tăng $\Delta \mathrm{Sales}$ tăng chậm dần (tiệm cận mức bão hòa thị trường), ngăn chặn hoàn toàn việc ngoại suy tuyến tính phi thực tế.
  - Cung cấp trọn bộ chỉ số kinh doanh phục vụ ra quyết định:
    - `Incremental_Sales_BRL`: Doanh thu gia tăng trực tiếp do tiếp thị mang lại.
    - `Sales_Growth_%`: Tỷ lệ tăng trưởng doanh số so với mức nền tảng ban đầu.
    - `ROAS`: Tỷ suất hoàn vốn trên chi tiêu quảng cáo ($\Delta \mathrm{Sales} / S$).
    - `Marketing_ROI_%`: Tỷ suất sinh lời ròng của chiến dịch tiếp thị.
