# 🧭 Mục Lục Toàn Diện: Hướng Dẫn Kỹ Thuật Dự Án (Description for AI Working)

Thư mục này chứa toàn bộ hệ thống các kịch bản mô tả kỹ thuật chi tiết theo từng phiên bản tiến hóa (`v1` đến `vn`), tái hiện đầy đủ tiến trình lặp (Iterative Progression) của dự án **Dự Báo Doanh Thu Sản Phẩm Theo Tuần (Weekly Product Sales Predictor)** tại [`main.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/main.py) và thư mục [`src/`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src).

Mỗi tài liệu được chuẩn hóa thành các hành động kỹ thuật cụ thể theo cấu trúc: `Bước 1. ...`, `Bước 2. ...` quy định rõ đầu vào, logic xử lý thuật toán, kiểm soát lỗi biên, phòng chống rò rỉ dữ liệu chuỗi thời gian (Anti-Leakage) và số liệu kiểm chứng thực nghiệm.

---

## 1. Bảng Điều Phối Theo Giai Đoạn Lớn (4 Phase Summaries)

| Giai đoạn | File tổng quan giai đoạn | Hàm điều phối trong `main.py` | Module mã nguồn chính |
| :---: | :--- | :--- | :--- |
| **GĐ 1** | 📄 [01_data_preprocessing_and_audit.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/01_data_preprocessing_and_audit.md) | `step_1_load_and_explore()`<br>`step_2_clean_and_preprocess()`<br>`step_3_build_weekly_panel()`<br>`step_5_split_and_scale()` | [`src/data/loader.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/loader.py)<br>[`src/data/builder.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/builder.py)<br>[`src/data/preprocessing.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/preprocessing.py) |
| **GĐ 2** | 📄 [02_feature_engineering_and_selection.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/02_feature_engineering_and_selection.md) | `step_3_build_weekly_panel()`<br>`step_4_run_eda()`<br>`step_5_split_and_scale()` | [`src/data/feature_engineering.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/data/feature_engineering.py)<br>[`src/evaluation/visualization.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/visualization.py) |
| **GĐ 3** | 📄 [03_model_training_and_tuning.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/03_model_training_and_tuning.md) | `step_6_train_models()`<br>`step_7_hyperparameter_tuning()` | [`src/models/regressors.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/regressors.py)<br>[`src/models/tree.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tree.py)<br>[`src/models/ensemble.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/ensemble.py)<br>[`src/models/linear.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/linear.py)<br>[`src/models/hurdle.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/hurdle.py)<br>[`src/models/tuning.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/tuning.py) |
| **GĐ 4** | 📄 [04_evaluation_and_deployment.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/04_evaluation_and_deployment.md) | `step_8_evaluate_and_compare()`<br>`step_9_error_analysis()`<br>`step_10_persistence_and_demo()` | [`src/evaluation/metrics.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/metrics.py)<br>[`src/evaluation/visualization.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/visualization.py)<br>[`src/evaluation/error_analysis.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/evaluation/error_analysis.py)<br>[`src/models/persistence.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/models/persistence.py)<br>[`src/deployment/predictor.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/src/deployment/predictor.py)<br>[`app.py`](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/app.py) |

---

## 2. Hệ Thống 39 Phiên Bản Tiến Hóa Chi Tiết (Iterative Progression Files)

### 📊 Nhóm 1: Khám Phá & Phân Tích Dữ Liệu (EDA - 6 bản)
- [EDAv1.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/EDAv1.md): Nạp 6 bảng dữ liệu Olist, khảo sát schema giao dịch và cấu trúc item.
- [EDAv2.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/EDAv2.md): Chuẩn hóa thời gian `order_purchase_timestamp` và cấu trúc bảng Sản phẩm – Tuần.
- [EDAv3.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/EDAv3.md): Phân tích phân phối doanh thu tuần lệch phải cực đoan và kiểm chứng phép biến đổi `log1p(sales)`.
- [EDAv4.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/EDAv4.md): Phân tích hiện tượng dữ liệu thưa và lạm phát số không (Zero-Inflation: >90% tuần bằng 0 BRL).
- [EDAv5.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/EDAv5.md): Phân tích biến động doanh thu theo mùa vụ, ngày trong tuần, tháng trong năm và mùa Black Friday.
- [EDAv6.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/EDAv6.md): Phân tích phân bố 73 danh mục ngành hàng Olist và ma trận tương quan đặc trưng.

### 🧹 Nhóm 2: Tiền Xử Lý & Chuẩn Hóa Chuỗi Thời Gian (Processing Data - 5 bản)
- [processing-data-v1.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/processing-data-v1.md): Nạp và kiểm toán 6 bảng dữ liệu nguồn Olist, chuyển đổi kiểu `datetime64[ns]`.
- [processing-data-v2.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/processing-data-v2.md): Làm sạch nguồn item trước aggregation qua `builder.clean_weekly_source_data()`, không kẹp nhãn mục tiêu.
- [processing-data-v3.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/processing-data-v3.md): Xây dựng bảng chuỗi thời gian Sản phẩm – Tuần (Weekly Panel Builder) bảo toàn các tuần 0 BRL (~1,2M bản ghi).
- [processing-data-v4.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/processing-data-v4.md): Phân chia Temporal Split 80/10/10 theo tuần lịch (Train: 655k | Val: 249k | Test: 293k) chống rò rỉ dữ liệu.
- [processing-data-v5.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/processing-data-v5.md): Pipeline tiền xử lý một chiều `CustomDataPipeline` (Median Imputer, One-Hot 73 danh mục, Z-Score Scaler) chỉ fit trên Train.

### ⚙️ Nhóm 3: Kỹ Nghệ Đặc Trưng Chuỗi Thời Gian (Feature Engineering - 8 bản)
- [feature-engineering-v1.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/feature-engineering-v1.md): Xây dựng cấu trúc ma trận đặc trưng thưa SciPy CSR / Dense NumPy tối ưu bộ nhớ cho 1,2 triệu mẫu.
- [feature-engineering-v2.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/feature-engineering-v2.md): Kỹ nghệ nhóm biến Quán tính trễ (`last_week_sales`, `sales_lag_2_weeks`, `sales_lag_4_weeks`).
- [feature-engineering-v3.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/feature-engineering-v3.md): Kỹ nghệ nhóm biến Thống kê trượt (`sales_mean_4_weeks`, `sales_std_4_weeks`).
- [feature-engineering-v4.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/feature-engineering-v4.md): Kỹ nghệ biến Sản lượng & Vòng đời sản phẩm (`last_week_units`, `weeks_since_first_sale`).
- [feature-engineering-v5.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/feature-engineering-v5.md): Kỹ nghệ biến Phản hồi khách hàng Anti-Leakage (`past_review_count`, `past_review_rating`).
- [feature-engineering-v6.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/feature-engineering-v6.md): Kỹ nghệ biến Hành vi khách hàng & Độ phủ địa lý (`customer_past_spend`, `customer_past_orders`, `customer_state_count`).
- [feature-engineering-v7.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/feature-engineering-v7.md): Kỹ nghệ biến Mùa vụ, Ngày lễ & Tuần hoàn lượng giác (`is_peak_shopping_season`, `is_holiday_week`, `month_sin`, `month_cos`).
- [feature-engineering-v8.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/feature-engineering-v8.md): Ghép nối toàn diện vector 89 đặc trưng, lưu trữ ma trận đệm cache `X_train.npz`, `y_train.npy` và kiểm toán tính toàn vẹn.

### 🤖 Nhóm 4: Huấn Luyện Mô Hình Hồi Quy From-Scratch (Model Training - 5 bản)
- [train-v1.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/train-v1.md): Thiết kế kiến trúc học máy From-Scratch thuần NumPy và hàm mất mát trong không gian log.
- [train-v2.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/train-v2.md): Huấn luyện hệ thống 4 mô hình cơ sở đối chuẩn (Zero, Revenue Mean, Log Mean, Persistence Baseline).
- [train-v3.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/train-v3.md): Huấn luyện Hồi quy Tuyến tính `LinearRegressionFromScratch` (Normal Equation OLS + Adaptive Clamping chống nổ số).
- [train-v4.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/train-v4.md): Huấn luyện Cây quyết định hồi quy `DecisionTreeRegressorFromScratch` (Variance Reduction + 16 Quantile Splits).
- [train-v5.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/train-v5.md): Huấn luyện các mô hình tổ hợp: `RandomForestRegressorFromScratch` (Bagging) và `GradientBoostingRegressorFromScratch` (Boosting).

### 🎯 Nhóm 5: Tinh Chỉnh Siêu Tham Số & Mô Hình Hai Giai Đoạn (Hyperparameter Tuning - 6 bản)
- [tuning-v1.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/tuning-v1.md): Thiết lập chiến lược tinh chỉnh siêu tham số trên tập Validation 10% độc lập theo thời gian (Holdout Validation).
- [tuning-v2.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/tuning-v2.md): Thiết kế lưới tham số GridSearchCV From-Scratch cho Random Forest (`n_estimators`, `max_depth`, `min_samples_split`).
- [tuning-v3.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/tuning-v3.md): Thiết kế lưới tham số GridSearchCV From-Scratch cho Decision Tree (`max_depth`, `min_samples_leaf`).
- [tuning-v4.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/tuning-v4.md): Thiết kế lưới tham số GridSearchCV From-Scratch cho Gradient Boosting (`n_estimators`, `max_depth`, `learning_rate`).
- [tuning-v5.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/tuning-v5.md): Nghiên cứu và huấn luyện mô hình Hai giai đoạn `TwoStageHurdleRegressorFromScratch` (Occurrence Stage $\times$ Severity Stage).
- [tuning-v6.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/tuning-v6.md): Tổng hợp kết quả Tuning, lưu nhật ký `tuning_history.csv` và lựa chọn mô hình chiến thắng chung cuộc.

### 📈 Nhóm 6: Đánh Giá Đa Chiều & Triển Khai Thực Tế (Evaluation & Deployment - 9 bản)
- [evaluation-v1.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/evaluation-v1.md): Thiết lập quy trình nghiệm thu độc lập trên 293.717 dòng Test Set và nguyên tắc nghịch đảo tiền tệ BRL.
- [evaluation-v2.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/evaluation-v2.md): Tính toán bộ 4 chỉ số hồi quy tiêu chuẩn: MSE, MAE, RMSE, $R^2$ Score trên thang đo tiền tệ thực tế.
- [evaluation-v3.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/evaluation-v3.md): Tính toán chỉ số chuỗi cung ứng thương mại điện tử: WAPE (Weighted Absolute Percentage Error).
- [evaluation-v4.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/evaluation-v4.md): Mô phỏng ma trận tổn thất tài chính thực tế: Supply Chain Cost Per Item (chi phí tồn kho vs đứt hàng).
- [evaluation-v5.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/evaluation-v5.md): Phân tích phần dư Residuals và mổ xẻ kinh tế Top 20 đơn hàng có sai số dự báo lớn nhất.
- [evaluation-v6.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/evaluation-v6.md): Đóng gói Model Artifact `best_sales_model.pkl` và kiểm tra khả năng tái lập 100%.
- [evaluation-v7.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/evaluation-v7.md): Triển khai dịch vụ dự báo 3 kịch bản kinh doanh và xây dựng giao diện tương tác Streamlit `app.py`.
- [evaluation-v8.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/evaluation-v8.md): Xây dựng mô hình Cold-Start Product Regressor dự báo tuần mở bán đầu tiên (Launch Week) từ đặc trưng nội dung (Content-Based) cho sản phẩm mới hoàn toàn.
- [evaluation-v9.md](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/docs/description_for_AI_working/evaluation-v9.md): Xây dựng mô hình Kinh tế lượng phản hồi Marketing phi tuyến (Diminishing Returns & Log-Response) đo lường doanh thu gia tăng (Incremental Sales) và hiệu quả sinh lời (ROAS, Marketing ROI).
