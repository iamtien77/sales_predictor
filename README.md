# 📈 Dự Báo Doanh Thu Sản Phẩm Theo Tuần (Weekly Product Sales Predictor)

> **Bài Lab Môn Học Máy (Machine Learning Lab Project)**
> **Phương pháp tiếp cận:** Cài đặt thuật toán từ đầu (From-Scratch) bằng **NumPy**, không sử dụng thư viện Scikit-Learn cho phần thuật toán mô hình và các bước tiền xử lý.
> **Tập dữ liệu:** Dữ liệu thương mại điện tử Olist Brazil (~1.2 triệu dòng sản phẩm–tuần) từ bộ dữ liệu công khai [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
> **Trạng thái kiểm thử:** 95/95 unit & integration tests pass (05/11/2026).

---

## 📌 1. Giới thiệu Bài toán

### 1.1. Bối cảnh & Mục tiêu Bài lab

Trong bài lab này, mục tiêu là xây dựng mô hình học máy để giải quyết bài toán: **Dự báo tổng doanh thu bán hàng (BRL) trong tuần tiếp theo cho từng sản phẩm** trên sàn thương mại điện tử Olist.

### 1.2. Dữ liệu & Ràng buộc Chuỗi Thời gian

- **Dạng bài toán:** Hồi quy chuỗi thời gian (Time-series Regression) trên bảng dữ liệu Sản phẩm – Tuần (*Product–Week Panel*).
- **Chu kỳ tuần:** Dữ liệu được tổng hợp theo tuần lịch từ **thứ Hai đến Chủ nhật**, mốc thời gian lấy ngày thứ Hai đầu tuần (`order_purchase_timestamp`).
- **Biến mục tiêu ($y$):** Tổng giá bán các món hàng (`price`) trong tuần đích, **không bao gồm phí vận chuyển** (`freight_value`). Số thứ tự món hàng (`order_item_id`) không phải là số lượng bán.
- **Xử lý tuần không phát sinh doanh số:** Với mỗi sản phẩm, từ tuần đầu tiên bán được đến tuần cuối quan sát, những tuần không bán được món nào vẫn được giữ lại với nhãn `sales = 0.0` BRL (phản ánh đặc tính thực tế của dữ liệu bán lẻ).
- **Không gian biến đổi:** Doanh thu có độ lệch lớn, nên mô hình được huấn luyện trong không gian Log:

```math
y_{\text{train}} = \log(1 + \text{sales})
```

  Khi dự báo và đánh giá, kết quả được chuyển ngược về đơn vị tiền tệ BRL và chặn dưới bằng 0:

```math
\hat{y}_{\text{sales}} = \max\left(0, \exp(\hat{y}) - 1\right)
```

---

## ⚙️ 2. Các Thuật Toán Tự Cài Đặt (From-Scratch bằng NumPy)

Bài lab yêu cầu tự cài đặt các lớp mô hình hồi quy, thuật toán tối ưu, độ đo đánh giá và pipeline tiền xử lý bằng NumPy/Pandas thuần túy, không dùng Scikit-Learn.

### 2.1. Năm Thuật toán Hồi quy

1. **Hồi quy Tuyến tính (`LinearRegressionFromScratch`):**
   - Giải phương trình OLS Normal Equation: $w = (X^T X + \alpha I)^{-1} X^T y$ với điều chuẩn Ridge nhỏ $\alpha_{\text{ridge}} = 10^{-5}$ để giữ tính ổn định số học.
   - Hỗ trợ thêm bộ giải Gradient Descent lặp (`solver='gradient_descent'`) để quan sát quá trình hội tụ.
   - Giới hạn dự báo log trong phạm vi hợp lý theo tập Train để tránh tràn số.
2. **Cây Quyết định (`DecisionTreeRegressorFromScratch`):**
   - Xây dựng cây hồi quy nhị phân đệ quy dựa trên tiêu chuẩn Giảm phương sai (Variance Reduction):

```math
\Delta \text{Var} = \text{Var}(D) - \left( \frac{|D_L|}{|D|} \text{Var}(D_L) + \frac{|D_R|}{|D|} \text{Var}(D_R) \right)
```

   - Sử dụng 16 điểm phân vị ứng viên (*Quantile Split Candidates*) để giảm thời gian tìm điểm chia.
3. **Rừng Ngẫu nhiên (`RandomForestRegressorFromScratch`):**
   - Thuật toán Bagging: Lấy mẫu bootstrap cho từng cây và chọn ngẫu nhiên tập con đặc trưng ($k = \sqrt{d}$) tại mỗi nút chia nhánh để giảm quá khớp.
4. **Gradient Boosting (`GradientBoostingRegressorFromScratch`):**
   - Huấn luyện tuần tự các cây hồi quy trên phần dư âm (Residuals) với hệ số học co hẹp (*learning rate*) $\eta \in (0, 1]$:

```math
\hat{y}^{(m)}(x) = \hat{y}^{(m-1)}(x) + \eta \cdot h_m(x)
```

5. **Two-Stage Hurdle (`TwoStageHurdleRegressorFromScratch`):**
   - Mô hình 2 giai đoạn phù hợp với dữ liệu có nhiều tuần không bán được hàng:
     - **Giai đoạn 1 (Phân loại):** Dự báo xác suất tuần đó có bán được hàng hay không $\Pr(y > 0 \mid x)$.
     - **Giai đoạn 2 (Hồi quy):** Cây hồi quy học trên các tuần có phát sinh doanh số dương ($y > 0$).
     - **Công thức tính:** $\hat{y} = \Pr(y > 0 \mid x) \cdot \mathrm{expm1}(\hat{y}_{log,pos})$.

### 2.2. Các Thử nghiệm Mở rộng

1. **Thử nghiệm Sản phẩm Mới (`ColdStartProductRegressor`):**
   - Thử nghiệm ước lượng doanh thu 7 ngày đầu kể từ giao dịch đầu tiên dựa trên giá bán và thuộc tính sản phẩm.
   - Bắt buộc đối chiếu với baseline giá bán đơn vị (`sales = first_price`) và trung vị danh mục; loại bỏ các mẫu có nhãn vượt ranh giới thời gian.
2. **Mô phỏng Kịch bản Tiếp thị (`simulate_marketing_response`):**
   - Mô phỏng sự thay đổi của doanh số khi thay đổi ngân sách quảng cáo dựa trên hàm phản hồi logarit:

```math
\text{Uplift Factor} = 1.0 + \beta_{\text{ad}} \cdot \ln\left(1 + \frac{\text{Ad Spend}}{S_{ref}}\right)
```

   - Tham số mang tính chất giả định để làm quen với mô phỏng độ nhạy, không đại diện cho dữ liệu chiến dịch thực tế.

### 2.3. Bốn Mô hình Đối chứng Cơ sở (Baselines)

Để đánh giá mô hình học máy có thực sự mang lại hiệu quả hay không, bài lab đối chiếu với 4 baseline:

- **Zero Baseline (`ZeroRegressorFromScratch`):** Luôn dự báo 0 BRL (mốc chuẩn trên dữ liệu thưa).
- **Revenue Mean Baseline (`RevenueMeanRegressorFromScratch`):** Dự báo hằng số bằng trung bình doanh thu thực tế của tập Train.
- **Log Mean Baseline (`MeanRegressorFromScratch`):** Dự báo trung bình nhãn trong không gian log.
- **Previous Week Baseline (`PreviousWeekRegressorFromScratch`):** Lấy doanh thu tuần trước làm dự báo cho tuần này.

### 2.4. Pipeline Tiền Xử lý Thuần NumPy

- `CustomImputer`: Điền giá trị khuyết thiếu (Median cho biến số, Mode cho biến phân loại).
- `CustomOneHotEncoder`: Mã hóa các danh mục sang ma trận nhị phân thưa (Sparse CSR Matrix), gán vector 0 cho nhãn mới ở tập Test.
- `CustomStandardScaler`: Chuẩn hóa Z-Score $Z = (X - \mu) / (\sigma + 10^{-8})$.
- `CustomDataPipeline`: Đóng gói quy trình tiền xử lý thành một chuỗi thống nhất.
- **Quy tắc quan trọng:** Toàn bộ pipeline tiền xử lý chỉ được `fit` trên tập Train (`train_df`). Các tập Validation và Test chỉ áp dụng `transform` dựa trên tham số đã học từ Train để tránh rò rỉ dữ liệu (Data Leakage).

---

## 📊 3. Hệ thống Đặc trưng (89 Features)

Các đặc trưng trễ (lag) và trượt (rolling) được tính từ các tuần trước tuần dự báo.

| Nhóm đặc trưng | Tên thuộc tính | Ý nghĩa & Quy tắc tính |
| :--- | :--- | :--- |
| **Giá bán** | `price` | Giá bán trung bình tuần gần nhất có giao dịch trước tuần đích. |
| **Doanh thu trễ** | `last_week_sales` | Doanh thu tuần liền kề trước tuần dự báo ($t-1$). |
| | `sales_lag_2_weeks`, `sales_lag_4_weeks` | Doanh thu cách 2 và 4 tuần lịch ($t-2, t-4$). |
| **Thống kê trượt** | `sales_mean_4_weeks`, `sales_std_4_weeks` | Trung bình và độ lệch chuẩn doanh thu 4 tuần gần nhất. |
| **Sản lượng** | `last_week_units` | Tổng số lượng sản phẩm bán được ở tuần liền trước. |
| **Vòng đời** | `weeks_since_first_sale` | Số tuần kể từ lần bán hàng đầu tiên của sản phẩm. |
| **Đánh giá lịch sử** | `past_review_count`, `past_review_rating` | Số lượt và điểm đánh giá trung bình trước tuần dự báo. |
| **Khách hàng** | `customer_past_spend`, `customer_past_orders` | Chi tiêu và số đơn lịch sử của nhóm khách mua hàng. |
| | `customer_state_count` | Số lượng bang của nhóm khách mua hàng. |
| **Thời gian & Mùa vụ** | `month`, `quarter`, `week_of_year` | Tháng, Quý và Tuần trong năm của tuần dự báo. |
| | `is_peak_shopping_season` | Đánh dấu các tuần mua sắm cao điểm cuối năm. |
| **Danh mục** | `category_*` | Vector One-Hot đại diện cho các danh mục sản phẩm Olist. |

---

## 🏆 4. Kết quả thực nghiệm hiện tại

<!-- GENERATED RESULTS: python -m src.evaluation.reporting -->
Các bảng dưới được sinh trực tiếp từ CSV trong `results/metrics` và artifact triển khai.
Chạy `.venv/Scripts/python.exe -m src.evaluation.reporting` để đồng bộ sau thực nghiệm.
Mô hình tuần được chọn bằng Validation: **Two-Stage Hurdle**, Validation RMSE **55.9393 BRL**.
Chế độ huấn luyện tuần: **full**. Thời điểm huấn luyện tuần được ghi trong metadata: `2026-11-05 10:23:00`.
Đây là snapshot của lần chạy đã lưu, không trộn số liệu từ lần chạy khác.

### 4.1. Dự báo sản phẩm–tuần (Test, BRL)

| Model | MSE | MAE | RMSE | R2 | WAPE | Cost_Per_Item_BRL | Fit Time (s) | Target_Scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Two-Stage Hurdle | 2423.7333 | 8.4426 | 49.2314 | 0.1148 | 1.9730 | 44.7952 | 33.6800 | BRL |
| Tuned Model | 2541.2236 | 4.2989 | 50.4106 | 0.0719 | 1.0046 | 40.6158 | 0.0000 | BRL |
| Decision Tree | 2615.3331 | 4.3274 | 51.1403 | 0.0449 | 1.0113 | 41.2640 | 15.0600 | BRL |
| Gradient Boosting | 2708.2243 | 4.4176 | 52.0406 | 0.0109 | 1.0323 | 42.4902 | 153.0500 | BRL |
| Random Forest | 2725.3346 | 4.4322 | 52.2047 | 0.0047 | 1.0357 | 42.6916 | 41.8000 | BRL |
| Revenue Mean Baseline | 2752.2779 | 11.8955 | 52.4622 | -0.0052 | 2.7798 | 56.3489 | 0.0000 | BRL |
| Mean Baseline | 2754.5202 | 4.5026 | 52.4835 | -0.0060 | 1.0522 | 43.1895 | 0.0000 | BRL |
| Zero Baseline | 2756.4820 | 4.2792 | 52.5022 | -0.0067 | 1.0000 | 42.7919 | 0.0000 | BRL |
| Previous Week Baseline | 5683.5250 | 9.0765 | 75.3892 | -1.0757 | 2.1211 | 42.4533 | 0.0000 | BRL |
| Linear Regression | 30378515.8542 | 44.3768 | 5511.6709 | -11093.4570 | 10.3704 | 120.7011 | 0.5200 | BRL |

RMSE và MAE phản ánh các mục tiêu khác nhau; không suy ra lợi ích tồn kho chỉ từ R².
Chi phí là điểm phạt giả định trên sai lệch doanh thu, không phải chi phí vận hành được đo thực tế.
Chưa có đánh giá riêng precision/recall cho phát hiện tuần nhu cầu đột biến.
Kết quả tuần dùng snapshot `delivered` cuối kỳ, chưa tái dựng trạng thái đơn hàng tại từng forecast origin; chưa đo ảnh hưởng của hạn chế này lên metrics.

### 4.2. Proxy bảy ngày từ giao dịch đầu tiên

Nguồn: `cold_start_evaluation.csv`, quy trình: `cold_start_manifest.json`.
Train: **23428** sản phẩm; Test: **8110**;
loại **483** mẫu Train có cửa sổ nhãn vượt mốc **2018-04-24T00:00:00**.
Giá là giá giao dịch đầu tiên. Mọi cửa sổ nhãn phải có đủ bảy ngày quan sát.

| Model | MAE_BRL | RMSE_BRL | R2_Score | Test_Count |
| --- | --- | --- | --- | --- |
| CategoryMedianColdStartBaseline | 126.6543 | 313.9562 | 0.0022 | 8110 |
| SingleUnitPriceBaseline | 23.0145 | 120.9134 | 0.8520 | 8110 |
| ColdStartProductRegressor | 39.7939 | 135.4682 | 0.8142 | 8110 |

So sánh cả baseline danh mục và baseline một sản phẩm (`sales = first_price`).
Không chọn mô hình bằng Test: cây với tham số cố định là ứng viên thực nghiệm được lưu,
không được gọi là tốt nhất nếu baseline thắng. Tỷ lệ Test chỉ bán một đơn vị: **85.47%**.

Với sản phẩm chỉ bán một đơn vị trong cửa sổ bảy ngày, target bằng đúng giá giao dịch đầu đã biết, nên baseline `sales = first_price` dự báo chính xác. Tỷ lệ lớn của nhóm này giải thích vì sao baseline là đối chứng mạnh; cần đọc MAE và RMSE trong bảng trên để xác định mô hình nào thắng trong lần chạy được báo cáo, không suy rộng cho mọi tập dữ liệu. Cây gom nhiều sản phẩm vào mỗi lá và học trên `log1p(sales)`, nên không bắt buộc giữ quan hệ `sales = first_price`; phép nghịch đảo log cũng không tối ưu trực tiếp MSE trên BRL. Đây là các cơ chế có thể góp phần gây sai số, chưa phải kết luận nhân quả từ một thí nghiệm ablation.

Giới hạn: mốc bắt đầu là lần bán đầu, không phải ngày niêm yết; target bao gồm giao dịch đầu đã biết.
Không có sản phẩm chưa bán; metadata sản phẩm và trạng thái delivered là snapshot cuối,
chưa có lịch sử để xác minh chúng sẵn có tại thời điểm dự báo. Chưa chứng minh cold-start trước bán.
<!-- END GENERATED RESULTS -->

## 📁 5. Cấu trúc Thư mục Dự án

```
lab_2 sales_predictor/
│
├── data/                               # Dữ liệu dự án
│   ├── raw/                            # Dữ liệu gốc Olist từ Kaggle & file olist_merged_sales_dataset.csv
│   └── processed/                      # Cache ma trận CSR thưa (X_train, y_train, ...) và pipeline
│
├── docs/                               # Tài liệu thiết kế và báo cáo bài lab
│   ├── algorithms/                     # Tài liệu chi tiết các mô hình From-Scratch và baseline
│   ├── report/                         # Báo cáo tổng kết bài lab môn học
│   ├── description_for_AI_working/     # Nhật ký kỹ thuật và tiến trình các phiên bản
│   └── ML project_ Predicting Product Sales.pdf # Đề bài gốc của môn học
│
├── src/                                # Toàn bộ mã nguồn cốt lõi (From-Scratch)
│   ├── config.py                       # Cấu hình tập trung đường dẫn, tên cột, siêu tham số
│   ├── data/                           # Module nạp, làm sạch, tạo bảng tuần và kỹ nghệ đặc trưng
│   │   ├── builder.py                  # Ghép nối bảng Olist gốc & xây dựng bảng sản phẩm–tuần
│   │   ├── loader.py                   # Nạp dữ liệu an toàn & kiểm tra chất lượng
│   │   ├── preprocessing.py            # CustomImputer, CustomOneHotEncoder, CustomStandardScaler, Pipeline
│   │   └── feature_engineering.py      # WeeklySalesFeatureEngineer (lag, rolling, review, rfm)
│   ├── models/                         # Module mô hình From-Scratch, Baselines & Tối ưu hóa
│   │   ├── base.py                     # Lớp cơ sở trừu tượng BaseRegressor, BaseClassifier
│   │   ├── baselines.py                # 4 Mô hình Baseline đối chuẩn (Zero, Mean, Log-Mean, PreviousWeek)
│   │   ├── linear.py                   # Hồi quy tuyến tính LinearRegressionScratch & Adaptive Clamping
│   │   ├── tree.py                     # Cây quyết định DecisionTreeRegressorScratch (Variance Reduction)
│   │   ├── ensemble.py                 # Mô hình tổ hợp RandomForestRegressorScratch & GradientBoosting
│   │   ├── hurdle.py                   # Mô hình 2 giai đoạn TwoStageHurdleRegressor
│   │   ├── cold_start.py               # Mô hình Cold-Start & Category Median Baseline
│   │   ├── regressors.py               # Giao diện hợp nhất các mô hình hồi quy
│   │   ├── tuning.py                   # grid_search_from_scratch & tune_all trên Validation
│   │   └── persistence.py              # Đóng gói và nạp mô hình (pickle serialization)
│   ├── evaluation/                     # Module đánh giá, phân tích lỗi và trực quan hóa
│   │   ├── metrics.py                  # Tính MSE, MAE, RMSE, R2, WAPE thuần NumPy
│   │   ├── error_analysis.py           # Phân tích phần dư & mổ xẻ Top 20 đơn hàng sai số lớn nhất
│   │   └── visualization.py            # Vẽ các biểu đồ EDA & đánh giá mô hình
│   ├── deployment/                     # Module suy luận thực tế
│   │   └── predictor.py                # predict_sales đơn/batch, Cold-Start & kịch bản tiếp thị
│   └── utils/                          # Công cụ hỗ trợ
│       └── logger.py                   # Hệ thống ghi nhật ký thực thi (Logging)
│
├── notebooks/                          # Hệ thống 5 Jupyter Notebooks theo từng giai đoạn
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_training_and_comparison.ipynb
│   ├── 04_hyperparameter_tuning_and_error_analysis.ipynb
│   └── 05_scenario_prediction_and_demo.ipynb
│
├── tests/                              # Bộ kiểm thử tự động (Pytest)
│   ├── conftest.py                     # Cấu hình fixtures kiểm thử
│   ├── test_preprocessing.py           # Kiểm thử Imputer, OneHot, Scaler, Temporal Split
│   ├── test_feature_engineering.py     # Kiểm thử các biến trễ, rolling, tương quan
│   ├── test_models.py                  # Kiểm thử toán học 4 thuật toán From-Scratch & GridSearch
│   ├── test_hurdle_and_metrics.py      # Kiểm thử mô hình Two-Stage Hurdle và hàm độ đo WAPE
│   ├── test_cold_start_and_marketing.py# Kiểm thử mô hình Cold-Start và phản hồi Marketing
│   ├── test_deployment.py              # Kiểm thử hàm dự báo và mô phỏng kịch bản
│   ├── test_weekly_forecasting.py      # Kiểm thử toàn vẹn chuỗi thời gian tuần, zero-sales
│   ├── test_requirement_updates.py     # Kiểm thử chống rò rỉ review, thang đo BRL, tuning
│   └── test_pipeline_contracts.py      # Kiểm thử hợp đồng tham số đầu vào
│
├── results/                            # Kết quả chính thức từ main.py pipeline
│   ├── figures/                        # Các biểu đồ xuất bản (EDA, so sánh mô hình, phần dư)
│   ├── metrics/                        # Bảng số liệu: model_comparison.csv, tuning_history.csv
│   └── saved_models/                   # Artifact mô hình tối ưu đã lưu
│
├── main.py                             # Điểm khởi chạy tự động toàn bộ luồng bài lab (CLI)
├── app.py                              # Ứng dụng Web tương tác Streamlit (Demo & Thử nghiệm)
├── requirements.txt                    # Danh sách thư viện phụ thuộc
└── README.md                           # Hướng dẫn tổng quan bài lab (File hiện tại)
```

---

## 🚀 6. Quy trình Bài Lab trong `main.py`

Toàn bộ quy trình từ dữ liệu thô đến mô hình hoàn chỉnh được điều phối tự động qua các bước rõ ràng trong [main.py](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/main.py):

```mermaid
flowchart TD
    S1["Bước 1: Nạp dữ liệu giao dịch Olist gốc"] --> S2["Bước 2: Lọc dữ liệu giao dịch thô"]
    S2 --> S3["Bước 3: Tổng hợp bảng chuỗi thời gian sản phẩm–tuần"]
    S3 --> S4["Bước 4: Khám phá Dữ liệu EDA trên toàn bộ dữ liệu"]
    S4 --> S_split["Bước Split: Phân chia Chuỗi thời gian 80/10/10 trước khi tiền xử lý"]
    S_split --> S5["Bước 5: Tiền xử lý & Kỹ nghệ đặc trưng (Fit DUY NHẤT trên Train)"]
    S5 --> S6["Bước 6: Huấn luyện các mô hình From-Scratch & Baselines"]
    S6 --> S7["Bước 7: Tinh chỉnh siêu tham số GridSearchCV trên Validation 10%"]
    S7 --> S8["Bước 8: Đánh giá độc lập trên tập Test 10% (Thang đo BRL)"]
    S8 --> S9["Bước 9: Phân tích phần dư & sai số dự báo"]
    S9 --> S10["Bước 10: Lưu mô hình và chạy thử các kịch bản dự báo"]
```

---

## 🖥️ 7. Ứng dụng Web Demo Streamlit (`app.py`)

Ứng dụng Web hỗ trợ kiểm tra và thử nghiệm trực quan kết quả của bài lab qua 5 tab:

1. **Dự báo sản phẩm đơn lẻ (Tab 1):** Nhập ngày tuần, danh mục, giá bán và lịch sử để xem dự báo doanh số tuần kế tiếp.
2. **Dự báo hàng loạt (Tab 2):** Tải tệp CSV danh sách sản phẩm để nhận tệp dự báo kết quả.
3. **Mô phỏng giả định giảm giá (Tab 3):** Tính toán doanh thu và sản lượng giả định theo tỷ lệ chiết khấu.
4. **Thử nghiệm sản phẩm mới (Tab 4):** Ước lượng doanh thu 7 ngày đầu từ các thuộc tính sản phẩm và giá bán.
5. **Mô phỏng ngân sách tiếp thị (Tab 5):** Thử nghiệm đường cong phản hồi doanh số khi thay đổi chi phí quảng cáo.

Khởi chạy ứng dụng:

```powershell
.venv/Scripts/python.exe -m streamlit run app.py
```

---

## 📓 8. Hệ thống Jupyter Notebooks (`notebooks/`)

Các notebook được thiết kế theo đúng thứ tự logic của bài lab:

1. [01_exploratory_data_analysis.ipynb](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/notebooks/01_exploratory_data_analysis.ipynb): Khám phá dữ liệu (EDA) trên toàn bộ bảng sản phẩm–tuần, sau đó phân chia chuỗi thời gian 80/10/10.
2. [02_preprocessing_and_feature_engineering.ipynb](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/notebooks/02_preprocessing_and_feature_engineering.ipynb): Xây dựng pipeline tiền xử lý: fit duy nhất trên Train, transform sang Validation và Test.
3. [03_model_training_and_comparison.ipynb](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/notebooks/03_model_training_and_comparison.ipynb): Huấn luyện các mô hình From-Scratch và so sánh với baseline trên tập Validation.
4. [04_hyperparameter_tuning_and_error_analysis.ipynb](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/notebooks/04_hyperparameter_tuning_and_error_analysis.ipynb): Tinh chỉnh siêu tham số trên Validation, đánh giá trên Test và phân tích sai số.
5. [05_scenario_prediction_and_demo.ipynb](file:///e:/Hoc_May_Tren_truong/lab_2%20sales_predictor/notebooks/05_scenario_prediction_and_demo.ipynb): Thử nghiệm các kịch bản dự báo (sản phẩm mới, ngân sách tiếp thị, giảm giá).

---

## 🛠️ 9. Hướng dẫn Cài đặt & Chạy Bài Lab

### 9.1. Môi trường

- Python: 3.10, 3.11 hoặc 3.12.
- Thư viện cần thiết: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `streamlit`, `pytest`.

### 9.2. Cài đặt

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv/Scripts/Activate.ps1

# Cài đặt thư viện
pip install -r requirements.txt
```

### 9.3. Các Câu Lệnh Chính

| Mục đích | Câu lệnh PowerShell |
| :--- | :--- |
| **Chạy toàn bộ pipeline bài lab** | `.venv/Scripts/python.exe main.py` |
| **Chạy chế độ nhanh (Fast Mode)** | `.venv/Scripts/python.exe main.py --fast-mode --skip-tuning` |
| **Chạy ứng dụng Web demo** | `.venv/Scripts/python.exe -m streamlit run app.py` |
| **Chạy toàn bộ bài kiểm thử (Full Test)** | `.venv/Scripts/python.exe -m pytest tests -v` |
| **Chạy kiểm thử nhanh (bỏ qua tuning)** | `.venv/Scripts/python.exe -m pytest tests -k "not test_tuning and not test_experiments" -v` |

---

## 📌 10. Giới hạn của Bài Lab & Dữ liệu

Một số điểm hạn chế cần lưu ý:
1. **Dữ liệu tiếp thị:** Tập dữ liệu Olist không có chi phí quảng cáo thực tế; phần mô phỏng tiếp thị chỉ là bài toán giả định theo hàm logarit để làm quen với mô phỏng độ nhạy.
2. **Dữ liệu sản phẩm mới (Cold-Start):** Dữ liệu không ghi nhận ngày niêm yết của sản phẩm mà chỉ có ngày phát sinh giao dịch đầu tiên; do đó bài lab sử dụng mốc 7 ngày sau lần bán đầu tiên làm proxy ước lượng.
3. **Tuần không bán được hàng:** Dữ liệu chưa phân biệt được tuần có doanh số bằng 0 là do sản phẩm hết hàng trong kho hay do không có người mua.

---

## 📫 Thông tin Liên hệ

| Thông tin | Chi tiết |
| :--- | :--- |
| 🧑‍💻 **Tác giả:** | **Nguyễn Sanh Tiến** |
| 📧 **Email:** | [nguyensanhtien2411@gmail.com](mailto:nguyensanhtien2411@gmail.com) |
| 📱 **Điện thoại:** | `0374900253` |
| 🎓 **Trường đại học:** | **Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh (UTH)** |
