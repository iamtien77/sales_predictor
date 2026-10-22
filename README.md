# 📊 Predicting Product Sales

## 1. Mô tả bài toán

Một công ty e-commerce muốn **dự đoán doanh số bán hàng** (sales) của sản phẩm nhằm **tối ưu hóa quản lý kho** và **chiến lược marketing**.

Đây là bài toán **hồi quy (regression)** — dự đoán một biến liên tục (sales) dựa trên nhiều đặc trưng đầu vào.

### Dữ liệu đầu vào
- **Dataset chính thức**: **Brazilian E-Commerce Public Dataset by Olist** (Kaggle)
  - **File nguồn**: `data/raw/olist_customers_dataset.csv` (~99,442 dòng dữ liệu khách hàng thực tế)
  - **Cấu trúc trường dữ liệu gốc**:
    - `customer_id`: Khóa định danh của khách hàng cho từng đơn hàng
    - `customer_unique_id`: Định danh duy nhất của khách hàng (dùng để tính toán tần suất mua lại, lịch sử mua sắm và CLV)
    - `customer_zip_code_prefix`: Mã bưu chính khu vực sinh sống của khách hàng
    - `customer_city`: Thành phố của khách hàng
    - `customer_state`: Bang/khu vực địa lý của khách hàng
- **Các nhóm đặc trưng theo đề bài**:
  - **Product information & Popularity**: Danh mục sản phẩm (`category`), giá (`price`), mô tả (`description`), số lượng đánh giá (`num_reviews`), điểm đánh giá (`rating`), điểm phổ biến (`popularity_score`)
  - **Customer demographics & Location**: Vị trí địa lý (`location` / `customer_city` / `customer_state`), độ tuổi (`age`), nhóm tuổi (`age_group`), giới tính (`gender`)
  - **Purchase history & CLV**: Số lần mua trước (`previous_purchases`), tần suất mua (`purchase_frequency`), số ngày mua gần nhất (`recency_days`), giá trị đơn trung bình (`avg_order_value`), giá trị vòng đời khách hàng (`customer_lifetime_value`)
  - **Marketing campaign data**: Chiến dịch chạy (`campaigns_run`), ngân sách quảng cáo (`ad_spend`), kênh tiếp thị (`campaign_channel`), tỷ lệ giảm giá (`discount_rate`), giá sau giảm (`discounted_price`), tỷ số chi phí quảng cáo/giá (`ad_spend_to_price_ratio`)
  - **Seasonal & holiday**: Tháng (`month`), quý (`quarter`), ngày trong tuần (`day_of_week`), cờ ngày lễ (`is_holiday`), mùa cao điểm mua sắm (`is_peak_shopping_season`), và các đặc trưng trễ (`sales_lag_1`, `sales_rolling_mean_7`)

### Biến mục tiêu (Target Variable)
- **Product sales** — biến số liên tục (numerical), phản ánh tổng doanh số hoặc lượng tiêu thụ sản phẩm cần dự báo.

### Đầu ra mong muốn
- Giá trị dự đoán sales cho từng sản phẩm
- Bảng so sánh hiệu suất giữa các mô hình hồi quy (RMSE, MAE, MSE, R²)
- Biểu đồ trực quan chuẩn xuất bản (Actual vs Predicted, Residuals Plot, Feature Importance, Model Comparison)
- Dự đoán theo 3 kịch bản kinh doanh thực tế (New Product Launch, Marketing Campaign, Seasonal Promotion)

---

## 2. Xác định yêu cầu

### Yêu cầu chức năng
| # | Yêu cầu | Mô tả |
|---|---------|-------|
| F1 | Tải dữ liệu | Tải dataset e-commerce từ Kaggle hoặc file local |
| F2 | Tiền xử lý | Xử lý missing values, outliers, encode categorical, chia train/test, scale |
| F3 | Feature Engineering | Tạo features mới: product popularity, CLV, seasonal trends, purchase frequency |
| F4 | Huấn luyện mô hình | Train 4 mô hình: Linear Regression, Decision Tree, Random Forest, Gradient Boosting |
| F5 | Đánh giá | Tính MSE, MAE, RMSE, R² |
| F6 | Hyperparameter Tuning | Tối ưu tham số bằng GridSearchCV |
| F7 | Phân tích lỗi | Phân tích residuals, tìm mẫu có sai số lớn nhất |
| F8 | Trực quan hóa | EDA, Actual vs Predicted, Residuals, Feature Importance, so sánh mô hình |
| F9 | Deployment | Lưu model, dự đoán sales cho sản phẩm mới, dự đoán theo kịch bản |

### Yêu cầu phi chức năng
- Code có cấu trúc module rõ ràng, dễ bảo trì
- Logging theo dõi quá trình chạy
- Kết quả (biểu đồ, bảng) được lưu ra thư mục `results/`
- Có file `requirements.txt` để tái tạo môi trường

---

## 3. Phân tích nhiệm vụ từng module

### 3.1. `config.py` — Cấu hình project
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| Đường dẫn dữ liệu | `RAW_DATA_FILE = data/raw/olist_customers_dataset.csv`, `PROCESSED_DATA_FILE = data/processed/cleaned_sales_data.csv`, thư mục `results/` |
| Tên features | Định nghĩa đầy đủ 5 nhóm features: Product, Customer Demographics, Purchase/CLV, Marketing, Seasonal & Lag |
| Tham số chung | Tỷ lệ chia 8:1:1 (`train_size=0.8`, `val_size=0.1`, `test_size=0.1`), `random_state=42`, `cv_folds=5`, `USE_TEMPORAL_SPLIT=True` |
| Hyperparameter grids | Dict tham số GridSearchCV cho Decision Tree, Random Forest, Gradient Boosting, XGBoost |

### 3.2. `data/loader.py` — Tải dữ liệu
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `load_data(filepath)` | Nạp dataset khách hàng Olist E-commerce từ `data/raw/olist_customers_dataset.csv` (hỗ trợ fallback utf-8, latin-1), trả về DataFrame |
| `explore_data(df)` | In shape, kiểu dữ liệu, tỷ lệ missing values, phân bố địa lý theo bang/thành phố, thống kê mô tả |

### 3.3. `data/preprocessing.py` — Tiền xử lý dữ liệu
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `handle_missing_values(df)` | Xử lý giá trị thiếu bằng imputation (mean/median) hoặc drop |
| `handle_outliers(df, columns)` | Phát hiện outliers bằng IQR hoặc Z-score, xử lý (clip/remove) |
| `encode_categorical(df, columns)` | Chuyển biến categorical → numerical (One-Hot Encoding / Ordinal) |
| `split_data(df, target, train_size=0.8, val_size=0.1, test_size=0.1, temporal=True)` | Phân chia 3 tập **Train / Val / Test (8:1:1)** hỗ trợ cả Temporal Split (chống rò rỉ dữ liệu) và Random Split |
| `scale_features(X_train, X_val, X_test)` / `build_preprocessor_pipeline()` | Chuẩn hóa số và mã hóa categorical bằng `ColumnTransformer`, fit trên Train (80%) và transform trên Val (10%), Test (10%) |

### 3.4. `data/feature_engineering.py` — Tạo features mới
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `add_product_popularity(df)` | Tạo feature popularity dựa trên số lượng reviews, ratings |
| `add_customer_lifetime_value(df)` | Tính Customer Lifetime Value (CLV) từ lịch sử mua hàng |
| `add_seasonal_trends(df)` | Thêm features mùa vụ: month, quarter, day_of_week, is_holiday, is_weekend |
| `add_lag_and_rolling_features(df)` | Thêm lag sales (doanh số chu kỳ trước) và rolling mean (trung bình trượt 7 ngày/30 ngày) |
| `add_purchase_frequency(df)` | Tính tần suất mua hàng của khách |
| `add_marketing_features(df)` | Tính ROI marketing, tỷ lệ ad spend / revenue |

### 3.5. `models/regressors.py` — Các mô hình hồi quy
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `get_models()` | Trả về dict `{name: model}` cho Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| `train_and_evaluate(models, X_train, X_test, y_train, y_test)` | Huấn luyện từng model, tính metrics (MSE, MAE, RMSE, R²) |

### 3.6. `models/tuning.py` — Tối ưu Hyperparameter
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `tune_decision_tree(X, y)` | GridSearchCV cho Decision Tree: `max_depth`, `min_samples_split`, `min_samples_leaf` |
| `tune_random_forest(X, y)` | GridSearchCV cho Random Forest: `n_estimators`, `max_depth`, `max_features` |
| `tune_gradient_boosting(X, y)` | GridSearchCV cho Gradient Boosting: `n_estimators`, `learning_rate`, `max_depth` |
| `tune_all(X, y)` | Gọi tất cả, trả về dict `{name: best_estimator}` |

### 3.7. `models/persistence.py` — Lưu/tải model
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `save_model(model, scaler, path)` | Serialize model + scaler bằng joblib |
| `load_model(path)` | Deserialize model đã lưu |
| `save_pipeline(pipeline, path)` | Lưu toàn bộ pipeline (scaler → encoder → model) |
| `load_pipeline(path)` | Tải pipeline đã lưu |

### 3.8. `evaluation/metrics.py` — Tính toán metrics hồi quy
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `compute_metrics(y_test, y_pred)` | Tính MSE, MAE, RMSE, R² |
| `print_regression_reports(results)` | In báo cáo metrics cho từng mô hình |
| `build_summary_table(all_results)` | Tạo DataFrame tổng hợp, sắp xếp theo RMSE |

### 3.9. `evaluation/visualization.py` — Trực quan hóa
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `plot_eda(df)` | EDA: phân bố sales, correlation heatmap, boxplot outliers |
| `plot_actual_vs_predicted(results, y_test)` | Scatter plot Actual vs Predicted cho từng model |
| `plot_residuals(results, y_test)` | Biểu đồ residuals (residual vs predicted, histogram) |
| `plot_feature_importance(model, names)` | Bar chart Feature Importance (Random Forest / Gradient Boosting) |
| `plot_model_comparison(summary_df)` | Biểu đồ cột grouped so sánh MSE/MAE/RMSE/R² |
| `plot_seasonal_trends(df)` | Biểu đồ xu hướng sales theo mùa (line chart theo month/quarter) |

### 3.10. `evaluation/error_analysis.py` — Phân tích lỗi dự đoán
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `analyze_residuals(X_test, y_test, y_pred, names)` | Phân tích residuals theo từng feature |
| `find_worst_predictions(X_test, y_test, y_pred, n)` | Tìm n mẫu có sai số lớn nhất, xem đặc trưng |
| `plot_error_distribution(y_test, y_pred)` | Histogram phân bố sai số |
| `compare_error_patterns(results, X_test, y_test)` | So sánh pattern lỗi giữa các mô hình |

### 3.11. `deployment/predictor.py` — Triển khai dự đoán
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `predict_sales(features, model, scaler)` | Nhận vector features → trả về giá trị sales dự đoán |
| `demo_prediction(model, scaler, X_test, y_test)` | Lấy ngẫu nhiên 1 mẫu test, dự đoán và so sánh với actual |
| `predict_scenarios(model, scaler, scenarios)` | Dự đoán nhiều kịch bản (new product, marketing campaign, seasonal promo) |

### 3.12. `utils/logger.py` — Logging và tiện ích
| Nhiệm vụ | Chi tiết |
|-----------|----------|
| `setup_logger(name, log_file)` | Tạo logger ghi ra console + file |
| `log_step(step_name)` | Decorator đo thời gian chạy từng bước |
| `print_section_header(title)` | In header đẹp phân cách các section |

### 3.13. `main.py` — Entry point
| Bước | Gọi module | Mô tả |
|------|-----------|-------|
| 1 | `data.loader` | Tải + khám phá dữ liệu |
| 2 | `data.preprocessing` | Xử lý missing, outliers, encode, split, scale |
| 3 | `data.feature_engineering` | Tạo features mới (popularity, CLV, seasonal, ...) |
| 4 | `evaluation.visualization` | EDA biểu đồ |
| 5 | `models.regressors` | Huấn luyện 4 mô hình hồi quy |
| 6 | `evaluation.metrics` | Đánh giá metrics |
| 7 | `models.tuning` | Tối ưu hyperparameter |
| 8 | `evaluation.error_analysis` | Phân tích lỗi dự đoán |
| 9 | `evaluation.visualization` | Biểu đồ so sánh tổng hợp |
| 10 | `models.persistence` | Lưu model tốt nhất |
| 11 | `deployment.predictor` | Demo dự đoán sales + kịch bản |

---

## Task 1 — Weekly Product Sales Data

The first two notebooks implement the Task 1 workflow described in `docs/TASK_2.pdf`:

- Build a Monday-start Product–Week panel from the six local Olist source tables. Each product is represented from its first delivered sale through the final observed week, including weeks with zero revenue.
- Create lagged sales, four-week rolling statistics, product age, prior-only review/customer-history features, and calendar features. Reviews are grouped by their creation week and are not available to a prediction until a later week.
- Split by complete calendar weeks in chronological order (80% train, 10% validation, 10% test). Missing-value imputation, Z-score scaling, and one-hot category encoding are fit on training data only. Revenue is never capped.
- The model matrix has 16 numeric predictors and 73 known Olist category indicators (89 predictors total). Categories not in the training/catalog vocabulary are handled as all-zero one-hot vectors.
- `01_exploratory_data_analysis.ipynb` saves the eight-chart EDA to `notebooks/results/figures/`. `02_preprocessing_and_feature_engineering.ipynb` saves the panels to `data/processed/`, sparse feature matrices and targets to `data/processed/`, and the fitted preprocessor to `notebooks/results/saved_models/`.

The required source files are the six Olist CSVs in `data/raw/`. Run the first two notebooks from top to bottom, then continue with the model-training notebooks as appropriate.

---

## 4. Công nghệ sử dụng

| Thư viện | Mục đích |
|----------|----------|
| `pandas` | Xử lý dữ liệu dạng bảng |
| `numpy` | Tính toán số học |
| `scikit-learn` | Mô hình ML, tiền xử lý, pipeline, đánh giá |
| `xgboost` / `lightgbm` | Thuật toán Gradient Boosting hiệu năng cao |
| `matplotlib` | Vẽ biểu đồ cơ bản |
| `seaborn` | Biểu đồ thống kê nâng cao |
| `joblib` | Lưu/tải model và pipeline |
| `streamlit` | Giao diện Web demo trực quan cho dự đoán theo kịch bản |

---

## 5. Cấu trúc thư mục dự án

```text
sales_predictor/
├── main.py                         # Entry point, điều phối toàn bộ workflow
├── requirements.txt                # Thư viện phụ thuộc
├── README.md                       # Tài liệu phân tích và thiết kế bài toán
├── .gitignore                      # Bảo vệ dữ liệu và models
├── data/                           # Thư mục lưu trữ dữ liệu thực tế (được .gitignore)
│   ├── raw/                        # File CSV gốc tải về từ Kaggle
│   │   └── olist_customers_dataset.csv # Dataset khách hàng E-commerce thực tế Olist (~99,442 dòng)
│   └── processed/                  # File dữ liệu sau khi làm sạch và thêm features
│       └── cleaned_sales_data.csv  # Dữ liệu sạch kèm đầy đủ 5 nhóm đặc trưng
├── notebooks/                      # Thử nghiệm tương tác và báo cáo từng bước
│   ├── 01_exploratory_data_analysis.ipynb              # 1. EDA, phân bố target, tương quan
│   ├── 02_preprocessing_and_feature_engineering.ipynb # 2. Làm sạch & tạo đặc trưng
│   ├── 03_model_training_and_comparison.ipynb         # 3. Train 4 mô hình & so sánh metrics
│   ├── 04_hyperparameter_tuning_and_error_analysis.ipynb # 4. Tối ưu GridSearchCV & phân tích lỗi
│   └── 05_scenario_prediction_and_demo.ipynb          # 5. Dự đoán theo kịch bản (Deployment)
├── results/                        # Toàn bộ output kết quả huấn luyện
│   ├── figures/                    # Biểu đồ trực quan hóa (.png)
│   ├── metrics/                    # Bảng so sánh chỉ số đánh giá (.csv)
│   └── saved_models/               # File model và pipeline đã lưu (.joblib)
└── src/                            # Toàn bộ mã nguồn dự án
    ├── __init__.py
    ├── config.py                   # Cấu hình tập trung (paths, features, hyperparameter grids)
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py               # Tải dữ liệu từ CSV/Kaggle
    │   ├── preprocessing.py        # Tiền xử lý (missing, outlier, encode, temporal/random split)
    │   └── feature_engineering.py  # Tạo features (popularity, CLV, seasonal, lag & rolling)
    ├── models/
    │   ├── __init__.py
    │   ├── regressors.py           # Các mô hình: LR, DT, RF, Gradient Boosting, XGBoost
    │   ├── tuning.py               # GridSearchCV hyperparameter tuning
    │   └── persistence.py          # Lưu/tải toàn bộ Pipeline (preprocessor + model)
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py              # MSE, MAE, RMSE, R²
    │   ├── visualization.py        # EDA, Actual vs Predicted, Residuals, Feature Importance
    │   └── error_analysis.py       # Phân tích các trường hợp sai lệch lớn nhất (worst cases)
    ├── deployment/
    │   ├── __init__.py
    │   └── predictor.py            # Dự đoán sales theo kịch bản (new product, marketing)
    └── utils/
        ├── __init__.py
        └── logger.py               # Logging theo dõi và decorator đo thời gian chạy
└── tests/                          # Bộ kiểm thử tự động (Unit Tests với Pytest)
    ├── __init__.py
    ├── conftest.py                 # Fixtures dữ liệu mẫu
    ├── test_preprocessing.py       # Kiểm thử làm sạch & split dữ liệu
    ├── test_feature_engineering.py # Kiểm thử tạo 5 nhóm đặc trưng
    ├── test_models.py              # Kiểm thử mô hình, tuning & lưu pipeline
    └── test_deployment.py          # Kiểm thử dự đoán theo kịch bản
```

---

## 6. Các điểm cốt lõi về kỹ thuật (Best Practices)

1. **Phòng tránh Data Leakage (Rò rỉ dữ liệu tương lai):**
   - Dữ liệu bán hàng e-commerce mang tính chất thời gian. Dự án hỗ trợ **Temporal Split** (huấn luyện trên các tháng trước, kiểm tra trên các tháng sau) thay vì chỉ random split thông thường.
   - Các phép tính thống kê (mean, standard scaler, frequency encoder) chỉ được `fit` trên tập Train và `transform` trên tập Test/Inference.
2. **End-to-End Pipeline Serialization:**
   - Khi lưu mô hình (`persistence.py`), lưu cả bộ tiền xử lý (`ColumnTransformer` / `Pipeline`) đi kèm mô hình. Nhờ đó khi có dữ liệu mới ở phần Deployment, dữ liệu tự động được scale và encode tương thích hoàn toàn mà không sợ lệch số chiều.
3. **Phân tích lỗi (Error Analysis):**
   - Không chỉ dừng ở báo cáo MSE/RMSE, module `error_analysis.py` tự động trích xuất các sản phẩm bị dự đoán lệch nhiều nhất để rút kinh nghiệm và tinh chỉnh features.

---

## 7. Hướng dẫn chạy dự án (Quickstart)

### 7.1. Cài đặt môi trường
Khởi tạo môi trường ảo và cài đặt các thư viện cần thiết:
```bash
python -m venv .venv
# Trên Windows PowerShell:
.venv\Scripts\Activate.ps1
# Hoặc trên Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 7.2. Chạy toàn bộ quy trình tự động (Pipeline End-to-End)
Thực thi toàn bộ luồng từ nạp dữ liệu Olist, tiền xử lý, huấn luyện 4 mô hình, đánh giá, phân tích lỗi đến dự đoán 3 kịch bản:
```bash
python main.py
```

### 7.3. Chạy kiểm thử tự động (Unit Tests)
```bash
pytest -v
```

### 7.4. Thử nghiệm tương tác trên Jupyter Notebooks
Mở và chạy lần lượt 5 notebook trong thư mục `notebooks/`:
1. `01_exploratory_data_analysis.ipynb`: Xây dựng/đọc Product–Week panel và tạo báo cáo EDA 8 biểu đồ
2. `02_preprocessing_and_feature_engineering.ipynb`: Kỹ nghệ đặc trưng chuỗi thời gian, temporal split và tiền xử lý fit trên train
3. `03_model_training_and_comparison.ipynb`: Huấn luyện và so sánh 4 mô hình hồi quy
4. `04_hyperparameter_tuning_and_error_analysis.ipynb`: Tối ưu siêu tham số GridSearchCV & đồ thị phần dư
5. `05_scenario_prediction_and_demo.ipynb`: Demo dự đoán 3 kịch bản kinh doanh thực tế (New Product, Marketing, Seasonal Promo)

